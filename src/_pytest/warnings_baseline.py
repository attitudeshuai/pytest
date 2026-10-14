# mypy: allow-untyped-defs
"""Warnings baseline support.

A warnings baseline is a persistent registry of the warnings captured during a
test run. Every captured warning is registered by the emitting test node id,
warning category, code location (file and line) and message text.

On subsequent runs the captured warnings are compared against the baseline:

* warnings that are not in the baseline are *new* and are either annotated in
  the terminal summary or turn the run into a failure
  (``--warnings-baseline-on-new=warn|fail``);
* baseline entries that are no longer observed are reported as candidates to
  be written off.

The baseline is registered with one of two explicit, all-or-nothing entry
points:

* ``--warnings-baseline-update`` atomically rewrites the whole baseline from
  the warnings of the current run (entries that disappeared are written off);
* ``--warnings-baseline-merge`` atomically merges the warnings of the current
  run into the existing baseline (entries not observed in this run are kept),
  which allows registering a baseline across several partial or sharded runs.

When no baseline is configured, this module is inert and the warnings capture,
summary and filtering behaviour is unchanged. The baseline never alters
warning filters; it is a layer on top of the warnings pytest already captures.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import datetime
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Literal
from typing import TYPE_CHECKING
import warnings

from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import UsageError
import pytest


if TYPE_CHECKING:
    from _pytest.main import Session
    from _pytest.terminal import TerminalReporter


#: Schema version of baseline files written by this pytest. Only the major
#: component is relevant for compatibility: files with a different major
#: version are rejected instead of being silently misinterpreted.
SCHEMA_VERSION = "1.0"

_SCHEMA_MAJOR = SCHEMA_VERSION.split(".", 1)[0]

BaselineMode = Literal["compare", "update", "merge"]
NewWarningAction = Literal["warn", "fail"]

_MODE_UPDATE: Literal["update"] = "update"
_MODE_MERGE: Literal["merge"] = "merge"

#: Exit statuses for which a baseline must never be (re)written: the run did
#: not execute normally, so the captured warning set must not be mistaken for
#: the full picture (in particular, existing entries must not be written off).
_NO_WRITE_STATUSES = frozenset(
    {
        ExitCode.TESTS_FAILED,
        ExitCode.INTERRUPTED,
        ExitCode.INTERNAL_ERROR,
        ExitCode.USAGE_ERROR,
        ExitCode.NO_TESTS_COLLECTED,
    }
)


class BaselineError(ValueError):
    """Raised when a warnings baseline file is malformed or unsupported."""


@dataclass(frozen=True, order=True)
class BaselineEntry:
    """A single warning recorded in the baseline.

    The identity of a warning is the tuple of all five fields: the warning
    belongs to a test node (``nodeid`` is the empty string for configuration
    and collection warnings), has a ``category`` and originates from a code
    location (``filename`` and ``lineno``); the ``message`` text is part of
    the identity so that different warnings emitted from the same line remain
    distinguishable.
    """

    nodeid: str
    category: str
    filename: str
    lineno: int
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "nodeid": self.nodeid,
            "category": self.category,
            "filename": self.filename,
            "lineno": self.lineno,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: object) -> BaselineEntry:
        if not isinstance(data, dict):
            raise BaselineError("entry must be an object")
        strings = {}
        for field_name in ("nodeid", "category", "filename", "message"):
            value = data.get(field_name)
            if not isinstance(value, str):
                raise BaselineError(f"{field_name!r} must be a string")
            strings[field_name] = value
        lineno = data.get("lineno")
        # bool is a subclass of int; reject it explicitly.
        if isinstance(lineno, bool) or not isinstance(lineno, int) or lineno < 0:
            raise BaselineError("'lineno' must be a non-negative integer")
        if not strings["category"]:
            raise BaselineError("'category' must not be empty")
        if not strings["filename"]:
            raise BaselineError("'filename' must not be empty")
        return cls(
            nodeid=strings["nodeid"],
            category=strings["category"],
            filename=strings["filename"],
            lineno=lineno,
            message=strings["message"],
        )


@dataclass(frozen=True)
class BaselineDiff:
    """Result of comparing the warnings captured in a run with a baseline."""

    captured: list[BaselineEntry]
    baseline: list[BaselineEntry]
    #: Warnings captured in this run but absent from the baseline.
    new: list[BaselineEntry]
    #: Baseline entries not observed in this run (write-off candidates).
    missing: list[BaselineEntry]

    @property
    def matched_count(self) -> int:
        return len(self.captured) - len(self.new)


def _category_name(category: type[Warning]) -> str:
    # Built-in warning categories live in the ``builtins`` module; storing
    # just their name keeps the file readable while non-builtin categories
    # are stored with their importable dotted path for disambiguation.
    if category.__module__ == "builtins":
        return category.__name__
    return f"{category.__module__}.{category.__qualname__}"


def _relative_filename(filename: str, rootpath: Path) -> str:
    """Return ``filename`` relative to ``rootpath`` when it is inside it.

    Storing root-relative, POSIX-style paths makes baselines portable across
    checkouts and environments. Locations outside of the root directory (for
    example warnings emitted from installed third-party packages) are stored
    as given.
    """
    path = Path(os.path.normpath(filename))
    candidates = [(path, rootpath)]
    try:
        candidates.append((path.resolve(), rootpath.resolve()))
    except OSError:
        pass
    for candidate, root in candidates:
        if candidate.is_absolute() and root.is_absolute():
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                continue
    return path.as_posix()


def entry_from_warning(
    warning_message: warnings.WarningMessage,
    nodeid: str,
    rootpath: Path,
) -> BaselineEntry:
    """Build a :class:`BaselineEntry` from a captured ``WarningMessage``."""
    return BaselineEntry(
        nodeid=nodeid or "",
        category=_category_name(warning_message.category),
        filename=_relative_filename(warning_message.filename, rootpath),
        lineno=int(warning_message.lineno),
        message=str(warning_message.message),
    )


def current_environment() -> dict[str, str]:
    """Return the generation-environment marker for a newly written baseline."""
    return {
        "pytest": pytest.__version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
    }


def _utcnow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_baseline(
    entries: set[BaselineEntry] | list[BaselineEntry] | tuple[BaselineEntry, ...],
    *,
    generated_at: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build the serializable baseline structure from warning entries."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or _utcnow_iso(),
        "generator": environment or current_environment(),
        "warnings": [entry.to_dict() for entry in sorted(set(entries))],
    }


def dumps_baseline(baseline: dict[str, object]) -> str:
    """Serialize a baseline structure to text."""
    return json.dumps(baseline, indent=2, ensure_ascii=False) + "\n"


def parse_baseline(
    text: str,
) -> tuple[list[BaselineEntry], dict[str, str] | None]:
    """Parse baseline text.

    Returns the recorded warning entries and the generation-environment
    marker. Raises :class:`BaselineError` on any structural problem.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BaselineError("top-level value must be a JSON object")

    version = data.get("schema_version")
    if not isinstance(version, str):
        raise BaselineError("missing or invalid 'schema_version' field")
    if version.split(".", 1)[0] != _SCHEMA_MAJOR:
        raise BaselineError(
            f"unsupported baseline schema_version {version!r}; this pytest "
            f"supports {_SCHEMA_MAJOR}.x baselines"
        )

    raw_entries = data.get("warnings")
    if not isinstance(raw_entries, list):
        raise BaselineError("'warnings' field must be a list")
    entries: list[BaselineEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        try:
            entries.append(BaselineEntry.from_dict(raw_entry))
        except BaselineError as exc:
            raise BaselineError(f"warnings[{index}]: {exc}") from exc

    raw_environment = data.get("generator")
    environment: dict[str, str] | None = None
    if raw_environment is not None:
        if not isinstance(raw_environment, dict):
            raise BaselineError("'generator' field must be an object")
        environment = {str(key): str(value) for key, value in raw_environment.items()}

    return entries, environment


def diff_baseline(
    captured: set[BaselineEntry] | list[BaselineEntry],
    baseline: set[BaselineEntry] | list[BaselineEntry],
) -> BaselineDiff:
    """Compare captured warnings with baseline entries."""
    captured_set = set(captured)
    baseline_set = set(baseline)
    return BaselineDiff(
        captured=sorted(captured_set),
        baseline=sorted(baseline_set),
        new=sorted(captured_set - baseline_set),
        missing=sorted(baseline_set - captured_set),
    )


def write_baseline_atomic(path: Path, baseline: dict[str, object]) -> None:
    """Write a baseline file atomically.

    The content is first written to a temporary file in the same directory
    and then moved into place with :func:`os.replace`, so the baseline is
    either fully updated or not changed at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_baseline(baseline)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _describe_environment(environment: dict[str, str] | None) -> str:
    if not environment:
        return "unknown environment"
    implementation = environment.get("python_implementation", "Python")
    return ", ".join(
        (
            f"pytest {environment.get('pytest', '?')}",
            f"{implementation} {environment.get('python', '?')}",
            str(environment.get("platform", "?")),
        )
    )


def environment_mismatch_note(
    current: dict[str, str],
    recorded: dict[str, str] | None,
) -> str | None:
    """Return a note when the baseline was generated in another environment."""
    if not recorded:
        return None
    compared = ("pytest", "python", "python_implementation", "platform")
    if all(recorded.get(key) == current.get(key) for key in compared):
        return None
    return (
        "note: baseline was generated in a different environment "
        f"({_describe_environment(recorded)}); current environment: "
        f"{_describe_environment(current)}"
    )


def _plural(count: int, word: str, plural: str | None = None) -> str:
    if count == 1:
        return f"{count} {word}"
    return f"{count} {plural if plural is not None else word + 's'}"


def format_entry_lines(entry: BaselineEntry) -> list[str]:
    """Render a baseline entry in the style of the warnings summary."""
    message = entry.message.replace("\n", "\n  ")
    lines = []
    if entry.nodeid:
        lines.append(entry.nodeid)
    lines.append(f"  {entry.filename}:{entry.lineno}: {entry.category}: {message}")
    return lines


class WarningsBaselinePlugin:
    """Collect warnings for a baseline and compare/update at session end."""

    def __init__(
        self,
        config: Config,
        *,
        path: Path,
        mode: BaselineMode,
        on_new: NewWarningAction,
        baseline_entries: list[BaselineEntry],
        baseline_environment: dict[str, str] | None,
    ) -> None:
        self.config = config
        self.path = path
        self.mode: BaselineMode = mode
        self.on_new: NewWarningAction = on_new
        self._baseline_entries = list(baseline_entries)
        self.baseline_environment = baseline_environment
        self._captured: set[BaselineEntry] = set()
        self.diff: BaselineDiff | None = None
        self.written = False
        self.write_skipped_reason: str | None = None
        self.update_added = 0
        self.update_removed = 0
        self.merge_added = 0
        self.failure_triggered = False

    #
    # Warning collection
    #
    @pytest.hookimpl
    def pytest_warning_recorded(
        self,
        warning_message: warnings.WarningMessage,
        nodeid: str,
        when: str,
    ) -> None:
        self._captured.add(
            entry_from_warning(warning_message, nodeid, self.config.rootpath)
        )

    #
    # Comparison and baseline updates
    #
    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(
        self,
        session: Session,
        exitstatus: int | ExitCode,
    ) -> None:
        captured = sorted(self._captured)
        old = sorted(self._baseline_entries)
        captured_set = set(captured)
        old_set = set(old)
        self.diff = diff_baseline(captured, old)

        if self.mode in (_MODE_UPDATE, _MODE_MERGE):
            if session.exitstatus in _NO_WRITE_STATUSES:
                self.write_skipped_reason = (
                    "the session did not complete successfully; the baseline "
                    "was left unchanged on purpose"
                )
                return
            if self.mode == _MODE_MERGE:
                final_entries = sorted(old_set | captured_set)
                self.merge_added = len(captured_set - old_set)
            else:
                final_entries = captured
                self.update_added = len(captured_set - old_set)
                self.update_removed = len(old_set - captured_set)
            write_baseline_atomic(
                self.path,
                build_baseline(final_entries),
            )
            self.written = True
            self._baseline_entries = final_entries
            return

        if (
            self.on_new == "fail"
            and self.diff.new
            and session.exitstatus == ExitCode.OK
        ):
            session.exitstatus = ExitCode.WARNINGS_BASELINE_ERROR
            self.failure_triggered = True

    #
    # Terminal reporting
    #
    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(
        self,
        terminalreporter: TerminalReporter,
        exitstatus: int | ExitCode,
        config: Config,
    ) -> None:
        if self.write_skipped_reason is not None:
            terminalreporter.write_line(
                f"warnings baseline not changed: {self.write_skipped_reason}.",
                red=True,
            )
            return

        if self.written:
            self._write_update_status(terminalreporter)
            return

        section_visible = not terminalreporter.no_summary and terminalreporter.hasopt(
            "w"
        )
        if section_visible and self.diff is not None and self.mode == "compare":
            self._write_diff_section(terminalreporter)

        if self.failure_triggered:
            assert self.diff is not None
            terminalreporter.write_line(
                "Tests pass, but "
                f"{_plural(len(self.diff.new), 'warning')} "
                f"{'are' if len(self.diff.new) != 1 else 'is'} not recorded in "
                f"the warnings baseline: {self._display_path()}",
                red=True,
            )
            if not section_visible:
                terminalreporter.write_line(
                    "rerun with the warnings summary enabled to see the "
                    "warnings, or use --warnings-baseline-update to register "
                    "them in the baseline",
                    red=True,
                )

    def _write_update_status(self, terminalreporter: TerminalReporter) -> None:
        display_path = self._display_path()
        if self.mode == _MODE_MERGE:
            total = len(self._baseline_entries)
            terminalreporter.write_line(
                f"warnings baseline merged: "
                f"{_plural(self.merge_added, 'new entry', 'new entries')} "
                f"registered in {display_path}; "
                f"{_plural(total, 'entry', 'entries')} in total"
            )
        else:
            total = len(self._baseline_entries)
            changes = []
            if self.update_added:
                changes.append(f"{self.update_added} added")
            if self.update_removed:
                changes.append(f"{self.update_removed} written off")
            change_text = ", ".join(changes) if changes else "no changes"
            terminalreporter.write_line(
                f"warnings baseline updated: {display_path} now contains "
                f"{_plural(total, 'entry', 'entries')} ({change_text})"
            )

    def _write_diff_section(self, terminalreporter: TerminalReporter) -> None:
        assert self.diff is not None
        terminalreporter.write_sep("=", "warnings baseline", yellow=True, bold=False)
        terminalreporter.write_line(
            f"baseline: {self._display_path()} (schema {SCHEMA_VERSION})"
        )
        mismatch = environment_mismatch_note(
            current_environment(), self.baseline_environment
        )
        if mismatch is not None:
            terminalreporter.write_line(mismatch, yellow=True)

        new = self.diff.new
        missing = self.diff.missing
        if not new and not missing:
            if self.diff.matched_count == 0 and not self.diff.baseline:
                terminalreporter.write_line(
                    "no warnings captured and the baseline is empty"
                )
                return
            matched = _plural(self.diff.matched_count, "captured warning")
            total = _plural(
                len(self.diff.baseline), "baseline entry", "baseline entries"
            )
            terminalreporter.write_line(
                f"all {matched} match the {total}; nothing new to report"
            )
            return

        if new:
            terminalreporter.write_line("")
            terminalreporter.write_line(
                f"new warnings not in baseline ({_plural(len(new), 'warning')})"
                " -- not filtered, only reported:"
            )
            for entry in new:
                for line in format_entry_lines(entry):
                    terminalreporter.write_line(line)

        if missing:
            terminalreporter.write_line("")
            terminalreporter.write_line(
                f"baseline warnings not observed this run "
                f"({_plural(len(missing), 'entry', 'entries')}); rerun with "
                "--warnings-baseline-update to write them off, or with "
                "--warnings-baseline-merge to keep them:"
            )
            for entry in missing:
                for line in format_entry_lines(entry):
                    terminalreporter.write_line(line)

    def _display_path(self) -> str:
        try:
            return os.path.relpath(self.path, self.config.invocation_params.dir)
        except ValueError:
            # Different drives on Windows.
            return str(self.path)


def configure_warnings_baseline(config: Config) -> None:
    """Register the baseline plugin when a baseline path is configured.

    Does nothing (and changes nothing) when no baseline is configured.
    Raises :class:`~pytest.UsageError` for conflicting options, an unreadable
    or malformed baseline, or a missing baseline in comparison mode.
    """
    option = config.option
    update = bool(option.warnings_baseline_update)
    merge = bool(option.warnings_baseline_merge)
    if update and merge:
        raise UsageError(
            "--warnings-baseline-update and --warnings-baseline-merge cannot "
            "be combined"
        )

    cli_path = option.warnings_baseline
    path: Path | None = None
    if cli_path:
        path = Path(str(cli_path))
        if not path.is_absolute():
            path = config.invocation_params.dir / path
    else:
        ini_paths = config.getini("warnings_baseline")
        if ini_paths:
            path = Path(ini_paths[0])

    if path is None:
        if update or merge:
            flag = (
                "--warnings-baseline-update" if update else "--warnings-baseline-merge"
            )
            raise UsageError(
                f"{flag} requires a baseline path: pass "
                "--warnings-baseline=PATH or set the 'warnings_baseline' "
                "configuration option"
            )
        return

    path = Path(os.path.abspath(path))
    mode: BaselineMode = _MODE_UPDATE if update else _MODE_MERGE if merge else "compare"
    on_new: NewWarningAction = (
        option.warnings_baseline_on_new
        or config.getini("warnings_baseline_on_new")
        or "warn"
    )
    if on_new not in ("warn", "fail"):
        raise UsageError(
            f"invalid value {on_new!r} for warnings_baseline_on_new: "
            "expected 'warn' or 'fail'"
        )

    entries: list[BaselineEntry] = []
    environment: dict[str, str] | None = None
    if path.exists():
        if not path.is_file():
            raise UsageError(f"warnings baseline path is not a file: {path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise UsageError(f"could not read warnings baseline {path}: {exc}") from exc
        try:
            entries, environment = parse_baseline(text)
        except BaselineError as exc:
            raise UsageError(
                f"could not parse warnings baseline {path}: {exc}"
            ) from exc
    elif mode == "compare":
        raise UsageError(
            f"warnings baseline file not found: {path}\n"
            "run with --warnings-baseline-update to create it from the "
            "warnings captured in this run"
        )

    plugin = WarningsBaselinePlugin(
        config,
        path=path,
        mode=mode,
        on_new=on_new,
        baseline_entries=entries,
        baseline_environment=environment,
    )
    config.pluginmanager.register(plugin, "warnings-baseline-plugin")
