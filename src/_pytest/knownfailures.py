"""Known-failures baseline governance.

A manifest file, independent from the test code, registers test node ids which
are known to fail, together with a reason and an expiry date. Failures of
registered tests are executed and reported as ordinary failures, but they do
not turn the session's exit code red while their exemption is active.

Expired exemptions stop suppressing failures and are reported prominently, as
are exempted tests which unexpectedly pass (so the stale entries can be
removed). Entries of tests that now pass can be written off automatically with
``--known-failures-prune``.

When no manifest is configured, this plugin does nothing and pytest's
judgement and exit code semantics are unchanged.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import OverrideIniAction
from _pytest.config.argparsing import Parser
from _pytest.reports import TestReport


if TYPE_CHECKING:
    from _pytest.main import Session
    from _pytest.terminal import TerminalReporter

_MANIFEST_VERSION = 1
_ALLOWED_TOP_LEVEL_KEYS = {"version", "known_failures"}
_ALLOWED_ENTRY_KEYS = {"nodeid", "reason", "expires"}


@dataclasses.dataclass(frozen=True)
class KnownFailureEntry:
    """A single exempted test node, with governance metadata."""

    #: Full test node id, e.g. ``test_x.py::test_y`` or
    #: ``test_x.py::test_y[param]``.
    nodeid: str
    #: Why this failure is tolerated, e.g. a ticket reference.
    reason: str
    #: Last day on which the exemption is still active.
    expires: datetime.date

    def is_expired(self, today: datetime.date) -> bool:
        return self.expires < today


def parse_manifest(
    data: object,
) -> tuple[list[KnownFailureEntry], list[str]]:
    """Parse and validate a decoded manifest document.

    :returns: ``(entries, errors)``; every structurally valid entry is
        returned, even when other entries or the document itself contain
        errors. ``errors`` is a list of human-readable problems.
    """
    errors: list[str] = []
    entries: list[KnownFailureEntry] = []

    if not isinstance(data, dict):
        return entries, ["manifest must be a JSON object with a 'known_failures' list"]

    for key in sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS):
        errors.append(f"unknown manifest key: {key!r}")

    version = data.get("version", _MANIFEST_VERSION)
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _MANIFEST_VERSION
    ):
        errors.append("manifest 'version' must be the integer 1")

    raw_entries = data.get("known_failures")
    if not isinstance(raw_entries, list):
        errors.append("manifest 'known_failures' must be a list of entry objects")
        return entries, errors

    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        location = f"known_failures[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{location}: entry must be a JSON object")
            continue

        for key in raw_entry:
            if key not in _ALLOWED_ENTRY_KEYS:
                errors.append(f"{location}: unknown key {key!r}")

        nodeid = raw_entry.get("nodeid")
        if not isinstance(nodeid, str) or not nodeid.strip():
            errors.append(f"{location}: 'nodeid' must be a non-empty string")
            nodeid = None

        reason = raw_entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{location}: 'reason' must be a non-empty string")

        expires_raw = raw_entry.get("expires")
        expires: datetime.date | None = None
        if not isinstance(expires_raw, str):
            errors.append(f"{location}: 'expires' must be a 'YYYY-MM-DD' date string")
        elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expires_raw):
            errors.append(
                f"{location}: 'expires' must be a 'YYYY-MM-DD' date "
                f"string, got {expires_raw!r}"
            )
        else:
            try:
                expires = datetime.date.fromisoformat(expires_raw)
            except ValueError:
                errors.append(
                    f"{location}: 'expires' is not a valid calendar date: "
                    f"{expires_raw!r}"
                )

        if nodeid is None:
            continue
        if nodeid in seen:
            errors.append(f"{location}: duplicate nodeid {nodeid!r}")
            continue
        seen.add(nodeid)
        if expires is not None and isinstance(reason, str) and reason.strip():
            entries.append(
                KnownFailureEntry(nodeid=nodeid, reason=reason, expires=expires)
            )

    return entries, errors


def load_manifest(
    path: Path,
) -> tuple[list[KnownFailureEntry], list[str], bool]:
    """Read a manifest file from disk.

    :returns: ``(entries, errors, has_version)``; ``has_version`` records
        whether the document carried an explicit ``version`` key, so that a
        pruned rewrite preserves the author's shape.
    """
    try:
        # utf-8-sig also accepts plain UTF-8 and tolerates a BOM, which
        # Windows editors tend to prepend.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [], [f"cannot read known-failures manifest {path}: {exc}"], False

    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            [],
            [
                f"known-failures manifest {path} is not valid JSON: "
                f"{exc.msg} (line {exc.lineno} column {exc.colno})"
            ],
            False,
        )

    entries, errors = parse_manifest(data)
    has_version = isinstance(data, dict) and "version" in data
    return entries, errors, has_version


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption(
        "--known-failures",
        metavar="FILE",
        dest="known_failures_file",
        default=None,
        help="Path to a JSON manifest of known (exempted) failing tests. "
        "Such tests are still executed and reported, but their failures do "
        "not make the exit code red until their entry expires.",
    )
    group.addoption(
        "--known-failures-strict",
        action=OverrideIniAction,
        ini_option="known_failures_strict",
        ini_value="true",
        help="Refuse to run when the known-failures manifest is invalid "
        "(bad format, duplicate entries or non-existent test node ids). "
        "Without this, problems are reported as warnings.",
    )
    group.addoption(
        "--known-failures-prune",
        action="store_true",
        dest="known_failures_prune",
        default=False,
        help="After the run, remove entries whose tests now pass from the "
        "known-failures manifest (write off fixed failures).",
    )
    parser.addini(
        "known_failures",
        "Path to a JSON manifest of known (exempted) failing tests; see "
        "--known-failures.",
        default="",
    )
    parser.addini(
        "known_failures_strict",
        "Refuse to run when the known-failures manifest is invalid, instead "
        "of reporting problems as warnings.",
        type="bool",
        default=False,
    )


def pytest_configure(config: Config) -> None:
    path = config.getoption("known_failures_file") or config.getini("known_failures")
    if path:
        plugin = KnownFailuresPlugin(config, Path(str(path)))
        config.pluginmanager.register(plugin, "known-failures-plugin")
        # The xdist hooks only exist as hookspecs when xdist is loaded,
        # either via its entry point ("xdist") or via ``-p xdist.plugin``.
        if config.pluginmanager.hasplugin("xdist") or config.pluginmanager.hasplugin(
            "xdist.plugin"
        ):
            config.pluginmanager.register(
                _XdistControllerHooks(plugin),
                "known-failures-xdist-hooks",
            )


class _XdistControllerHooks:
    """xdist-controller-only hooks, sharing the main plugin's state."""

    def __init__(self, plugin: KnownFailuresPlugin) -> None:
        self.plugin = plugin

    def pytest_xdist_setupnodes(self, config: Config, specs: object) -> None:
        # Remember how many worker collections to wait for before reconciling.
        try:
            self.plugin._xdist_expected_nodes = len(specs)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - defensive, specs is a sequence
            self.plugin._xdist_expected_nodes = 0

    def pytest_xdist_node_collection_finished(self, node: object, ids: object) -> None:
        # The controller never runs the ordinary collection hooks, so
        # aggregate the workers' node ids and reconcile once the last worker
        # has reported, before any test is scheduled.
        plugin = self.plugin
        if isinstance(ids, (list, tuple, set, frozenset)):
            plugin._xdist_collected.update(str(nodeid) for nodeid in ids)
        plugin._xdist_finished_nodes.add(id(node))
        if (
            plugin._xdist_expected_nodes
            and len(plugin._xdist_finished_nodes) >= plugin._xdist_expected_nodes
        ):
            plugin._reconcile(plugin._xdist_collected)


class KnownFailuresPlugin:
    def __init__(self, config: Config, path: Path) -> None:
        self.config = config
        if not path.is_absolute():
            path = config.invocation_params.dir / path
        self.manifest_path = path
        self.strict: bool = bool(config.getini("known_failures_strict"))
        self.prune: bool = bool(config.getoption("known_failures_prune"))
        # Fix "today" for the whole run for stable, reproducible behaviour.
        # Expiry is a local calendar date, not an instant.
        self.today = datetime.datetime.now().date()

        loaded_entries, self.hard_errors, self._has_version = load_manifest(
            self.manifest_path
        )
        self.entries: dict[str, KnownFailureEntry] = {
            entry.nodeid: entry for entry in loaded_entries
        }
        #: Entries matched against a collected test node id.
        self.matched: dict[str, KnownFailureEntry] = {}
        #: xdist controller bookkeeping for worker-collected node ids.
        self._xdist_expected_nodes = 0
        self._xdist_finished_nodes: set[int] = set()
        self._xdist_collected: set[str] = set()
        self._reconciled = False
        #: Validation problems only detectable after collection (loose mode).
        self.collection_warnings: list[str] = []
        #: node id -> entry, one entry per test whose failure was suppressed.
        self.suppressed: dict[str, KnownFailureEntry] = {}
        #: Node ids of exempted tests which failed while their entry had
        #: expired; their failure counts as a regular failure.
        self.expired_failures: set[str] = set()
        #: Entries whose test reported a clean passing call phase.
        self.unexpected_passes: list[KnownFailureEntry] = []
        #: Node ids with at least one failing phase (setup/call/teardown).
        self.failed_nodeids: set[str] = set()
        self.pruned_count = 0
        self._refused = False
        self.session: Session | None = None

        if self.hard_errors and self.strict:
            raise UsageError(
                "invalid known-failures manifest:\n" + "\n".join(self.hard_errors)
            )

    def pytest_sessionstart(self, session: Session) -> None:
        self.session = session

    def pytest_collection_finish(self, session: Session) -> None:
        # Do not second-guess node ids when collection itself failed; the
        # collection error is already reported and may explain missing ids.
        if session.testsfailed:
            self._reconciled = True
            return
        if hasattr(self.config, "workerinput"):
            # xdist worker: it only sees a slice of the tests, so strict
            # validation is the controller's job. Still build the local map
            # so known failures do not trigger worker-side stop conditions.
            self._reconcile(
                {item.nodeid for item in session.items}, enforce_strict=False
            )
        else:
            self._reconcile({item.nodeid for item in session.items})

    def _reconcile(self, collected: set[str], *, enforce_strict: bool = True) -> None:
        self._reconciled = True
        # In loose mode structurally valid entries are still enforced even
        # when the manifest has problems; only the offending entries are
        # called out. (In strict mode a problematic manifest never reaches
        # this point.)
        self.matched = {
            nodeid: entry
            for nodeid, entry in self.entries.items()
            if nodeid in collected
        }
        unknown = [nodeid for nodeid in self.entries if nodeid not in collected]
        if unknown:
            self.collection_warnings = [
                f"manifest entry does not match any collected test: {nodeid!r}"
                for nodeid in unknown
            ]
            if self.strict and enforce_strict:
                self._refused = True
                raise UsageError(
                    "invalid known-failures manifest:\n"
                    + "\n".join(self.collection_warnings)
                )

    @hookimpl(wrapper=True)
    def pytest_runtest_logreport(self, report: TestReport):
        # On an xdist controller collection hooks are skipped, so the
        # baseline is reconciled lazily once all workers have reported.
        if not self._reconciled and self._xdist_collected:
            self._reconcile(self._xdist_collected)
        # Observe the counters before the Session hook (tryfirst) and the
        # other plugins react to this report.
        assert self.session is not None
        session = self.session
        testsfailed_before = session.testsfailed
        shouldfail_before = bool(session._shouldfail)
        shouldstop_before = bool(session._shouldstop)
        yield

        if not isinstance(report, TestReport):
            return

        entry = self.matched.get(report.nodeid)
        if entry is None or hasattr(report, "wasxfail"):
            return

        if report.failed:
            self.failed_nodeids.add(report.nodeid)
            if entry.is_expired(self.today):
                # Expired exemptions no longer suppress anything: the failure
                # keeps counting towards the exit code.
                self.expired_failures.add(report.nodeid)
                return

            # The Session hook counted this failed report; undo the count so
            # the known failure does not redden the exit code.
            if session.testsfailed > 0:
                session.testsfailed -= 1
            maxfail = session.config.getvalue("maxfail")
            if (
                maxfail
                and not shouldfail_before
                and session._shouldfail
                and testsfailed_before < maxfail <= testsfailed_before + 1
            ):
                # This known failure was the report that crossed the maxfail
                # threshold, so it must not have stopped the run.
                session._shouldfail = False
            if not shouldstop_before and session._shouldstop:
                # e.g. --stepwise reacted to this known failure; keep going.
                session._shouldstop = False
            self.suppressed[report.nodeid] = entry

        elif report.passed and report.when == "call":
            self.unexpected_passes.append(entry)

    def pytest_sessionfinish(self, session: Session, exitstatus: int | object) -> None:
        # xdist workers only see a slice of the tests and would race rewriting
        # the file; only the controller process prunes.
        if hasattr(self.config, "workerinput"):
            return
        if self.prune and not self.hard_errors:
            self._prune_manifest()

    def _prune_manifest(self) -> None:
        # Only write off entries that ran, passed cleanly and had no failing
        # phase; entries for tests not collected this run are preserved.
        remove_ids = {
            entry.nodeid
            for entry in self.unexpected_passes
            if entry.nodeid not in self.failed_nodeids
        }
        if not remove_ids:
            return

        payload: dict[str, object] = {
            "known_failures": [
                {
                    "nodeid": entry.nodeid,
                    "reason": entry.reason,
                    "expires": entry.expires.isoformat(),
                }
                for entry in self.entries.values()
                if entry.nodeid not in remove_ids
            ]
        }
        if self._has_version:
            payload = {"version": _MANIFEST_VERSION, **payload}

        tmp_path = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.manifest_path)
        self.pruned_count = len(remove_ids)

    def pytest_terminal_summary(
        self,
        terminalreporter: TerminalReporter,
        exitstatus: int | object,
        config: Config,
    ) -> None:
        if not self._reconciled and self._xdist_collected:
            # No reports were delivered (e.g. every collection was empty),
            # but the workers still told us which ids exist. This fallback is
            # display-only: it must not raise UsageError from the summary.
            self._reconcile(self._xdist_collected, enforce_strict=False)
        if self._refused:
            return

        warnings = [*self.hard_errors, *self.collection_warnings]
        if warnings:
            terminalreporter.write_sep(
                "!",
                "KNOWN-FAILURES BASELINE PROBLEMS - exemptions not enforced "
                "for the affected entries",
                red=True,
                bold=True,
            )
            for warning in warnings:
                terminalreporter.write_line(f"WARNING: {warning}", red=True)

        expired = [
            entry for entry in self.entries.values() if entry.is_expired(self.today)
        ]
        if expired:
            terminalreporter.write_sep(
                "!",
                "EXPIRED KNOWN-FAILURE EXEMPTIONS - renew or remove them",
                red=True,
                bold=True,
            )
            for entry in expired:
                suffix = ""
                if entry.nodeid in self.expired_failures:
                    suffix = "  -> still failing; counted as a regular failure"
                terminalreporter.write_line(
                    f"{entry.nodeid} (expired {entry.expires.isoformat()},"
                    f" reason: {entry.reason}){suffix}",
                    red=True,
                )

        clean_passes = [
            entry
            for entry in self.unexpected_passes
            if entry.nodeid not in self.failed_nodeids
        ]
        if clean_passes:
            terminalreporter.write_sep(
                "!",
                "UNEXPECTED SUCCESSES OF EXEMPTED TESTS - write off their "
                "manifest entries (--known-failures-prune)",
                yellow=True,
                bold=True,
            )
            for entry in clean_passes:
                terminalreporter.write_line(
                    f"{entry.nodeid} (reason: {entry.reason})",
                    yellow=True,
                )

        if self.suppressed:
            terminalreporter.write_sep(
                "=",
                "KNOWN FAILURES - reported above, but not counted in the exit code",
                green=True,
                bold=True,
            )
            for nodeid, entry in self.suppressed.items():
                terminalreporter.write_line(
                    f"{nodeid} (reason: {entry.reason}, "
                    f"expires: {entry.expires.isoformat()})",
                    green=True,
                )

        if self.pruned_count:
            terminalreporter.write_line(
                f"known-failures: removed {self.pruned_count} "
                f"{'entry' if self.pruned_count == 1 else 'entries'} from "
                f"{self.manifest_path}",
                green=True,
            )
