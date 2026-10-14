# mypy: allow-untyped-defs
from __future__ import annotations

import json
from pathlib import Path
import warnings

from _pytest.config import ExitCode
from _pytest.pytester import Pytester
from _pytest.warnings_baseline import BaselineEntry
from _pytest.warnings_baseline import BaselineError
from _pytest.warnings_baseline import build_baseline
from _pytest.warnings_baseline import current_environment
from _pytest.warnings_baseline import diff_baseline
from _pytest.warnings_baseline import dumps_baseline
from _pytest.warnings_baseline import entry_from_warning
from _pytest.warnings_baseline import environment_mismatch_note
from _pytest.warnings_baseline import format_entry_lines
from _pytest.warnings_baseline import parse_baseline
from _pytest.warnings_baseline import SCHEMA_VERSION
from _pytest.warnings_baseline import write_baseline_atomic
import pytest


pytestmark = pytest.mark.filterwarnings("default")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_warning(
    message: str = "some warning",
    category: type[Warning] = UserWarning,
    *,
    filename: str = "test_mod.py",
    lineno: int = 5,
) -> warnings.WarningMessage:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.warn_explicit(
            category(message),
            category=category,
            filename=filename,
            lineno=lineno,
        )
    return recorded[0]


def _entry(
    *,
    nodeid: str = "test_mod.py::test_func",
    category: str = "UserWarning",
    filename: str = "test_mod.py",
    lineno: int = 5,
    message: str = "some warning",
) -> BaselineEntry:
    return BaselineEntry(
        nodeid=nodeid,
        category=category,
        filename=filename,
        lineno=lineno,
        message=message,
    )


def _write_baseline(
    pytester: Pytester,
    entries: list[BaselineEntry],
    *,
    name: str = "baseline.json",
    environment: dict[str, str] | None = None,
) -> Path:
    path = pytester.path / name
    path.write_text(
        dumps_baseline(build_baseline(entries, environment=environment)),
        encoding="utf-8",
    )
    return path


def _read_baseline(pytester: Pytester, name: str = "baseline.json") -> dict:
    return json.loads((pytester.path / name).read_text(encoding="utf-8"))


def _baseline_tuples(data: dict) -> set[tuple]:
    return {
        (
            entry["nodeid"],
            entry["category"],
            entry["filename"],
            entry["lineno"],
            entry["message"],
        )
        for entry in data["warnings"]
    }


@pytest.fixture
def warning_file(pytester: Pytester) -> Path:
    # Line layout after pytester processing:
    # 1: import warnings, 3: def test_one, 4: UserWarning, 5: DeprecationWarning,
    # 7: def test_two, 8: UserWarning.
    return pytester.makepyfile(
        test_warning_file="""
            import warnings

            def test_one():
                warnings.warn(UserWarning("warning one"))
                warnings.warn(DeprecationWarning("deprecation one"))

            def test_two():
                warnings.warn(UserWarning("warning two"))
        """
    )


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestBaselineEntry:
    def test_roundtrip(self) -> None:
        entry = _entry()
        assert BaselineEntry.from_dict(entry.to_dict()) == entry

    def test_from_warning_fields(self) -> None:
        warning_message = _record_warning("hello")
        entry = entry_from_warning(
            warning_message, "test_mod.py::test_func", Path("/root/project")
        )
        assert entry == BaselineEntry(
            nodeid="test_mod.py::test_func",
            category="UserWarning",
            filename="test_mod.py",
            lineno=5,
            message="hello",
        )

    def test_from_warning_without_nodeid(self) -> None:
        warning_message = _record_warning()
        entry = entry_from_warning(warning_message, "", Path("/root/project"))
        assert entry.nodeid == ""

    def test_builtin_category_uses_bare_name(self) -> None:
        warning_message = _record_warning(category=DeprecationWarning)
        entry = entry_from_warning(warning_message, "", Path("/root"))
        assert entry.category == "DeprecationWarning"

    def test_custom_category_uses_dotted_name(self) -> None:
        class MyCustomWarning(Warning):
            pass

        warning_message = _record_warning(category=MyCustomWarning)
        entry = entry_from_warning(warning_message, "", Path("/root"))
        assert (
            entry.category
            == f"{__name__}.TestBaselineEntry.test_custom_category_uses_dotted_name.<locals>.MyCustomWarning"
        )

    def test_filename_outside_root_stays_absolute_posix(self, tmp_path: Path) -> None:
        warning_message = _record_warning(filename=str(tmp_path / "elsewhere" / "m.py"))
        entry = entry_from_warning(warning_message, "", tmp_path / "project")
        assert entry.filename.endswith("elsewhere/m.py")

    @pytest.mark.parametrize(
        "payload, match",
        [
            ([], "must be an object"),
            ({"nodeid": 1}, "'nodeid' must be a string"),
        ],
    )
    def test_from_dict_validation_simple(self, payload, match: str) -> None:
        with pytest.raises(BaselineError, match=match):
            BaselineEntry.from_dict(payload)

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("category", "", "'category' must not be empty"),
            ("filename", "", "'filename' must not be empty"),
            ("lineno", -1, "'lineno'"),
            ("lineno", True, "'lineno'"),
            ("lineno", "1", "'lineno'"),
            ("message", 1, "'message' must be a string"),
        ],
    )
    def test_from_dict_validation_fields(
        self, field: str, value: object, match: str
    ) -> None:
        payload = {
            "nodeid": "",
            "category": "UserWarning",
            "filename": "m.py",
            "lineno": 1,
            "message": "some warning",
        }
        payload[field] = value
        with pytest.raises(BaselineError, match=match):
            BaselineEntry.from_dict(payload)


class TestBaselineSerialization:
    def test_build_structure(self) -> None:
        baseline = build_baseline(
            [
                _entry(message="b"),
                _entry(message="a", lineno=1),
                _entry(message="a", lineno=1),  # duplicate
            ]
        )
        assert baseline["schema_version"] == SCHEMA_VERSION
        assert isinstance(baseline["generated_at"], str)
        assert set(baseline["generator"]) == {
            "pytest",
            "python",
            "python_implementation",
            "platform",
        }
        messages = [entry["message"] for entry in baseline["warnings"]]
        assert messages == ["a", "b"]  # sorted and de-duplicated

    def test_roundtrip(self) -> None:
        entries = [_entry(), _entry(nodeid="other.py::test_x", lineno=9)]
        text = dumps_baseline(build_baseline(entries))
        parsed, environment = parse_baseline(text)
        assert parsed == sorted(set(entries))
        assert environment == current_environment()

    def test_trailing_newline_and_indent(self) -> None:
        text = dumps_baseline(build_baseline([]))
        assert text.endswith("\n")
        assert '\n  "warnings": []' in text

    @pytest.mark.parametrize(
        "text, match",
        [
            ("{not json", "invalid JSON"),
            ("[]", "top-level value must be a JSON object"),
            ("{}", "missing or invalid 'schema_version'"),
            ('{"schema_version": 2}', "missing or invalid 'schema_version'"),
            ('{"schema_version": "2.0"}', "unsupported baseline schema_version"),
            (
                '{"schema_version": "1.0", "warnings": {}}',
                "'warnings' field must be a list",
            ),
            ('{"schema_version": "1.0", "warnings": [{}]}', r"warnings\[0\]"),
            (
                '{"schema_version": "1.0", "warnings": [], "generator": []}',
                "'generator' field must be an object",
            ),
        ],
    )
    def test_parse_errors(self, text: str, match: str) -> None:
        with pytest.raises(BaselineError, match=match):
            parse_baseline(text)

    def test_generator_optional(self) -> None:
        entries, environment = parse_baseline(
            '{"schema_version": "1.0", "warnings": []}'
        )
        assert entries == []
        assert environment is None

    def test_other_minor_schema_version_accepted(self) -> None:
        entries, _ = parse_baseline('{"schema_version": "1.9", "warnings": []}')
        assert entries == []


class TestWriteAtomic:
    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "baseline.json"
        write_baseline_atomic(target, build_baseline([]))
        assert target.exists()
        # No leftover temporary files.
        assert list(target.parent.glob("*.tmp")) == []

    def test_failure_leaves_existing_file_intact(self, tmp_path: Path) -> None:
        target = tmp_path / "baseline.json"
        write_baseline_atomic(target, build_baseline([_entry()]))
        before = target.read_text(encoding="utf-8")
        with pytest.raises(TypeError):
            write_baseline_atomic(target, {"warnings": {object()}})
        assert target.read_text(encoding="utf-8") == before
        assert list(tmp_path.glob(".*.tmp")) == []


class TestDiff:
    def test_diff_classifies(self) -> None:
        old = _entry(message="old")
        shared = _entry(message="shared", lineno=2)
        new = _entry(message="new", lineno=3)
        diff = diff_baseline([shared, new], [shared, old])
        assert diff.new == [new]
        assert diff.missing == [old]
        assert diff.matched_count == 1

    def test_diff_deduplicates(self) -> None:
        entry = _entry()
        diff = diff_baseline([entry, entry], [entry])
        assert diff.new == []
        assert diff.missing == []
        assert diff.captured == [entry]


class TestEnvironmentMismatch:
    def test_matching_environment(self) -> None:
        environment = current_environment()
        assert environment_mismatch_note(environment, environment) is None

    def test_different_environment(self) -> None:
        recorded = dict(current_environment())
        recorded["platform"] = "linux"
        note = environment_mismatch_note(current_environment(), recorded)
        assert note is not None
        assert "different environment" in note

    def test_missing_recorded_environment(self) -> None:
        assert environment_mismatch_note(current_environment(), None) is None


class TestFormatEntry:
    def test_with_nodeid(self) -> None:
        lines = format_entry_lines(_entry())
        assert lines == [
            "test_mod.py::test_func",
            "  test_mod.py:5: UserWarning: some warning",
        ]

    def test_without_nodeid(self) -> None:
        lines = format_entry_lines(_entry(nodeid=""))
        assert lines == ["  test_mod.py:5: UserWarning: some warning"]

    def test_multiline_message_indented(self) -> None:
        lines = format_entry_lines(_entry(message="line one\nline two"))
        assert lines[-1] == "  test_mod.py:5: UserWarning: line one\n  line two"


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestWarningsBaseline:
    """End-to-end behaviour through a pytester session."""

    BASELINE = "baseline.json"

    def _update(self, pytester: Pytester, *extra_args: str):
        return pytester.runpytest(
            "--warnings-baseline-update",
            "--warnings-baseline",
            self.BASELINE,
            *extra_args,
        )

    def _compare(self, pytester: Pytester, *extra_args: str):
        return pytester.runpytest("--warnings-baseline", self.BASELINE, *extra_args)

    def test_inert_without_baseline(self, pytester: Pytester, warning_file) -> None:
        result = pytester.runpytest()
        result.assert_outcomes(passed=2, warnings=3)
        assert result.ret == ExitCode.OK
        assert "warnings baseline" not in result.stdout.str()

    def test_update_creates_baseline(self, pytester: Pytester, warning_file) -> None:
        result = self._update(pytester)
        result.assert_outcomes(passed=2, warnings=3)
        assert result.ret == ExitCode.OK
        result.stdout.fnmatch_lines(
            ["warnings baseline updated: baseline.json now contains 3 entries *"]
        )

        data = _read_baseline(pytester)
        assert data["schema_version"] == SCHEMA_VERSION
        assert isinstance(data["generated_at"], str)
        assert set(data["generator"]) == {
            "pytest",
            "python",
            "python_implementation",
            "platform",
        }
        assert _baseline_tuples(data) == {
            (
                "test_warning_file.py::test_one",
                "UserWarning",
                "test_warning_file.py",
                4,
                "warning one",
            ),
            (
                "test_warning_file.py::test_one",
                "DeprecationWarning",
                "test_warning_file.py",
                5,
                "deprecation one",
            ),
            (
                "test_warning_file.py::test_two",
                "UserWarning",
                "test_warning_file.py",
                8,
                "warning two",
            ),
        }

    def test_entries_are_sorted(self, pytester: Pytester, warning_file) -> None:
        self._update(pytester)
        data = _read_baseline(pytester)
        assert data["warnings"] == sorted(
            data["warnings"],
            key=lambda entry: (
                entry["nodeid"],
                entry["category"],
                entry["filename"],
                entry["lineno"],
                entry["message"],
            ),
        )

    def test_identical_warnings_registered_once(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_looped="""
                import warnings

                def test_loop():
                    for i in range(5):
                        warnings.warn(UserWarning("same warning"))
            """
        )
        self._update(pytester)
        data = _read_baseline(pytester)
        assert len(data["warnings"]) == 1

    def test_compare_reports_full_match(self, pytester: Pytester, warning_file) -> None:
        self._update(pytester)
        result = self._compare(pytester)
        result.stdout.fnmatch_lines(
            [
                "=* warnings baseline =*",
                "*all 3 captured warnings match the 3 baseline entries*",
            ]
        )
        assert result.ret == ExitCode.OK

    def test_new_warning_annotated_by_default(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    warnings.warn(UserWarning("warning one"))
                    warnings.warn(DeprecationWarning("deprecation one"))

                def test_two():
                    warnings.warn(UserWarning("warning two"))

                def test_three():
                    warnings.warn(RuntimeWarning("brand new"))
            """
        )
        result = self._compare(pytester)
        assert result.ret == ExitCode.OK
        result.stdout.fnmatch_lines(
            [
                "*new warnings not in baseline (1 warning)*",
                "test_warning_file.py::test_three",
                "*RuntimeWarning: brand new*",
            ]
        )
        assert "not recorded in the warnings baseline" not in result.stdout.str()

    def test_new_warning_fails_run_via_cli(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    warnings.warn(UserWarning("warning one"))
                    warnings.warn(DeprecationWarning("deprecation one"))

                def test_two():
                    warnings.warn(UserWarning("warning two"))

                def test_three():
                    warnings.warn(RuntimeWarning("brand new"))
            """
        )
        result = self._compare(pytester, "--warnings-baseline-on-new", "fail")
        assert result.ret == ExitCode.WARNINGS_BASELINE_ERROR
        result.stdout.fnmatch_lines(
            [
                "*Tests pass, but 1 warning is not recorded in the warnings baseline*",
            ]
        )

    def test_new_warning_fails_run_via_ini(
        self, pytester: Pytester, warning_file
    ) -> None:
        pytester.makeini(
            """
            [pytest]
            warnings_baseline = baseline.json
            warnings_baseline_on_new = fail
            """
        )
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    pass

                def test_two():
                    pass

                def test_three():
                    warnings.warn(RuntimeWarning("brand new"))
            """
        )
        result = pytester.runpytest()
        assert result.ret == ExitCode.WARNINGS_BASELINE_ERROR

    def test_test_failure_takes_priority(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    warnings.warn(UserWarning("warning one"))
                    warnings.warn(DeprecationWarning("deprecation one"))
                    assert False

                def test_two():
                    warnings.warn(UserWarning("warning two"))
            """
        )
        result = self._compare(pytester, "--warnings-baseline-on-new", "fail")
        assert result.ret == ExitCode.TESTS_FAILED

    def test_disappeared_warning_is_write_off_candidate(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    warnings.warn(UserWarning("warning one"))
                    warnings.warn(DeprecationWarning("deprecation one"))

                def test_two():
                    pass
            """
        )
        result = self._compare(pytester)
        result.stdout.fnmatch_lines(
            [
                "*baseline warnings not observed this run (1 entry)*"
                "--warnings-baseline-update to write them off*",
                "test_warning_file.py::test_two",
                "*UserWarning: warning two*",
            ]
        )

        result = self._update(pytester)
        result.stdout.fnmatch_lines(
            ["*warnings baseline updated:* now contains 2 entries (1 written off)*"]
        )
        data = _read_baseline(pytester)
        assert all(entry["message"] != "warning two" for entry in data["warnings"])

        result = self._compare(pytester)
        result.stdout.fnmatch_lines(
            ["*all 2 captured warnings match the 2 baseline entries*"]
        )

    def test_merge_keeps_unobserved_entries(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        result = pytester.runpytest(
            "--warnings-baseline-merge",
            "--warnings-baseline",
            self.BASELINE,
            "-k",
            "test_one",
        )
        assert result.ret == ExitCode.OK
        assert len(_read_baseline(pytester)["warnings"]) == 3

    def test_merge_registers_entries_from_partial_run(
        self, pytester: Pytester, warning_file
    ) -> None:
        # Baseline only knows test_two's warning; test_one's warnings get
        # registered through a merge of a partial (test_one-only) run.
        _write_baseline(
            pytester,
            [
                BaselineEntry(
                    nodeid="test_warning_file.py::test_two",
                    category="UserWarning",
                    filename="test_warning_file.py",
                    lineno=8,
                    message="warning two",
                )
            ],
        )
        result = pytester.runpytest(
            "--warnings-baseline-merge",
            "--warnings-baseline",
            self.BASELINE,
            "-k",
            "test_one",
        )
        assert result.ret == ExitCode.OK
        result.stdout.fnmatch_lines(
            ["*warnings baseline merged: 2 new entries registered*"]
        )
        assert len(_read_baseline(pytester)["warnings"]) == 3

    def test_update_on_partial_run_writes_off(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.runpytest(
            "--warnings-baseline-update",
            "--warnings-baseline",
            self.BASELINE,
            "-k",
            "test_one",
        )
        data = _read_baseline(pytester)
        assert {entry["nodeid"] for entry in data["warnings"]} == {
            "test_warning_file.py::test_one"
        }

    def test_missing_baseline_is_usage_error(
        self, pytester: Pytester, warning_file
    ) -> None:
        result = self._compare(pytester)
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(
            [
                "*ERROR: warnings baseline file not found*",
                "*--warnings-baseline-update to create it*",
            ]
        )

    def test_update_merge_conflict_is_usage_error(
        self, pytester: Pytester, warning_file
    ) -> None:
        result = pytester.runpytest(
            "--warnings-baseline-update",
            "--warnings-baseline-merge",
            "--warnings-baseline",
            self.BASELINE,
        )
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*cannot be combined*"])

    def test_update_without_path_is_usage_error(
        self, pytester: Pytester, warning_file
    ) -> None:
        result = pytester.runpytest("--warnings-baseline-update")
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*requires a baseline path*"])

    def test_corrupt_baseline_is_usage_error(
        self, pytester: Pytester, warning_file
    ) -> None:
        (pytester.path / self.BASELINE).write_text("{not json", encoding="utf-8")
        result = self._compare(pytester)
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*could not parse warnings baseline*"])

    def test_unsupported_schema_is_usage_error(
        self, pytester: Pytester, warning_file
    ) -> None:
        (pytester.path / self.BASELINE).write_text(
            '{"schema_version": "2.0", "warnings": []}', encoding="utf-8"
        )
        result = self._compare(pytester)
        assert result.ret == ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*unsupported baseline schema_version*"])

    def test_update_skipped_when_tests_fail(self, pytester: Pytester) -> None:
        pytester.makepyfile(
            test_fail="""
                import warnings

                def test_fail():
                    warnings.warn(UserWarning("before failure"))
                    assert False
            """
        )
        result = self._update(pytester)
        assert result.ret == ExitCode.TESTS_FAILED
        assert not (pytester.path / self.BASELINE).exists()
        result.stdout.fnmatch_lines(["*warnings baseline not changed*"])

    def test_update_skipped_when_nothing_collected(
        self, pytester: Pytester, warning_file
    ) -> None:
        result = self._update(pytester, "-k", "no_such_test")
        assert result.ret == ExitCode.NO_TESTS_COLLECTED
        assert not (pytester.path / self.BASELINE).exists()

    def test_ignored_warnings_are_not_registered(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester, "-W", "ignore::UserWarning")
        data = _read_baseline(pytester)
        assert _baseline_tuples(data) == {
            (
                "test_warning_file.py::test_one",
                "DeprecationWarning",
                "test_warning_file.py",
                5,
                "deprecation one",
            ),
        }

    def test_error_filter_semantics_unchanged(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        result = self._compare(
            pytester,
            "--warnings-baseline-on-new",
            "fail",
            "-W",
            "error::UserWarning",
        )
        # The UserWarnings turn into test errors; the baseline layer does not
        # interfere with filterwarnings semantics.
        assert result.ret == ExitCode.TESTS_FAILED

    def test_collection_warning_registered_without_nodeid(
        self, pytester: Pytester
    ) -> None:
        pytester.makepyfile(
            test_collection_warning="""
                import warnings

                warnings.warn(UserWarning("collection time warning"))

                def test_one():
                    pass
            """
        )
        self._update(pytester)
        data = _read_baseline(pytester)
        (entry,) = (
            e for e in data["warnings"] if e["message"] == "collection time warning"
        )
        assert entry["nodeid"] == ""
        assert entry["category"] == "UserWarning"
        assert entry["filename"] == "test_collection_warning.py"

    def test_environment_mismatch_note(self, pytester: Pytester, warning_file) -> None:
        self._update(pytester)
        data = _read_baseline(pytester)
        data["generator"]["platform"] = "linux"
        data["generator"]["python"] = "2.7.18"
        (pytester.path / self.BASELINE).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        result = self._compare(pytester)
        result.stdout.fnmatch_lines(
            [
                "*note: baseline was generated in a different environment*linux*",
            ]
        )

    def test_disable_warnings_hides_section_but_not_failure(
        self, pytester: Pytester, warning_file
    ) -> None:
        self._update(pytester)
        pytester.makepyfile(
            test_warning_file="""
                import warnings

                def test_one():
                    pass

                def test_two():
                    pass

                def test_three():
                    warnings.warn(RuntimeWarning("brand new"))
            """
        )
        result = self._compare(
            pytester,
            "--disable-warnings",
            "--warnings-baseline-on-new",
            "fail",
        )
        assert result.ret == ExitCode.WARNINGS_BASELINE_ERROR
        output = result.stdout.str()
        assert "new warnings not in baseline" not in output
        assert "is not recorded in the warnings baseline" in output

    def test_ini_baseline_path(self, pytester: Pytester, warning_file) -> None:
        pytester.makeini(
            """
            [pytest]
            warnings_baseline = ini-baseline.json
            """
        )
        result = pytester.runpytest("--warnings-baseline-update")
        assert result.ret == ExitCode.OK
        assert (pytester.path / "ini-baseline.json").exists()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*all 3 captured warnings match*"])

    def test_toml_baseline_path(self, pytester: Pytester, warning_file) -> None:
        # Native TOML mode (pytest.toml / [pytest]) validates paths as a
        # list of strings.
        pytester.maketoml(
            """
            [pytest]
            warnings_baseline = ["toml-baseline.json"]
            """
        )
        result = pytester.runpytest("--warnings-baseline-update")
        assert result.ret == ExitCode.OK
        assert (pytester.path / "toml-baseline.json").exists()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*all 3 captured warnings match*"])

    def test_pyproject_ini_mode_baseline_path(
        self, pytester: Pytester, warning_file
    ) -> None:
        # In pyproject.toml's [tool.pytest.ini_options] (INI-compatible
        # mode), the path is a plain scalar string.
        pytester.makepyprojecttoml(
            """
            [tool.pytest.ini_options]
            warnings_baseline = "pyproject-baseline.json"
            """
        )
        result = pytester.runpytest("--warnings-baseline-update")
        assert result.ret == ExitCode.OK
        assert (pytester.path / "pyproject-baseline.json").exists()
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*all 3 captured warnings match*"])

    def test_update_is_atomic(self, pytester: Pytester, warning_file) -> None:
        self._update(pytester)
        before = (pytester.path / self.BASELINE).read_bytes()
        self._update(pytester)
        after = (pytester.path / self.BASELINE).read_bytes()
        # Same captured warnings -> identical file apart from generated_at,
        # and no leftover temp files next to the baseline.
        before_json = json.loads(before)
        after_json = json.loads(after)
        assert before_json["warnings"] == after_json["warnings"]
        assert list(pytester.path.glob(".*.tmp")) == []
