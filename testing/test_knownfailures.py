# mypy: disallow-untyped-defs
from __future__ import annotations

import datetime
import json
from pathlib import Path

from _pytest.knownfailures import parse_manifest
from _pytest.pytester import Pytester
import pytest


MANIFEST_NAME = "known-failures.json"
TODAY = datetime.datetime.now().date()
FUTURE = (TODAY + datetime.timedelta(days=365)).isoformat()
YESTERDAY = (TODAY - datetime.timedelta(days=1)).isoformat()

SUPPRESSED_BANNER = "KNOWN FAILURES - reported above, but not counted in the exit code"
EXPIRED_BANNER = "EXPIRED KNOWN-FAILURE EXEMPTIONS"
UNEXPECTED_BANNER = "UNEXPECTED SUCCESSES OF EXEMPTED TESTS"
PROBLEMS_BANNER = "KNOWN-FAILURES BASELINE PROBLEMS"


def write_manifest(pytester: Pytester, data: object, name: str = MANIFEST_NAME) -> Path:
    path = pytester.path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def entry(
    nodeid: str, *, reason: str = "JIRA-1", expires: str = FUTURE
) -> dict[str, str]:
    return {"nodeid": nodeid, "reason": reason, "expires": expires}


@pytest.fixture
def suite_pytester(pytester: Pytester) -> Pytester:
    pytester.makepyfile(
        test_known="""
def test_ok():
    assert True

def test_bad():
    assert False, "boom message"

def test_after_bad():
    assert True
"""
    )
    return pytester


# ---------------------------------------------------------------------------
# Pure parser/validator tests.
# ---------------------------------------------------------------------------


def test_parse_manifest_valid() -> None:
    data = {
        "version": 1,
        "known_failures": [
            {"nodeid": "test_x.py::test_a", "reason": "why", "expires": "2030-01-02"}
        ],
    }
    entries, errors = parse_manifest(data)
    assert errors == []
    assert len(entries) == 1
    assert entries[0].nodeid == "test_x.py::test_a"
    assert entries[0].reason == "why"
    assert entries[0].expires == datetime.date(2030, 1, 2)


def test_parse_manifest_version_optional() -> None:
    entries, errors = parse_manifest(
        {"known_failures": []},
    )
    assert entries == []
    assert errors == []


@pytest.mark.parametrize(
    "data",
    [
        [],
        "nope",
        42,
        {},
        {"known_failures": {}},
        {"known_failures": "x"},
        {"version": 2, "known_failures": []},
        {"version": "1", "known_failures": []},
        {"bogus": 1, "known_failures": []},
    ],
)
def test_parse_manifest_bad_document(data: object) -> None:
    _, errors = parse_manifest(data)
    assert errors


@pytest.mark.parametrize(
    "raw_entry",
    [
        [],
        "x",
        {"nodeid": "test_x.py::test_a", "reason": "r", "expires": "2030-01-01", "x": 1},
        {"reason": "r", "expires": "2030-01-01"},
        {"nodeid": "", "reason": "r", "expires": "2030-01-01"},
        {"nodeid": 4, "reason": "r", "expires": "2030-01-01"},
        {"nodeid": "a", "reason": "", "expires": "2030-01-01"},
        {"nodeid": "a", "reason": 5, "expires": "2030-01-01"},
        {"nodeid": "a", "reason": "r", "expires": "2030-1-1"},
        {"nodeid": "a", "reason": "r", "expires": "01/02/2030"},
        {"nodeid": "a", "reason": "r", "expires": "2030-13-01"},
        {"nodeid": "a", "reason": "r"},
        {"nodeid": "a", "expires": "2030-01-01"},
    ],
)
def test_parse_manifest_bad_entry(raw_entry: object) -> None:
    _, errors = parse_manifest({"known_failures": [raw_entry]})
    assert errors


def test_parse_manifest_duplicate_nodeid() -> None:
    data = {
        "known_failures": [
            entry("test_x.py::test_a"),
            entry("test_x.py::test_a", reason="second"),
        ]
    }
    entries, errors = parse_manifest(data)
    assert len(entries) == 1
    assert len(errors) == 1
    assert "duplicate" in errors[0]
    assert "test_x.py::test_a" in errors[0]


def test_parse_manifest_partial_recovery() -> None:
    data = {
        "known_failures": [
            entry("test_x.py::test_good"),
            {"nodeid": "test_x.py::test_bad_date", "reason": "r", "expires": "nope"},
        ]
    }
    entries, errors = parse_manifest(data)
    assert [e.nodeid for e in entries] == ["test_x.py::test_good"]
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# Behaviour without a manifest: judgement and exit codes stay unchanged.
# ---------------------------------------------------------------------------


def test_without_manifest_failure_stays_red(suite_pytester: Pytester) -> None:
    result = suite_pytester.runpytest()
    assert result.ret == 1
    assert SUPPRESSED_BANNER not in result.stdout.str()
    assert EXPIRED_BANNER not in result.stdout.str()
    assert UNEXPECTED_BANNER not in result.stdout.str()


def test_without_manifest_success_stays_green(pytester: Pytester) -> None:
    pytester.makepyfile(test_ok="def test_ok():\n    assert True\n")
    result = pytester.runpytest()
    assert result.ret == 0
    assert "known-failures" not in result.stdout.str().lower()


# ---------------------------------------------------------------------------
# Core suppression semantics.
# ---------------------------------------------------------------------------


def test_exempt_failure_is_reported_but_exit_green(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_bad", reason="JIRA-42")]},
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    # The failure is still executed and reported in full.
    result.assert_outcomes(passed=2, failed=1)
    assert result.ret == 0
    stdout = result.stdout.str()
    assert "FAILED test_known.py::test_bad" in stdout
    assert "boom message" in stdout  # failure section rendered
    assert SUPPRESSED_BANNER in stdout
    assert "test_known.py::test_bad (reason: JIRA-42, expires:" in stdout


def test_unlisted_failure_keeps_exit_red(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_mix="""
def test_exempt_bad():
    assert False, "known"

def test_new_bad():
    assert False, "brand new failure"
"""
    )
    write_manifest(
        pytester,
        {"known_failures": [entry("test_mix.py::test_exempt_bad")]},
    )
    result = pytester.runpytest("--known-failures", MANIFEST_NAME)
    # test_new_bad is not on the baseline and keeps the exit code red.
    assert result.ret == 1
    result.assert_outcomes(failed=2)
    stdout = result.stdout.str()
    assert SUPPRESSED_BANNER in stdout
    assert "test_mix.py::test_exempt_bad" in stdout
    assert "brand new failure" in stdout


def test_exempt_failure_does_not_trigger_maxfail(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_maxfail="""
def test_bad():
    assert False

def test_after():
    assert True
"""
    )
    write_manifest(
        pytester,
        {"known_failures": [entry("test_maxfail.py::test_bad")]},
    )
    result = pytester.runpytest("--known-failures", MANIFEST_NAME, "--maxfail=1")
    # Both tests ran; the known failure must not stop the session.
    result.assert_outcomes(passed=1, failed=1)
    assert result.ret == 0


def test_exempt_setup_error_is_suppressed(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_setup="""
import pytest

@pytest.fixture
def broken():
    raise RuntimeError("setup boom")

def test_bad(broken):
    assert True
"""
    )
    write_manifest(
        pytester,
        {"known_failures": [entry("test_setup.py::test_bad")]},
    )
    result = pytester.runpytest("--known-failures", MANIFEST_NAME)
    result.assert_outcomes(errors=1)
    assert result.ret == 0
    assert "test_setup.py::test_bad" in result.stdout.str()
    assert SUPPRESSED_BANNER in result.stdout.str()


def test_parametrized_exact_match(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_param="""
import pytest

@pytest.mark.parametrize("value", [1, 2])
def test_value(value):
    assert value == 2
"""
    )
    write_manifest(
        pytester,
        {"known_failures": [entry("test_param.py::test_value[1]")]},
    )
    result = pytester.runpytest("--known-failures", MANIFEST_NAME)
    result.assert_outcomes(passed=1, failed=1)
    assert result.ret == 0


# ---------------------------------------------------------------------------
# Expiry semantics.
# ---------------------------------------------------------------------------


def test_expired_entry_failure_counts_again(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {
            "known_failures": [
                entry("test_known.py::test_bad", reason="old", expires=YESTERDAY)
            ]
        },
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 1
    stdout = result.stdout.str()
    assert EXPIRED_BANNER in stdout
    assert "still failing; counted as a regular failure" in stdout
    assert SUPPRESSED_BANNER not in stdout


def test_expires_today_is_still_active(pytester: Pytester) -> None:
    pytester.makepyfile(test_today="def test_bad():\n    assert False\n")
    write_manifest(
        pytester,
        {
            "known_failures": [
                entry("test_today.py::test_bad", expires=TODAY.isoformat())
            ]
        },
    )
    result = pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 0
    assert SUPPRESSED_BANNER in result.stdout.str()
    assert EXPIRED_BANNER not in result.stdout.str()


def test_expired_entry_that_now_passes_gets_both_notices(
    suite_pytester: Pytester,
) -> None:
    write_manifest(
        suite_pytester,
        {
            "known_failures": [
                entry("test_known.py::test_ok", reason="fixed?", expires=YESTERDAY),
                entry("test_known.py::test_bad"),
            ]
        },
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 0
    stdout = result.stdout.str()
    assert EXPIRED_BANNER in stdout
    assert UNEXPECTED_BANNER in stdout
    assert "test_known.py::test_ok" in stdout


# ---------------------------------------------------------------------------
# Unexpected success.
# ---------------------------------------------------------------------------


def test_unexpected_success_is_announced(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_ok", reason="JIRA-7")]},
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 1  # test_bad still fails normally
    stdout = result.stdout.str()
    assert UNEXPECTED_BANNER in stdout
    assert "test_known.py::test_ok (reason: JIRA-7)" in stdout


def test_teardown_failure_blocks_write_off(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_teardown="""
import pytest

@pytest.fixture
def bad_teardown():
    yield
    raise RuntimeError("teardown boom")

def test_case(bad_teardown):
    assert True
"""
    )
    manifest = write_manifest(
        pytester,
        {"known_failures": [entry("test_teardown.py::test_case")]},
    )
    result = pytester.runpytest(
        "--known-failures", MANIFEST_NAME, "--known-failures-prune"
    )
    # The teardown error is exempt, the call passed; the entry is kept.
    assert result.ret == 0
    assert UNEXPECTED_BANNER not in result.stdout.str()
    assert SUPPRESSED_BANNER in result.stdout.str()
    assert "test_teardown.py::test_case" in manifest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Strict vs loose validation.
# ---------------------------------------------------------------------------


def test_strict_invalid_json_refuses(suite_pytester: Pytester) -> None:
    manifest = suite_pytester.path / MANIFEST_NAME
    manifest.write_text("{ not json", encoding="utf-8")
    result = suite_pytester.runpytest(
        "--known-failures", MANIFEST_NAME, "--known-failures-strict"
    )
    assert result.ret == 4
    assert "invalid known-failures manifest" in result.stderr.str()
    assert "not valid JSON" in result.stderr.str()
    # Refusal happens during configuration, before any collection output.
    assert result.stdout.lines == []


def test_loose_invalid_json_warns_and_runs(suite_pytester: Pytester) -> None:
    manifest = suite_pytester.path / MANIFEST_NAME
    manifest.write_text("{ not json", encoding="utf-8")
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    # Nothing is suppressed with an unreadable manifest, failures stay real.
    assert result.ret == 1
    stdout = result.stdout.str()
    assert PROBLEMS_BANNER in stdout
    assert "not valid JSON" in stdout
    assert SUPPRESSED_BANNER not in stdout


def test_strict_duplicate_refuses(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {
            "known_failures": [
                entry("test_known.py::test_bad"),
                entry("test_known.py::test_bad", reason="again"),
            ]
        },
    )
    result = suite_pytester.runpytest(
        "--known-failures", MANIFEST_NAME, "--known-failures-strict"
    )
    assert result.ret == 4
    assert "duplicate" in result.stderr.str()


def test_loose_duplicate_warns_but_keeps_first(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {
            "known_failures": [
                entry("test_known.py::test_bad", reason="first"),
                entry("test_known.py::test_bad", reason="again"),
            ]
        },
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 0
    stdout = result.stdout.str()
    assert PROBLEMS_BANNER in stdout
    assert "duplicate" in stdout
    assert "reason: first" in stdout


def test_strict_unknown_nodeid_refuses(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_does_not_exist")]},
    )
    result = suite_pytester.runpytest(
        "--known-failures", MANIFEST_NAME, "--known-failures-strict"
    )
    assert result.ret == 4
    assert "does not match any collected test" in result.stderr.str()
    assert "test_does_not_exist" in result.stderr.str()
    assert "no tests ran" in result.stdout.str()


def test_loose_unknown_nodeid_warns(suite_pytester: Pytester) -> None:
    write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_does_not_exist")]},
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 1  # real failure test_bad remains red
    stdout = result.stdout.str()
    assert PROBLEMS_BANNER in stdout
    assert "does not match any collected test" in stdout


def test_strict_missing_file_refuses(suite_pytester: Pytester) -> None:
    result = suite_pytester.runpytest(
        "--known-failures", "missing.json", "--known-failures-strict"
    )
    assert result.ret == 4
    assert "cannot read known-failures manifest" in result.stderr.str()


def test_loose_missing_file_warns(suite_pytester: Pytester) -> None:
    result = suite_pytester.runpytest("--known-failures", "missing.json")
    assert result.ret == 1
    assert PROBLEMS_BANNER in result.stdout.str()
    assert "cannot read known-failures manifest" in result.stdout.str()


def test_empty_manifest_is_valid(suite_pytester: Pytester) -> None:
    write_manifest(suite_pytester, {"known_failures": []})
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 1
    assert PROBLEMS_BANNER not in result.stdout.str()
    assert SUPPRESSED_BANNER not in result.stdout.str()


def test_strict_via_ini(suite_pytester: Pytester) -> None:
    suite_pytester.makeini(
        """
        [pytest]
        known_failures_strict = true
        """
    )
    write_manifest(suite_pytester, {"known_failures": []})
    (suite_pytester.path / MANIFEST_NAME).write_text("{ broken", encoding="utf-8")
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 4


# ---------------------------------------------------------------------------
# INI configuration.
# ---------------------------------------------------------------------------


def test_manifest_path_from_ini(suite_pytester: Pytester) -> None:
    suite_pytester.makeini(
        f"""
        [pytest]
        known_failures = {MANIFEST_NAME}
        """
    )
    write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_bad")]},
    )
    result = suite_pytester.runpytest()
    assert result.ret == 0
    assert SUPPRESSED_BANNER in result.stdout.str()


# ---------------------------------------------------------------------------
# Prune (write-off).
# ---------------------------------------------------------------------------


def test_prune_removes_passing_entries(suite_pytester: Pytester) -> None:
    manifest = write_manifest(
        suite_pytester,
        {
            "version": 1,
            "known_failures": [
                entry("test_known.py::test_ok", reason="fixed now"),
                entry("test_known.py::test_bad", reason="still broken"),
            ],
        },
    )
    result = suite_pytester.runpytest(
        "--known-failures", MANIFEST_NAME, "--known-failures-prune"
    )
    assert result.ret == 0
    stdout = result.stdout.str()
    assert "removed 1 entry" in stdout
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert [e["nodeid"] for e in data["known_failures"]] == ["test_known.py::test_bad"]

    # A second run no longer announces the unexpected success.
    result2 = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result2.ret == 0
    assert UNEXPECTED_BANNER not in result2.stdout.str()
    assert SUPPRESSED_BANNER in result2.stdout.str()


def test_prune_is_opt_in(suite_pytester: Pytester) -> None:
    manifest = write_manifest(
        suite_pytester,
        {"known_failures": [entry("test_known.py::test_ok")]},
    )
    result = suite_pytester.runpytest("--known-failures", MANIFEST_NAME)
    assert result.ret == 1
    assert [
        e["nodeid"] for e in json.loads(manifest.read_text())["known_failures"]
    ] == ["test_known.py::test_ok"]


def test_prune_preserves_uncollected_entries(pytester: Pytester) -> None:
    pytester.makepyfile(
        test_a="""
def test_a_bad():
    assert False
""",
        test_b="""
def test_b_ok():
    assert True
""",
    )
    manifest = write_manifest(
        pytester,
        {
            "known_failures": [
                entry("test_a.py::test_a_bad"),
                # Passes, but is not part of the selected run.
                entry("test_b.py::test_b_ok"),
            ]
        },
    )
    result = pytester.runpytest(
        "test_a.py",
        "--known-failures",
        MANIFEST_NAME,
        "--known-failures-prune",
    )
    assert result.ret == 0
    kept = {e["nodeid"] for e in json.loads(manifest.read_text())["known_failures"]}
    assert kept == {"test_a.py::test_a_bad", "test_b.py::test_b_ok"}


# ---------------------------------------------------------------------------
# xdist: the controller process decides the exit code from the workers'
# forwarded reports, so reconciliation happens there from the union of ids.
# ---------------------------------------------------------------------------


@pytest.fixture
def xdist_pytester(pytester: Pytester) -> Pytester:
    pytest.importorskip("xdist")
    # Pytester disables plugin autoload; subprocess runs must be told to
    # explicitly load xdist (in-process runs cannot parse ``-n`` either).
    pytester.plugins = ["xdist.plugin"]
    pytester.makepyfile(
        test_x="""
def test_ok():
    assert True

def test_bad():
    assert False, "boom"
"""
    )
    return pytester


def test_xdist_exempt_failure_exit_green(xdist_pytester: Pytester) -> None:
    write_manifest(
        xdist_pytester,
        {"known_failures": [entry("test_x.py::test_bad")]},
    )
    result = xdist_pytester.runpytest_subprocess(
        "-n", "2", "--known-failures", MANIFEST_NAME
    )
    assert result.ret == 0
    stdout = result.stdout.str()
    assert "FAILED" in stdout
    assert SUPPRESSED_BANNER in stdout


def test_xdist_unlisted_failure_stays_red(xdist_pytester: Pytester) -> None:
    write_manifest(xdist_pytester, {"known_failures": []})
    result = xdist_pytester.runpytest_subprocess(
        "-n", "2", "--known-failures", MANIFEST_NAME
    )
    assert result.ret == 1


def test_xdist_strict_unknown_nodeid_refuses(xdist_pytester: Pytester) -> None:
    write_manifest(
        xdist_pytester,
        {"known_failures": [entry("test_x.py::test_ghost")]},
    )
    result = xdist_pytester.runpytest_subprocess(
        "-n",
        "2",
        "--known-failures",
        MANIFEST_NAME,
        "--known-failures-strict",
    )
    assert result.ret == 4
    assert "does not match any collected test" in result.stderr.str()


def test_xdist_prune_runs_on_controller(xdist_pytester: Pytester) -> None:
    manifest = write_manifest(
        xdist_pytester,
        {
            "known_failures": [
                entry("test_x.py::test_ok"),
                entry("test_x.py::test_bad"),
            ]
        },
    )
    result = xdist_pytester.runpytest_subprocess(
        "-n",
        "2",
        "--known-failures",
        MANIFEST_NAME,
        "--known-failures-prune",
    )
    assert result.ret == 0
    assert "removed 1 entry" in result.stdout.str()
    kept = {e["nodeid"] for e in json.loads(manifest.read_text())["known_failures"]}
    assert kept == {"test_x.py::test_bad"}
