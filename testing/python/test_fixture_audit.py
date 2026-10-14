"""Tests for the read-only fixture resolution audit (``--fixture-audit``)."""

from __future__ import annotations

import json

from _pytest.pytester import Pytester


def test_fixture_audit_options_in_help(pytester: Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(
        [
            "*--fixture-audit*",
            "*--fixture-audit-json=PATH*",
        ]
    )


def test_audit_does_not_run_tests(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def arg1():
            return 1

        def test_arg1(arg1):
            raise AssertionError("test body must not be executed during audit")
        """
    )
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*no tests ran*"])


def test_audit_zero_items(pytester: Pytester) -> None:
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result.stdout.no_fnmatch_line("*fixture audit for*")


def test_per_test_human_override_chain(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture
        def f():
            '''conftest f'''
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            '''module f'''

        class TestX:
            @pytest.fixture
            def f(self, f):
                '''class f'''

            def test_a(self, f):
                pass
        """
    )
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*fixture audit for *test_per_test_human_override_chain.py::TestX::test_a*",
            "*f  (requested by: parameter)*",
            "*[[]shadowed[]]*conftest.py*f*-- session-global*",
            "*[[]super[]]*test_per_test_human_override_chain.py*f*-- test module*",
            "*[[]winner[]]*test_per_test_human_override_chain.py*f*-- class*TestX*",
            "*reason: The closest of 3 definition(s)*chain index 2*",
        ]
    )


def test_super_fixture_layer_marked(pytester: Pytester) -> None:
    """An outer layer statically requested by the winner (super fixture)
    is marked [super] and active."""
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            return "outer"

        class TestX:
            @pytest.fixture
            def f(self, f):
                return "class:" + f

            def test_a(self, f):
                pass
        """
    )
    result = pytester.runpytest("--fixture-audit", "-v")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*[[]super[]]*f*-- test module*",
            "*[[]winner[]]*f*-- class*TestX*",
        ]
    )
    result.stdout.no_fnmatch_line("*[[]shadowed[]]*")


def test_unresolved_fixture_reported(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        def test_missing(does_not_exist):
            pass
        """
    )
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        ["*does_not_exist*", "*no definition visible to this item*"]
    )


def test_direct_parametrize_excluded(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("x", [1])
        def test_a(x):
            pass
        """
    )
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result.stdout.no_fnmatch_line("*fixture audit for*")


def test_json_per_test_chain(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture
        def f():
            '''conftest f'''
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            '''module f'''

        @pytest.fixture
        def dep(f):
            '''depends on f'''

        class TestX:
            @pytest.fixture
            def f(self, f):
                '''class f (statically requests super)'''

            def test_a(self, f, dep):
                pass
        """
    )
    audit_path = pytester.path / "audit.json"
    result = pytester.runpytest("--fixture-audit-json", str(audit_path))
    assert result.ret == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == "1.0"
    assert {"definitions", "items", "rootpath", "pytest_version"} <= set(report)

    item = next(i for i in report["items"] if i["nodeid"].endswith("TestX::test_a"))
    fx = next(f for f in item["fixtures"] if f["argname"] == "f")
    assert fx["requested_by"] == "parameter"
    assert fx["found"] is True
    assert fx["winner_index"] == 2
    [conftest_layer, module_layer, class_layer] = fx["chain"]

    # Ordering: outermost first, winner last.
    assert conftest_layer["index"] == 0
    assert conftest_layer["role"] == "shadowed"
    assert conftest_layer["active"] is False
    assert conftest_layer["shadowed_by_index"] == 1
    assert conftest_layer["defined_at"]["kind"] == "session"
    assert conftest_layer["location"]["line"] is not None
    assert conftest_layer["location"]["path"].endswith("conftest.py")
    assert "closer" in conftest_layer["reason"]

    assert module_layer["role"] == "super"
    assert module_layer["active"] is True
    assert module_layer["defined_at"]["kind"] == "module"

    assert class_layer["role"] == "winner"
    assert class_layer["active"] is True
    assert class_layer["shadowed_by_index"] is None
    assert class_layer["defined_at"]["kind"] == "class"

    assert fx["winner_reason"]
    assert "chain index 2" in fx["winner_reason"]

    dep = next(f for f in item["fixtures"] if f["argname"] == "dep")
    assert dep["requested_by"] == "parameter"
    assert dep["chain"][0]["role"] == "winner"


def test_json_missing_fixture_found_false(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        def test_missing(does_not_exist):
            pass
        """
    )
    audit_path = pytester.path / "audit.json"
    result = pytester.runpytest("--fixture-audit-json", str(audit_path))
    assert result.ret == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    fx = next(
        f
        for i in report["items"]
        for f in i["fixtures"]
        if f["argname"] == "does_not_exist"
    )
    assert fx["found"] is False
    assert fx["winner_index"] is None
    assert fx["chain"] == []


def test_json_plugin_name_clash_same_node(pytester: Pytester) -> None:
    """Two plugins defining the same fixture at session level: later
    registration wins silently; the loser is marked fully overridden."""
    pytester.makepyfile(
        plugin_a="""
        import pytest

        @pytest.fixture
        def collide():
            return "a"
        """,
        plugin_b="""
        import pytest

        @pytest.fixture
        def collide():
            return "b"
        """,
    )
    pytester.makeconftest(
        """
        pytest_plugins = ("plugin_a", "plugin_b")
        """
    )
    pytester.makepyfile(
        """
        def test_collide(collide):
            assert collide == "b"
        """
    )
    audit_path = pytester.path / "audit.json"
    result = pytester.runpytest("--fixture-audit-json", str(audit_path))
    assert result.ret == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))

    collide_defs = [d for d in report["definitions"] if d["argname"] == "collide"]
    assert len(collide_defs) == 2
    statuses = {d["location"]["module"]: d["status"] for d in collide_defs}
    assert statuses == {"plugin_a": "overridden", "plugin_b": "active"}
    loser = next(d for d in collide_defs if d["location"]["module"] == "plugin_a")
    assert len(loser["overridden_by"]) == 1
    override = loser["overridden_by"][0]
    assert override["relation"] == "same_node_later_registration"
    assert override["location"]["module"] == "plugin_b"
    assert "later" in override["reason"]
    winner = next(d for d in collide_defs if d["location"]["module"] == "plugin_b")
    assert winner["overridden_by"] == []

    item = report["items"][0]
    fx = next(f for f in item["fixtures"] if f["argname"] == "collide")
    assert [layer["location"]["module"] for layer in fx["chain"]] == [
        "plugin_a",
        "plugin_b",
    ]
    assert fx["winner_index"] == 1
    assert "registration" in fx["winner_reason"]

    # Resolution itself is unchanged: the test passes with the winner.
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_json_partial_override_in_tree(pytester: Pytester) -> None:
    """A conftest fixture overridden by a module fixture is still effective
    for items outside that module: status partially_overridden."""
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture
        def f():
            return "conftest"
        """
    )
    pytester.makepyfile(
        test_with_override="""
        import pytest

        @pytest.fixture
        def f():
            return "module"

        def test_local(f):
            assert f == "module"
        """,
        test_without_override="""
        def test_outer(f):
            assert f == "conftest"
        """,
    )
    audit_path = pytester.path / "audit.json"
    result = pytester.runpytest(
        "--fixture-audit-json",
        str(audit_path),
        "test_with_override.py",
        "test_without_override.py",
    )
    assert result.ret == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))

    fdefs = [d for d in report["definitions"] if d["argname"] == "f"]
    by_module = {d["location"]["module"]: d for d in fdefs}
    assert by_module["conftest"]["status"] == "partially_overridden"
    assert by_module["test_with_override"]["status"] == "active"
    conftest_def = by_module["conftest"]
    assert conftest_def["overridden_by"][0]["relation"] == "more_specific_node"

    local_item = next(i for i in report["items"] if i["nodeid"].endswith("test_local"))
    local_f = next(f for f in local_item["fixtures"] if f["argname"] == "f")
    assert local_f["winner_index"] == 1
    assert local_f["chain"][0]["role"] == "shadowed"
    assert local_f["chain"][1]["role"] == "winner"

    outer_item = next(i for i in report["items"] if i["nodeid"].endswith("test_outer"))
    outer_f = next(f for f in outer_item["fixtures"] if f["argname"] == "f")
    assert outer_f["winner_index"] == 0
    assert outer_f["chain"][0]["defined_at"]["kind"] == "session"


def test_fixtures_listing_marks_overridden(pytester: Pytester) -> None:
    pytester.makepyfile(
        plugin_a="""
        import pytest

        @pytest.fixture
        def collide():
            '''loser'''
            return "a"
        """,
        plugin_b="""
        import pytest

        @pytest.fixture
        def collide():
            '''winner'''
            return "b"
        """,
    )
    pytester.makeconftest(
        """
        pytest_plugins = ("plugin_a", "plugin_b")
        """
    )
    result = pytester.runpytest("--fixtures", "--fixture-audit")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "collide [[]overridden: never takes effect[]]*-- *plugin_a.py*",
            "*OVERRIDDEN: another definition with the same name always takes precedence*",
            "*plugin_b.py*-- session-global node*",
            "collide -- *plugin_b.py*",
        ]
    )
    # The winner line carries no override marker.
    result.stdout.no_fnmatch_line("*[[]overridden*plugin_b*")


def test_fixtures_listing_partial_override_marker(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture
        def f():
            '''from conftest'''
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            '''from module'''

        def test_a(f):
            pass
        """
    )
    result = pytester.runpytest("--fixtures", "--fixture-audit", "-v")
    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "f [[]overridden for some items[]]*-- *conftest.py*",
            "*OVERRIDDEN IN PART*takes effect only for items outside the subtree of*",
            "*test_fixtures_listing_partial_override_marker.py*",
            "f -- *test_fixtures_listing_partial_override_marker.py*",
        ]
    )


def test_plain_fixtures_listing_unchanged(pytester: Pytester) -> None:
    """Without --fixture-audit the listing output is unchanged."""
    pytester.makepyfile(
        plugin_a="""
        import pytest

        @pytest.fixture
        def collide():
            return "a"
        """,
        plugin_b="""
        import pytest

        @pytest.fixture
        def collide():
            return "b"
        """,
    )
    pytester.makeconftest(
        """
        pytest_plugins = ("plugin_a", "plugin_b")
        """
    )
    result = pytester.runpytest("--fixtures")
    assert result.ret == 0
    result.stdout.no_fnmatch_line("*overridden*")


def test_json_stdout_is_pure_json(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            return 1

        def test_a(f):
            pass
        """
    )
    result = pytester.runpytest("--fixture-audit-json", "-", "-q")
    assert result.ret == 0
    report = json.loads(str(result.stdout))
    assert report["schema_version"] == "1.0"
    assert len(report["items"]) == 1


def test_audit_readonly_does_not_change_resolution(pytester: Pytester) -> None:
    """Audit collection must not alter fixture precedence or registration."""
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture
        def f():
            return "conftest"
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def f():
            return "module"

        def test_a(f):
            assert f == "module"
        """
    )
    # Audit first (collect-only), then a normal run in the same environment.
    result = pytester.runpytest("--fixture-audit")
    assert result.ret == 0
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_audit_requested_by_classification(pytester: Pytester) -> None:
    pytester.makeconftest(
        """
        import pytest

        @pytest.fixture(autouse=True)
        def auto1():
            yield

        @pytest.fixture
        def via_marker():
            yield
        """
    )
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def dep():
            return 1

        @pytest.fixture
        def chain(dep):
            return dep

        @pytest.mark.usefixtures("via_marker")
        def test_a(chain):
            pass
        """
    )
    audit_path = pytester.path / "audit.json"
    result = pytester.runpytest("--fixture-audit-json", str(audit_path))
    assert result.ret == 0
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    fx = {f["argname"]: f for f in report["items"][0]["fixtures"]}
    assert fx["chain"]["requested_by"] == "parameter"
    assert fx["dep"]["requested_by"] == "dependency"
    assert fx["via_marker"]["requested_by"] == "usefixtures"
    assert fx["auto1"]["requested_by"] == "autouse"
