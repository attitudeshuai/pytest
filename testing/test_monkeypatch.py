# mypy: allow-untyped-defs
from __future__ import annotations

from collections.abc import Generator
import os
from pathlib import Path
import re
import sys
import textwrap
import threading
from types import MappingProxyType
import warnings

from _pytest.monkeypatch import MonkeyPatch
from _pytest.monkeypatch import MonkeyPatchAssemblyError
from _pytest.pytester import Pytester
import pytest


@pytest.fixture
def mp() -> Generator[MonkeyPatch]:
    cwd = os.getcwd()
    sys_path = list(sys.path)
    yield MonkeyPatch()
    sys.path[:] = sys_path
    os.chdir(cwd)


def test_setattr() -> None:
    class A:
        x = 1

    monkeypatch = MonkeyPatch()
    with pytest.raises(AttributeError):
        monkeypatch.setattr(A, "notexists", 2)
    monkeypatch.setattr(A, "y", 2, raising=False)
    assert A.y == 2  # type: ignore
    monkeypatch.undo()
    assert not hasattr(A, "y")

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(A, "x", 2)
    assert A.x == 2
    monkeypatch.setattr(A, "x", 3)
    assert A.x == 3
    monkeypatch.undo()
    assert A.x == 1

    A.x = 5
    monkeypatch.undo()  # double-undo makes no modification
    assert A.x == 5

    with pytest.raises(TypeError):
        monkeypatch.setattr(A, "y")  # type: ignore[call-overload]


class TestSetattrWithImportPath:
    def test_string_expression(self, monkeypatch: MonkeyPatch) -> None:
        with monkeypatch.context() as mp:
            mp.setattr("os.path.abspath", lambda x: "hello2")
            assert os.path.abspath("123") == "hello2"

    def test_string_expression_class(self, monkeypatch: MonkeyPatch) -> None:
        with monkeypatch.context() as mp:
            mp.setattr("_pytest.config.Config", 42)
            import _pytest

            assert _pytest.config.Config == 42  # type: ignore

    def test_unicode_string(self, monkeypatch: MonkeyPatch) -> None:
        with monkeypatch.context() as mp:
            mp.setattr("_pytest.config.Config", 42)
            import _pytest

            assert _pytest.config.Config == 42  # type: ignore
            mp.delattr("_pytest.config.Config")

    def test_wrong_target(self, monkeypatch: MonkeyPatch) -> None:
        with pytest.raises(TypeError):
            monkeypatch.setattr(None, None)  # type: ignore[call-overload]

    def test_unknown_import(self, monkeypatch: MonkeyPatch) -> None:
        with pytest.raises(ImportError):
            monkeypatch.setattr("unkn123.classx", None)

    def test_unknown_attr(self, monkeypatch: MonkeyPatch) -> None:
        with pytest.raises(AttributeError):
            monkeypatch.setattr("os.path.qweqwe", None)

    def test_unknown_attr_non_raising(self, monkeypatch: MonkeyPatch) -> None:
        # https://github.com/pytest-dev/pytest/issues/746
        with monkeypatch.context() as mp:
            mp.setattr("os.path.qweqwe", 42, raising=False)
            assert os.path.qweqwe == 42  # type: ignore

    def test_setattr_failure_does_not_corrupt_undo(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """If setattr() raises (e.g. target doesn't support attribute
        setting), the undo stack should not contain a stale entry that
        crashes during teardown (#14161)."""

        class Immutable:
            __slots__ = ()

        target = Immutable()
        with pytest.raises(AttributeError):
            monkeypatch.setattr(target, "x", 42, raising=False)
        # undo() must not raise — no entry should be on the undo stack.
        monkeypatch.undo()

    def test_delattr(self, monkeypatch: MonkeyPatch) -> None:
        with monkeypatch.context() as mp:
            mp.delattr("os.path.abspath")
            assert not hasattr(os.path, "abspath")
            mp.undo()
            assert os.path.abspath  # type:ignore[truthy-function]


def test_delattr() -> None:
    class A:
        x = 1

    monkeypatch = MonkeyPatch()
    monkeypatch.delattr(A, "x")
    assert not hasattr(A, "x")
    monkeypatch.undo()
    assert A.x == 1

    monkeypatch = MonkeyPatch()
    monkeypatch.delattr(A, "x")
    with pytest.raises(AttributeError):
        monkeypatch.delattr(A, "y")
    monkeypatch.delattr(A, "y", raising=False)
    monkeypatch.setattr(A, "x", 5, raising=False)
    assert A.x == 5
    monkeypatch.undo()
    assert A.x == 1


def test_setitem() -> None:
    d = {"x": 1}
    monkeypatch = MonkeyPatch()
    monkeypatch.setitem(d, "x", 2)
    monkeypatch.setitem(d, "y", 1700)
    monkeypatch.setitem(d, "y", 1700)
    assert d["x"] == 2
    assert d["y"] == 1700
    monkeypatch.setitem(d, "x", 3)
    assert d["x"] == 3
    monkeypatch.undo()
    assert d["x"] == 1
    assert "y" not in d
    d["x"] = 5
    monkeypatch.undo()
    assert d["x"] == 5


def test_setitem_deleted_meanwhile() -> None:
    d: dict[str, object] = {}
    monkeypatch = MonkeyPatch()
    monkeypatch.setitem(d, "x", 2)
    del d["x"]
    monkeypatch.undo()
    assert not d


@pytest.mark.parametrize("before", [True, False])
def test_setenv_deleted_meanwhile(before: bool) -> None:
    key = "qwpeoip123"
    if before:
        os.environ[key] = "world"
    monkeypatch = MonkeyPatch()
    monkeypatch.setenv(key, "hello")
    del os.environ[key]
    monkeypatch.undo()
    if before:
        assert os.environ[key] == "world"
        del os.environ[key]
    else:
        assert key not in os.environ


def test_delitem() -> None:
    d: dict[str, object] = {"x": 1}
    monkeypatch = MonkeyPatch()
    monkeypatch.delitem(d, "x")
    assert "x" not in d
    monkeypatch.delitem(d, "y", raising=False)
    with pytest.raises(KeyError):
        monkeypatch.delitem(d, "y")
    assert not d
    monkeypatch.setitem(d, "y", 1700)
    assert d["y"] == 1700
    d["hello"] = "world"
    monkeypatch.setitem(d, "x", 1500)
    assert d["x"] == 1500
    monkeypatch.undo()
    assert d == {"hello": "world", "x": 1}


def test_failed_delattr(monkeypatch: MonkeyPatch) -> None:
    """If delattr() raises, no stale undo entry should be recorded (#14909)."""

    class A:
        __slots__ = ()
        x = 1

    a = A()
    with pytest.raises(AttributeError):
        monkeypatch.delattr(a, "x")
    assert a.x == 1
    # undo() must not raise — no entry should be on the undo stack.
    monkeypatch.undo()


def test_failed_setitem(monkeypatch: MonkeyPatch) -> None:
    """If setitem() raises, no stale undo entry should be recorded (#14909)."""
    mapping = MappingProxyType({"x": 1})
    with pytest.raises(TypeError):
        monkeypatch.setitem(mapping, "x", 2)
    assert mapping["x"] == 1
    # undo() must not raise — no entry should be on the undo stack.
    monkeypatch.undo()


def test_failed_delitem(monkeypatch: MonkeyPatch) -> None:
    """If delitem() raises, no stale undo entry should be recorded (#14909)."""
    mapping = MappingProxyType({"x": 1})
    with pytest.raises(TypeError):
        monkeypatch.delitem(mapping, "x")
    assert mapping["x"] == 1
    # undo() must not raise — no entry should be on the undo stack.
    monkeypatch.undo()


def test_setitem_delitem_oldval_captured_before_mutation(
    monkeypatch: MonkeyPatch,
) -> None:
    """For setitem/delitem the old value must be captured *before* the
    mutation so undo() restores the correct value (#14909). This case
    exercises both the capture-before line and the append-after line in
    the success path (no exception), so codecov patch coverage for the
    new lines stays 100% even when the surrounding pytest suite changes.
    """
    inner: dict[str, int] = {"x": 1, "y": 2}
    monkeypatch.setitem(inner, "x", 99)
    monkeypatch.delitem(inner, "y")
    monkeypatch.undo()
    assert inner == {"x": 1, "y": 2}


def test_setenv() -> None:
    monkeypatch = MonkeyPatch()
    with pytest.warns(pytest.PytestWarning):
        monkeypatch.setenv("XYZ123", 2)  # type: ignore[arg-type]
    import os

    assert os.environ["XYZ123"] == "2"
    monkeypatch.undo()
    assert "XYZ123" not in os.environ


def test_delenv() -> None:
    name = "xyz1234"
    assert name not in os.environ
    monkeypatch = MonkeyPatch()
    with pytest.raises(KeyError):
        monkeypatch.delenv(name, raising=True)
    monkeypatch.delenv(name, raising=False)
    monkeypatch.undo()
    os.environ[name] = "1"
    try:
        monkeypatch = MonkeyPatch()
        monkeypatch.delenv(name)
        assert name not in os.environ
        monkeypatch.setenv(name, "3")
        assert os.environ[name] == "3"
        monkeypatch.undo()
        assert os.environ[name] == "1"
    finally:
        if name in os.environ:
            del os.environ[name]


class TestEnvironWarnings:
    """
    os.environ keys and values should be native strings, otherwise it will cause problems with other modules (notably
    subprocess). On Python 2 os.environ accepts anything without complaining, while Python 3 does the right thing
    and raises an error.
    """

    VAR_NAME = "PYTEST_INTERNAL_MY_VAR"

    def test_setenv_non_str_warning(self, monkeypatch: MonkeyPatch) -> None:
        value = 2
        msg = (
            "Value of environment variable PYTEST_INTERNAL_MY_VAR type should be str, "
            "but got 2 (type: int); converted to str implicitly"
        )
        with pytest.warns(pytest.PytestWarning, match=re.escape(msg)):
            monkeypatch.setenv(str(self.VAR_NAME), value)  # type: ignore[arg-type]


def test_setenv_prepend() -> None:
    import os

    monkeypatch = MonkeyPatch()
    monkeypatch.setenv("XYZ123", "2", prepend="-")
    monkeypatch.setenv("XYZ123", "3", prepend="-")
    assert os.environ["XYZ123"] == "3-2"
    monkeypatch.undo()
    assert "XYZ123" not in os.environ


def test_monkeypatch_plugin(pytester: Pytester) -> None:
    reprec = pytester.inline_runsource(
        """
        def test_method(monkeypatch):
            assert monkeypatch.__class__.__name__ == "MonkeyPatch"
    """
    )
    res = reprec.countoutcomes()
    assert tuple(res) == (1, 0, 0), res


def test_syspath_prepend(mp: MonkeyPatch) -> None:
    old = list(sys.path)
    mp.syspath_prepend("world")
    mp.syspath_prepend("hello")
    assert sys.path[0] == "hello"
    assert sys.path[1] == "world"
    mp.undo()
    assert sys.path == old
    mp.undo()
    assert sys.path == old


def test_syspath_prepend_double_undo(mp: MonkeyPatch) -> None:
    old_syspath = sys.path[:]
    try:
        mp.syspath_prepend("hello world")
        mp.undo()
        sys.path.append("more hello world")
        mp.undo()
        assert sys.path[-1] == "more hello world"
    finally:
        sys.path[:] = old_syspath


def test_chdir_with_path_local(mp: MonkeyPatch, tmp_path: Path) -> None:
    mp.chdir(tmp_path)
    assert os.getcwd() == str(tmp_path)


def test_chdir_with_str(mp: MonkeyPatch, tmp_path: Path) -> None:
    mp.chdir(str(tmp_path))
    assert os.getcwd() == str(tmp_path)


def test_chdir_undo(mp: MonkeyPatch, tmp_path: Path) -> None:
    cwd = os.getcwd()
    mp.chdir(tmp_path)
    mp.undo()
    assert os.getcwd() == cwd


def test_chdir_double_undo(mp: MonkeyPatch, tmp_path: Path) -> None:
    mp.chdir(str(tmp_path))
    mp.undo()
    os.chdir(tmp_path)
    mp.undo()
    assert os.getcwd() == str(tmp_path)


def test_issue185_time_breaks(pytester: Pytester) -> None:
    pytester.makepyfile(
        """
        import time
        def test_m(monkeypatch):
            def f():
                raise Exception
            monkeypatch.setattr(time, "time", f)
    """
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        """
        *1 passed*
    """
    )


def test_importerror(pytester: Pytester) -> None:
    p = pytester.mkpydir("package")
    p.joinpath("a.py").write_text(
        textwrap.dedent(
            """\
        import doesnotexist

        x = 1
    """
        ),
        encoding="utf-8",
    )
    pytester.path.joinpath("test_importerror.py").write_text(
        textwrap.dedent(
            """\
        def test_importerror(monkeypatch):
            monkeypatch.setattr('package.a.x', 2)
    """
        ),
        encoding="utf-8",
    )
    result = pytester.runpytest()
    result.stdout.fnmatch_lines(
        """
        *import error in package.a: No module named 'doesnotexist'*
    """
    )


class Sample:
    @staticmethod
    def hello() -> bool:
        return True


class SampleInherit(Sample):
    pass


@pytest.mark.parametrize(
    "Sample",
    [Sample, SampleInherit],
    ids=["new", "new-inherit"],
)
def test_issue156_undo_staticmethod(Sample: type[Sample]) -> None:
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(Sample, "hello", None)
    assert Sample.hello is None

    monkeypatch.undo()  # type: ignore[unreachable]
    assert Sample.hello()


def test_undo_class_descriptors_delattr() -> None:
    class SampleParent:
        @classmethod
        def hello(_cls):
            pass

        @staticmethod
        def world():
            pass

    class SampleChild(SampleParent):
        pass

    monkeypatch = MonkeyPatch()

    original_hello = SampleChild.hello
    original_world = SampleChild.world
    monkeypatch.delattr(SampleParent, "hello")
    monkeypatch.delattr(SampleParent, "world")
    assert getattr(SampleParent, "hello", None) is None
    assert getattr(SampleParent, "world", None) is None

    monkeypatch.undo()
    assert original_hello == SampleChild.hello
    assert original_world == SampleChild.world


def test_undo_inherited_attribute_on_instance() -> None:
    """Undo must not leave an inherited attribute behind in the instance dict.

    See #10644.
    """

    class Parent:
        x = 1

    class Child(Parent):
        pass

    obj = Child()
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(obj, "x", 2)
    assert obj.x == 2
    assert vars(obj)["x"] == 2

    monkeypatch.undo()
    assert obj.x == 1
    assert "x" not in vars(obj)


def test_undo_inherited_non_data_descriptor_on_instance() -> None:
    """Undo must not freeze a descriptor which resolves dynamically.

    See #10644.
    """

    class Dynamic:
        def __init__(self) -> None:
            self.calls = 0

        def __get__(self, instance: object, owner: type | None = None) -> int:
            self.calls += 1
            return self.calls

    class Sample:
        value = Dynamic()

    obj = Sample()
    first, second = obj.value, obj.value
    assert first != second

    monkeypatch = MonkeyPatch()
    monkeypatch.setattr(obj, "value", -1)
    assert obj.value == -1

    monkeypatch.undo()
    assert "value" not in vars(obj)
    third, fourth = obj.value, obj.value
    assert third != fourth


def test_undo_inherited_method_on_instance() -> None:
    """Patching a method on an instance must not leave a bound method behind."""

    class Sample:
        def hello(self) -> str:
            return "hello"

    obj = Sample()
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(obj, "hello", lambda: "patched")
    assert obj.hello() == "patched"

    monkeypatch.undo()
    assert obj.hello() == "hello"
    assert "hello" not in vars(obj)


def test_undo_own_attribute_on_instance() -> None:
    """An attribute owned by the instance is still restored to its old value."""

    class Sample:
        x = "class"

        def __init__(self) -> None:
            self.x = "instance"

    obj = Sample()
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(obj, "x", "patched")
    assert obj.x == "patched"

    monkeypatch.undo()
    assert vars(obj)["x"] == "instance"


def test_undo_data_descriptor_on_instance() -> None:
    """A data descriptor owns the attribute, so the old value is set back through it."""

    class Sample:
        def __init__(self) -> None:
            self._x = 1

        @property
        def x(self) -> int:
            return self._x

        @x.setter
        def x(self, value: int) -> None:
            self._x = value

    obj = Sample()
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(obj, "x", 2)
    assert obj.x == 2

    monkeypatch.undo()
    assert obj.x == 1
    assert "x" not in vars(obj)


def test_undo_slot_attribute_on_instance() -> None:
    """Slot descriptors are data descriptors, so undo restores the slot value."""

    class Sample:
        __slots__ = ("x",)

        def __init__(self) -> None:
            self.x = 1

    obj = Sample()
    monkeypatch = MonkeyPatch()

    monkeypatch.setattr(obj, "x", 2)
    assert obj.x == 2

    monkeypatch.undo()
    assert obj.x == 1


def test_issue1338_name_resolving() -> None:
    pytest.importorskip("requests")
    monkeypatch = MonkeyPatch()
    try:
        monkeypatch.delattr("requests.sessions.Session.request")
    finally:
        monkeypatch.undo()


def test_context() -> None:
    monkeypatch = MonkeyPatch()

    import functools
    import inspect

    with monkeypatch.context() as m:
        m.setattr(functools, "partial", 3)
        assert not inspect.isclass(functools.partial)
    assert inspect.isclass(functools.partial)  # type: ignore[unreachable]


def test_context_classmethod() -> None:
    class A:
        x = 1

    with MonkeyPatch.context() as m:
        m.setattr(A, "x", 2)
        assert A.x == 2
    assert A.x == 1


@pytest.mark.filterwarnings(
    r"ignore:.*\bpkg_resources\b:DeprecationWarning",
    r"ignore:.*\bpkg_resources\b:UserWarning",
)
def test_syspath_prepend_with_namespace_packages(
    pytester: Pytester, monkeypatch: MonkeyPatch
) -> None:
    # Needs to be in sys.modules.
    pytest.importorskip("pkg_resources")

    for dirname in "hello", "world":
        d = pytester.mkdir(dirname)
        ns = d.joinpath("ns_pkg")
        ns.mkdir()
        ns.joinpath("__init__.py").write_text(
            "__import__('pkg_resources').declare_namespace(__name__)", encoding="utf-8"
        )
        lib = ns.joinpath(dirname)
        lib.mkdir()
        lib.joinpath("__init__.py").write_text(
            f"def check(): return {dirname!r}", encoding="utf-8"
        )

    # First call should not warn - namespace package not registered yet.
    monkeypatch.syspath_prepend("hello")
    # This registers ns_pkg as a namespace package.
    import ns_pkg.hello

    assert ns_pkg.hello.check() == "hello"

    with pytest.raises(ImportError):
        import ns_pkg.world

    # Prepending should call fixup_namespace_packages.
    # This call should warn - ns_pkg is now registered and "world" contains it
    with pytest.warns(pytest.PytestRemovedIn10Warning, match="legacy namespace"):
        monkeypatch.syspath_prepend("world")
    import ns_pkg.world

    assert ns_pkg.world.check() == "world"

    # Should invalidate caches via importlib.invalidate_caches.
    # Should not warn for path without namespace packages.
    modules_tmpdir = pytester.mkdir("modules_tmpdir")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monkeypatch.syspath_prepend(str(modules_tmpdir))

    modules_tmpdir.joinpath("main_app.py").write_text("app = True", encoding="utf-8")
    from main_app import app  # noqa: F401


class TestMonkeyPatchAssembly:
    def test_commit_then_parent_undo(self, mp: MonkeyPatch) -> None:
        class C:
            a = 1

        data = {"x": 1}
        try:
            with mp.assembly() as unit:
                unit.setattr(C, "a", 2)
                unit.setitem(data, "y", 2)
                unit.setenv("PYTEST_ASSEMBLY_TEST_VAR", "hello")
            # Changes stay in effect after the block: committed to parent.
            assert C.a == 2
            assert data == {"x": 1, "y": 2}
            assert os.environ["PYTEST_ASSEMBLY_TEST_VAR"] == "hello"

            mp.undo()
            assert C.a == 1
            assert data == {"x": 1}
            assert "PYTEST_ASSEMBLY_TEST_VAR" not in os.environ
        finally:
            os.environ.pop("PYTEST_ASSEMBLY_TEST_VAR", None)

    def test_failure_rolls_back_prior_steps(self, mp: MonkeyPatch) -> None:
        class C:
            a = 1
            b = 2

        data = {"x": 1}
        with pytest.raises(MonkeyPatchAssemblyError) as excinfo:
            with mp.assembly() as unit:
                unit.setattr(C, "a", 10)
                unit.setitem(data, "y", 20)
                unit.setattr(C, "no_such_attr", 30)

        error = excinfo.value
        assert error.step == 3
        assert error.operation is not None
        assert "setattr" in error.operation
        assert "no_such_attr" in error.operation
        assert isinstance(error.original, AttributeError)
        assert error.__cause__ is error.original
        assert error.rollback_errors == []
        # Already-applied steps were rolled back.
        assert C.a == 1
        assert data == {"x": 1}
        # Parent journal is not polluted: undo is a no-op and safe.
        mp.undo()
        assert C.a == 1
        assert data == {"x": 1}

    def test_failure_location_for_dotted_target(self, mp: MonkeyPatch) -> None:
        with pytest.raises(MonkeyPatchAssemblyError) as excinfo:
            with mp.assembly() as unit:
                unit.setattr("os.path.definitely_not_here", 1)
        error = excinfo.value
        assert error.step == 1
        assert error.operation is not None
        assert "os.path.definitely_not_here" in error.operation
        assert "step 1" in str(error)

    def test_named_unit_appears_in_error(self, mp: MonkeyPatch) -> None:
        class C:
            pass

        with pytest.raises(MonkeyPatchAssemblyError, match="env-setup") as excinfo:
            with mp.assembly(name="env-setup") as unit:
                unit.setattr(C, "a", 1, raising=False)
                unit.setattr(C, "missing", 2)
        assert excinfo.value.name == "env-setup"

    def test_env_helpers_count_as_a_single_step(self, mp: MonkeyPatch) -> None:
        # setenv()/delenv() apply themselves through setitem()/delitem();
        # that internal self-call must not be counted as another step.
        try:
            with mp.assembly() as unit:
                unit.setenv("PYTEST_ASSEMBLY_STEP_VAR", "1")
                unit.delenv("PYTEST_ASSEMBLY_STEP_VAR_MISSING", raising=False)
                unit.setattr("os.path.no_such_attr_xyz", 1)
        except MonkeyPatchAssemblyError as error:
            assert error.step == 3
            assert error.operation is not None
            assert error.operation.startswith("setattr")
        else:  # pragma: no cover
            pytest.fail("assembly did not fail")
        finally:
            os.environ.pop("PYTEST_ASSEMBLY_STEP_VAR", None)

    def test_rollback_is_reverse_order(self, mp: MonkeyPatch) -> None:
        order: list[tuple[str, object]] = []

        class Logger:
            a: int
            b: int

            def __setattr__(self, key: str, value: object) -> None:
                order.append((key, value))
                object.__setattr__(self, key, value)

        obj = Logger()
        object.__setattr__(obj, "a", 1)
        object.__setattr__(obj, "b", 2)

        with pytest.raises(MonkeyPatchAssemblyError):
            with mp.assembly() as unit:
                unit.setattr(obj, "a", 10)
                unit.setattr(obj, "b", 20)
                unit.setattr("os.path.no_such_attr_xyz", 1)

        # Apply order a, b; reverse-order restore b, a.
        assert order == [("a", 10), ("b", 20), ("b", 2), ("a", 1)]
        assert obj.a == 1
        assert obj.b == 2

    def test_rollback_continues_after_restore_failure(self, mp: MonkeyPatch) -> None:
        class FlakyObject:
            boom: str
            ok: str

            def __setattr__(self, key: str, value: object) -> None:
                if key == "boom" and value == "orig":
                    raise RuntimeError("cannot restore attribute")
                object.__setattr__(self, key, value)

        class FlakyDict(dict):  # type: ignore[type-arg]
            def __setitem__(self, key: object, value: object) -> None:
                if key == "boom" and value == "orig":
                    raise RuntimeError("cannot restore item")
                super().__setitem__(key, value)

        class Plain:
            p = "orig"

        flaky_obj = FlakyObject()
        object.__setattr__(flaky_obj, "boom", "orig")
        flaky_obj.ok = "orig_ok"
        flaky_dict: FlakyDict = FlakyDict()
        dict.update(flaky_dict, {"boom": "orig", "ok": "orig_ok"})
        plain_dict: dict[str, int] = {}

        with pytest.raises(MonkeyPatchAssemblyError) as excinfo:
            with mp.assembly() as unit:
                unit.setattr(flaky_obj, "boom", "new")
                unit.setattr(flaky_obj, "ok", "new_ok")
                unit.setattr(Plain, "p", "new")
                unit.setitem(flaky_dict, "boom", "new")
                unit.setitem(flaky_dict, "ok", "new_ok")
                unit.setitem(plain_dict, "x", 1)
                unit.setattr(Plain, "missing_attr", 1)

        error = excinfo.value
        # The single failing restore did not abort the remaining restores.
        assert Plain.p == "orig"
        assert flaky_obj.ok == "orig_ok"
        assert flaky_dict["ok"] == "orig_ok"
        assert "x" not in plain_dict
        # Unrestorable changes keep their patched values.
        assert flaky_obj.boom == "new"
        assert flaky_dict["boom"] == "new"
        # Restore failures are reported collectively.
        assert len(error.rollback_errors) == 2
        descriptions = [desc for desc, _ in error.rollback_errors]
        assert any("boom" in desc for desc in descriptions)
        assert all(isinstance(exc, RuntimeError) for _, exc in error.rollback_errors)
        assert "RuntimeError" in str(error)

    def test_delattr_and_delitem_rolled_back(self, mp: MonkeyPatch) -> None:
        class C:
            a = 1

        data = {"x": 1}
        with pytest.raises(MonkeyPatchAssemblyError):
            with mp.assembly() as unit:
                unit.delattr(C, "a")
                unit.delitem(data, "x")
                unit.setattr(C, "missing_attr", 2)
        assert C.a == 1
        assert data["x"] == 1

    def test_user_error_between_steps_propagates_unchanged(
        self, mp: MonkeyPatch
    ) -> None:
        class C:
            a = 1

        class MidError(Exception):
            pass

        with pytest.raises(MidError) as excinfo:
            with mp.assembly() as unit:
                unit.setattr(C, "a", 2)
                raise MidError("boom")
        assert type(excinfo.value) is MidError
        assert C.a == 1

    def test_user_error_plus_restore_failure_reports_both(
        self, mp: MonkeyPatch
    ) -> None:
        class FlakyObject:
            boom: str

            def __setattr__(self, key: str, value: object) -> None:
                if key == "boom" and value == "orig":
                    raise RuntimeError("cannot restore")
                object.__setattr__(self, key, value)

        class MidError(Exception):
            pass

        flaky_obj = FlakyObject()
        object.__setattr__(flaky_obj, "boom", "orig")

        with pytest.raises(MonkeyPatchAssemblyError) as excinfo:
            with mp.assembly(name="flaky-unit") as unit:
                unit.setattr(flaky_obj, "boom", "new")
                raise MidError("mid")
        error = excinfo.value
        assert error.step is None
        assert isinstance(error.original, MidError)
        assert isinstance(error.__cause__, MidError)
        assert len(error.rollback_errors) == 1
        assert "boom" in error.rollback_errors[0][0]

    def test_nested_inner_failure_isolated(self, mp: MonkeyPatch) -> None:
        class C:
            a = 1

        with mp.assembly() as outer:
            outer.setattr(C, "a", 2)
            with pytest.raises(MonkeyPatchAssemblyError):
                with outer.assembly(name="inner") as inner:
                    inner.setattr(C, "b", 3, raising=False)
                    inner.setattr(C, "missing_attr", 4)
            # Inner rollback does not touch outer changes.
            assert C.a == 2
            assert not hasattr(C, "b")
            outer.setattr(C, "c", 4, raising=False)

        assert C.a == 2
        assert C.c == 4  # type: ignore[attr-defined]
        assert not hasattr(C, "b")
        mp.undo()
        assert C.a == 1
        assert not hasattr(C, "b")
        assert not hasattr(C, "c")

    def test_nested_inner_success_outer_failure_rolls_back_all(
        self, mp: MonkeyPatch
    ) -> None:
        class C:
            a = 1

        with pytest.raises(MonkeyPatchAssemblyError):
            with mp.assembly() as outer:
                outer.setattr(C, "a", 2)
                with outer.assembly() as inner:
                    inner.setattr(C, "b", 3, raising=False)
                assert C.b == 3  # type: ignore[attr-defined]
                outer.setattr(C, "missing_attr", 4)
        assert C.a == 1
        assert not hasattr(C, "b")

    def test_failed_assembly_keeps_parent_patches(self, mp: MonkeyPatch) -> None:
        class C:
            pre = 0

        mp.setattr(C, "pre", 1, raising=False)
        with pytest.raises(MonkeyPatchAssemblyError):
            with mp.assembly() as unit:
                unit.setattr(C, "inside", 2, raising=False)
                unit.setattr(C, "missing_attr", 3)
        assert C.pre == 1
        assert not hasattr(C, "inside")
        mp.undo()
        assert C.pre == 0

    def test_syspath_prepend_and_chdir_committed(
        self, mp: MonkeyPatch, tmp_path: Path
    ) -> None:
        old_path = sys.path[:]
        old_cwd = os.getcwd()
        try:
            with mp.assembly() as unit:
                unit.chdir(tmp_path)
                unit.syspath_prepend(str(tmp_path))
            assert os.getcwd() == str(tmp_path)
            assert sys.path[0] == str(tmp_path)

            mp.undo()
            assert os.getcwd() == old_cwd
            assert sys.path == old_path
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path

    def test_concurrent_assemblies_are_isolated(self, mp: MonkeyPatch) -> None:
        class C:
            pass

        data: dict[str, int] = {}
        errors: list[BaseException] = []
        failed: list[int] = []
        worker_count = 12
        barrier = threading.Barrier(worker_count)

        def worker(i: int) -> None:
            barrier.wait()
            try:
                with mp.assembly(name=f"unit-{i}") as unit:
                    unit.setattr(C, f"attr_{i}", i, raising=False)
                    unit.setitem(data, f"key_{i}", i)
                    if i == 3:
                        unit.setattr(C, "totally_missing_attr", 99)
            except MonkeyPatchAssemblyError as exc:
                failed.append(i)
                errors.append(exc)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failed == [3]
        assert len(errors) == 1
        for i in range(worker_count):
            if i == 3:
                assert not hasattr(C, "attr_3")
                assert "key_3" not in data
            else:
                assert getattr(C, f"attr_{i}") == i
                assert data[f"key_{i}"] == i

        mp.undo()
        for i in range(worker_count):
            assert not hasattr(C, f"attr_{i}")
            assert f"key_{i}" not in data
