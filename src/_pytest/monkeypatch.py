# mypy: allow-untyped-defs
"""Monkeypatching and mocking functionality."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import MutableMapping
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import re
import sys
import threading
from types import TracebackType
from typing import Any
from typing import cast
from typing import final
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
import warnings


if TYPE_CHECKING:
    from typing_extensions import Self

from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.deprecated import MONKEYPATCH_LEGACY_NAMESPACE_PACKAGES
from _pytest.fixtures import fixture
from _pytest.warning_types import PytestWarning


RE_IMPORT_ERROR_NAME = re.compile(r"^No module named (.*)$")


K = TypeVar("K")
V = TypeVar("V")


@fixture
def monkeypatch() -> Generator[MonkeyPatch]:
    """A convenient fixture for monkey-patching.

    The fixture provides these methods to modify objects, dictionaries, or
    :data:`os.environ`:

    * :meth:`monkeypatch.setattr(obj, name, value, raising=True) <pytest.MonkeyPatch.setattr>`
    * :meth:`monkeypatch.delattr(obj, name, raising=True) <pytest.MonkeyPatch.delattr>`
    * :meth:`monkeypatch.setitem(mapping, name, value) <pytest.MonkeyPatch.setitem>`
    * :meth:`monkeypatch.delitem(obj, name, raising=True) <pytest.MonkeyPatch.delitem>`
    * :meth:`monkeypatch.setenv(name, value, prepend=None) <pytest.MonkeyPatch.setenv>`
    * :meth:`monkeypatch.delenv(name, raising=True) <pytest.MonkeyPatch.delenv>`
    * :meth:`monkeypatch.syspath_prepend(path) <pytest.MonkeyPatch.syspath_prepend>`
    * :meth:`monkeypatch.chdir(path) <pytest.MonkeyPatch.chdir>`
    * :meth:`monkeypatch.context() <pytest.MonkeyPatch.context>`
    * :meth:`monkeypatch.assembly() <pytest.MonkeyPatch.assembly>`

    All modifications will be undone after the requesting test function or
    fixture has finished. The ``raising`` parameter determines if a :class:`KeyError`
    or :class:`AttributeError` will be raised if the set/deletion operation does not have the
    specified target.

    To undo modifications done by the fixture in a contained scope,
    use :meth:`context() <pytest.MonkeyPatch.context>`.
    """
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


def resolve(name: str) -> object:
    # Simplified from zope.dottedname.
    parts = name.split(".")

    used = parts.pop(0)
    found: object = importlib.import_module(used)
    for part in parts:
        used += "." + part
        try:
            found = getattr(found, part)
        except AttributeError:
            pass
        else:
            continue
        # We use explicit un-nesting of the handling block in order
        # to avoid nested exceptions.
        try:
            importlib.import_module(used)
        except ImportError as ex:
            expected = str(ex).split()[-1]
            if expected == used:
                raise
            else:
                raise ImportError(f"import error in {used}: {ex}") from ex
        found = annotated_getattr(found, part, used)
    return found


def annotated_getattr(obj: object, name: str, ann: str) -> object:
    try:
        obj = getattr(obj, name)
    except AttributeError as e:
        raise AttributeError(
            f"{type(obj).__name__!r} object at {ann} has no attribute {name!r}"
        ) from e
    return obj


def derive_importpath(import_path: str, raising: bool) -> tuple[str, object]:
    if not isinstance(import_path, str) or "." not in import_path:
        raise TypeError(f"must be absolute import path string, not {import_path!r}")
    module, attr = import_path.rsplit(".", 1)
    target = resolve(module)
    if raising:
        annotated_getattr(target, attr, ann=module)
    return attr, target


def _is_data_descriptor(cls: type, name: str) -> bool:
    """Return True if looking up ``name`` on ``cls`` finds a data descriptor.

    Data descriptors take precedence over the instance ``__dict__``, so
    ``setattr()`` on an instance is routed through the descriptor instead of
    writing an entry into the instance ``__dict__``.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            descr_type = type(klass.__dict__[name])
            return hasattr(descr_type, "__set__") or hasattr(descr_type, "__delete__")
    return False


@final
class MonkeyPatch:
    """Helper to conveniently monkeypatch attributes/items/environment
    variables/syspath.

    Returned by the :fixture:`monkeypatch` fixture.

    .. versionchanged:: 6.2
        Can now also be used directly as `pytest.MonkeyPatch()`, for when
        the fixture is not available. In this case, use
        :meth:`with MonkeyPatch.context() as mp: <context>` or remember to call
        :meth:`undo` explicitly.
    """

    def __init__(self) -> None:
        self._setattr: list[tuple[object, str, object]] = []
        self._setitem: list[tuple[Mapping[Any, Any], object, object]] = []
        self._cwd: str | None = None
        self._savesyspath: list[str] | None = None
        # Serializes journal commits from concurrent transactional
        # assemblies; the single-point patch/undo paths do not take it.
        self._assembly_lock = threading.RLock()

    @classmethod
    @contextmanager
    def context(cls) -> Generator[MonkeyPatch]:
        """Context manager that returns a new :class:`MonkeyPatch` object
        which undoes any patching done inside the ``with`` block upon exit.

        Example:

        .. code-block:: python

            import functools


            def test_partial(monkeypatch):
                with monkeypatch.context() as m:
                    m.setattr(functools, "partial", 3)

        Useful in situations where it is desired to undo some patches before the test ends,
        such as mocking ``stdlib`` functions that might break pytest itself if mocked (for examples
        of this see :issue:`3290`).
        """
        m = cls()
        try:
            yield m
        finally:
            m.undo()

    def assembly(self, name: str | None = None) -> MonkeyPatchAssembly:
        """Open a transactional assembly unit scoped to this patch object.

        Returns a context manager which offers the same patch methods as
        :class:`MonkeyPatch` (``setattr``/``delattr``/``setitem``/
        ``delitem``/``setenv``/``delenv``/``syspath_prepend``/``chdir``).

        Changes inside the ``with`` block take effect immediately, but are
        journaled locally by the unit instead of on this object:

        * When the block finishes without error, all changes are committed
          to this patch object, so they are undone together with it (test
          tear-down, :meth:`undo`, or an enclosing assembly).
        * When any patch step fails, the changes already applied inside
          the unit are rolled back in reverse order and a
          :class:`MonkeyPatchAssemblyError` is raised. The error identifies
          the failing step (1-based index and a readable operation
          description) and keeps the original exception as its cause.
        * If restoring a single change fails, the rollback continues with
          the remaining changes and reports every restore failure
          collectively on the raised error instead of aborting halfway.
        * Units may be nested: a failing inner unit rolls back only its own
          changes and never pollutes the enclosing journal. Separate
          units, including units used concurrently from different threads,
          keep independent journals.

        :param name:
            Optional label of the unit, included in error reports.

        Example:

        .. code-block:: python

            with monkeypatch.assembly("feature-x environment") as patch:
                patch.setenv("FEATURE_X", "1")
                patch.setattr("os.getcwd", lambda: "/tmp")
        """
        return MonkeyPatchAssembly(self, name=name)

    @overload
    def setattr(
        self,
        target: str,
        name: object,
        value: NotSetType = ...,
        raising: bool = ...,
    ) -> None: ...

    @overload
    def setattr(
        self,
        target: object,
        name: str,
        value: object,
        raising: bool = ...,
    ) -> None: ...

    def setattr(
        self,
        target: str | object,
        name: object | str,
        value: object = NOTSET,
        raising: bool = True,
    ) -> None:
        """
        Set attribute value on target, memorizing the old value.

        For example:

        .. code-block:: python

            import os

            monkeypatch.setattr(os, "getcwd", lambda: "/")

        The code above replaces the :func:`os.getcwd` function by a ``lambda`` which
        always returns ``"/"``.

        For convenience, you can specify a string as ``target`` which
        will be interpreted as a dotted import path, with the last part
        being the attribute name:

        .. code-block:: python

            monkeypatch.setattr("os.getcwd", lambda: "/")

        Raises :class:`AttributeError` if the attribute does not exist, unless
        ``raising`` is set to False.

        **Where to patch**

        ``monkeypatch.setattr`` works by (temporarily) changing the object that a name points to with another one.
        There can be many names pointing to any individual object, so for patching to work you must ensure
        that you patch the name used by the system under test.

        See the section :ref:`Where to patch <python:where-to-patch>` in the :mod:`unittest.mock`
        docs for a complete explanation, which is meant for :func:`unittest.mock.patch` but
        applies to ``monkeypatch.setattr`` as well.
        """
        __tracebackhide__ = True
        import inspect

        if value is NOTSET:
            if not isinstance(target, str):
                raise TypeError(
                    "use setattr(target, name, value) or "
                    "setattr(target, value) with target being a dotted "
                    "import string"
                )
            value = name
            name, target = derive_importpath(target, raising)
        else:
            if not isinstance(name, str):
                raise TypeError(
                    "use setattr(target, name, value) with name being a string or "
                    "setattr(target, value) with target being a dotted "
                    "import string"
                )

        oldval = getattr(target, name, NOTSET)
        if raising and oldval is NOTSET:
            raise AttributeError(f"{target!r} has no attribute {name!r}")

        # avoid class descriptors like staticmethod/classmethod
        if inspect.isclass(target):
            oldval = target.__dict__.get(name, NOTSET)
        elif not _is_data_descriptor(type(target), name):
            # With no data descriptor in the way, the `setattr()` below writes
            # into the instance `__dict__`, so `undo()` has to restore that
            # `__dict__` entry. Assigning an inherited `oldval` back onto the
            # instance would instead leave behind a new entry shadowing the
            # class attribute, which permanently freezes descriptors that
            # resolve dynamically (#10644).
            target_dict = getattr(target, "__dict__", None)
            if isinstance(target_dict, Mapping):
                oldval = target_dict.get(name, NOTSET)
        setattr(target, name, value)
        self._setattr.append((target, name, oldval))

    def delattr(
        self,
        target: object | str,
        name: str | NotSetType = NOTSET,
        raising: bool = True,
    ) -> None:
        """Delete attribute ``name`` from ``target``.

        If no ``name`` is specified and ``target`` is a string
        it will be interpreted as a dotted import path with the
        last part being the attribute name.

        Raises AttributeError it the attribute does not exist, unless
        ``raising`` is set to False.
        """
        __tracebackhide__ = True
        import inspect

        if name is NOTSET:
            if not isinstance(target, str):
                raise TypeError(
                    "use delattr(target, name) or "
                    "delattr(target) with target being a dotted "
                    "import string"
                )
            name, target = derive_importpath(target, raising)

        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
        else:
            oldval = getattr(target, name, NOTSET)
            # Avoid class descriptors like staticmethod/classmethod.
            if inspect.isclass(target):
                oldval = target.__dict__.get(name, NOTSET)
            delattr(target, name)
            self._setattr.append((target, name, oldval))

    def setitem(self, dic: Mapping[K, V], name: K, value: V) -> None:
        """Set dictionary entry ``name`` to value."""
        oldval = dic.get(name, NOTSET)
        # Not all Mapping types support indexing, but MutableMapping doesn't support TypedDict
        dic[name] = value  # type: ignore[index]
        self._setitem.append((dic, name, oldval))

    def delitem(self, dic: Mapping[K, V], name: K, raising: bool = True) -> None:
        """Delete ``name`` from dict.

        Raises ``KeyError`` if it doesn't exist, unless ``raising`` is set to
        False.
        """
        if name not in dic:
            if raising:
                raise KeyError(name)
        else:
            oldval = dic.get(name, NOTSET)
            # Not all Mapping types support indexing, but MutableMapping doesn't support TypedDict
            del dic[name]  # type: ignore[attr-defined]
            self._setitem.append((dic, name, oldval))

    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        """Set environment variable ``name`` to ``value``.

        If ``prepend`` is a character, read the current environment variable
        value and prepend the ``value`` adjoined with the ``prepend``
        character.
        """
        if not isinstance(value, str):
            warnings.warn(  # type: ignore[unreachable]
                PytestWarning(
                    f"Value of environment variable {name} type should be str, but got "
                    f"{value!r} (type: {type(value).__name__}); converted to str implicitly"
                ),
                stacklevel=2,
            )
            value = str(value)
        if prepend and name in os.environ:
            value = value + prepend + os.environ[name]
        self.setitem(os.environ, name, value)

    def delenv(self, name: str, raising: bool = True) -> None:
        """Delete ``name`` from the environment.

        Raises ``KeyError`` if it does not exist, unless ``raising`` is set to
        False.
        """
        environ: MutableMapping[str, str] = os.environ
        self.delitem(environ, name, raising=raising)

    def syspath_prepend(self, path) -> None:
        """Prepend ``path`` to ``sys.path`` list of import locations."""
        if self._savesyspath is None:
            self._savesyspath = sys.path[:]
        sys.path.insert(0, str(path))

        # https://github.com/pypa/setuptools/blob/d8b901bc/docs/pkg_resources.txt#L162-L171
        # this is only needed when pkg_resources was already loaded by the namespace package
        if "pkg_resources" in sys.modules:
            import pkg_resources
            from pkg_resources import fixup_namespace_packages

            # Only issue deprecation warning if this call would actually have an
            # effect for this specific path.
            if (
                hasattr(pkg_resources, "_namespace_packages")
                and pkg_resources._namespace_packages
            ):
                path_obj = Path(str(path))
                for ns_pkg in pkg_resources._namespace_packages:
                    if ns_pkg is None:
                        continue
                    ns_pkg_path = path_obj / ns_pkg.replace(".", os.sep)
                    if ns_pkg_path.is_dir():
                        warnings.warn(
                            MONKEYPATCH_LEGACY_NAMESPACE_PACKAGES.format(), stacklevel=2
                        )
                        break

            fixup_namespace_packages(str(path))

        # A call to syspathinsert() usually means that the caller wants to
        # import some dynamically created files, thus with python3 we
        # invalidate its import caches.
        # This is especially important when any namespace package is in use,
        # since then the mtime based FileFinder cache (that gets created in
        # this case already) gets not invalidated when writing the new files
        # quickly afterwards.
        from importlib import invalidate_caches

        invalidate_caches()

    def chdir(self, path: str | os.PathLike[str]) -> None:
        """Change the current working directory to the specified path.

        :param path:
            The path to change into.
        """
        if self._cwd is None:
            self._cwd = os.getcwd()
        os.chdir(path)

    def undo(self) -> None:
        """Undo previous changes.

        This call consumes the undo stack. Calling it a second time has no
        effect unless you do more monkeypatching after the undo call.

        There is generally no need to call `undo()`, since it is
        called automatically during tear-down.

        .. note::
            The same `monkeypatch` fixture is used across a
            single test function invocation. If `monkeypatch` is used both by
            the test function itself and one of the test fixtures,
            calling `undo()` will undo all of the changes made in
            both functions.

            Prefer to use :meth:`context() <pytest.MonkeyPatch.context>` instead.
        """
        for obj, name, value in reversed(self._setattr):
            if value is not NOTSET:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self._setattr[:] = []
        for dictionary, key, value in reversed(self._setitem):
            if value is NOTSET:
                try:
                    # Not all Mapping types support indexing, but MutableMapping doesn't support TypedDict
                    del dictionary[key]
                except KeyError:
                    pass  # Was already deleted, so we have the desired state.
            else:
                # Not all Mapping types support indexing, but MutableMapping doesn't support TypedDict
                dictionary[key] = value
        self._setitem[:] = []
        if self._savesyspath is not None:
            sys.path[:] = self._savesyspath
            self._savesyspath = None

        if self._cwd is not None:
            os.chdir(self._cwd)
            self._cwd = None


def _safe_repr(value: object, limit: int = 60) -> str:
    """Best-effort short repr which never raises on broken ``__repr__``."""
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<{type(value).__name__} with broken __repr__: {exc}>"
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


class MonkeyPatchAssemblyError(Exception):
    """Raised when a transactional :meth:`MonkeyPatch.assembly` unit fails.

    Attributes:
        name:
            Optional label passed to :meth:`MonkeyPatch.assembly`.
        step:
            1-based index of the patch step that failed, or ``None`` when
            the ``with`` block raised between patch steps.
        operation:
            Readable description of the failing step, e.g.
            ``"setattr(target=<module 'os'>, name='getcwd')"``.
        original:
            The original exception that aborted the unit (also the chained
            cause).
        rollback_errors:
            List of ``(operation, exception)`` pairs collected while
            restoring already-applied changes. A failing restore does not
            stop the remaining restores.
    """

    def __init__(
        self,
        *,
        name: str | None,
        step: int | None,
        operation: str | None,
        original: BaseException | None,
        rollback_errors: list[tuple[str, BaseException]],
    ) -> None:
        self.name = name
        self.step = step
        self.operation = operation
        self.original = original
        self.rollback_errors = rollback_errors
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        unit = f" {self.name!r}" if self.name is not None else ""
        if self.step is not None:
            message = f"monkeypatch assembly{unit} failed at step {self.step}"
            if self.operation is not None:
                message += f" ({self.operation})"
            if self.original is not None:
                message += f": {type(self.original).__name__}: {self.original}"
        else:
            message = (
                f"monkeypatch assembly{unit} encountered restore failure(s) "
                "while rolling back"
            )
            if self.original is not None:
                message += f" after {type(self.original).__name__}: {self.original}"
        if self.rollback_errors:
            message += (
                f"\nrollback could not fully restore "
                f"{len(self.rollback_errors)} change(s):"
            )
            for operation, error in self.rollback_errors:
                message += f"\n  {operation} -> {type(error).__name__}: {error}"
        return message


# Internal subclass of the ``@final`` MonkeyPatch: both classes live in this
# module and the unit must reuse the exact same single-point patch
# implementation so that behavior inside a unit stays identical.
class MonkeyPatchAssembly(MonkeyPatch):  # type: ignore[misc]
    """Transactional assembly unit created by :meth:`MonkeyPatch.assembly`.

    While the ``with`` block runs this behaves exactly like a
    :class:`MonkeyPatch`, but every change is journaled locally:

    * a clean block exit commits the local journal to the parent patch
      object (or the enclosing unit);
    * an exception triggers a reverse-order rollback that keeps going
      even if an individual restore raises, collecting all restore
      failures for a single report;
    * each instance owns its journal, so nested and concurrent units
      cannot interleave or roll back each other's changes.
    """

    def __init__(self, parent: MonkeyPatch, name: str | None = None) -> None:
        super().__init__()
        self._assembly_parent: MonkeyPatch = parent
        self._assembly_name = name
        self._assembly_step_count = 0
        self._assembly_active_step: tuple[int, str] | None = None
        self._assembly_in_step = False
        self._assembly_closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        if self._assembly_closed:
            return None
        if exc_type is None:
            self._assembly_commit()
            self._assembly_closed = True
            return None

        failed_step = self._assembly_active_step
        rollback_errors = self._assembly_rollback()
        self._assembly_closed = True
        if failed_step is None and not rollback_errors:
            # The failure happened in user code between patch steps and
            # every change was restored cleanly: propagate the original
            # exception untouched.
            return None
        raise MonkeyPatchAssemblyError(
            name=self._assembly_name,
            step=failed_step[0] if failed_step is not None else None,
            operation=failed_step[1] if failed_step is not None else None,
            original=exc,
            rollback_errors=rollback_errors,
        ) from exc

    # -- step bookkeeping --------------------------------------------------

    def _assembly_step(self, description: str, action: Callable[[], None]) -> None:
        __tracebackhide__ = True
        if self._assembly_in_step:
            # Internal self-call from another step, e.g. setenv() applying
            # itself through setitem(): it is part of the outer step and
            # must not be counted or reported as a separate location.
            action()
            return
        self._assembly_step_count += 1
        self._assembly_active_step = (
            self._assembly_step_count,
            description,
        )
        self._assembly_in_step = True
        try:
            action()
        except BaseException:
            # Leave _assembly_active_step set so __exit__ can report this
            # step; _assembly_in_step resets on the way out of the block.
            self._assembly_in_step = False
            raise
        else:
            self._assembly_active_step = None
            self._assembly_in_step = False

    @staticmethod
    def _describe_target(target: object, operation: str, name: object) -> str:
        if isinstance(target, str):
            return f"{operation}(target={_safe_repr(target)})"
        if isinstance(name, str):
            return f"{operation}(target={_safe_repr(target)}, name={name!r})"
        return f"{operation}(target={_safe_repr(target)})"

    # -- patch operations (identical signatures and behavior as the parent)

    def setattr(
        self,
        target: str | object,
        name: object | str,
        value: object = NOTSET,
        raising: bool = True,
    ) -> None:
        __tracebackhide__ = True
        description = self._describe_target(target, "setattr", name)
        # The parent exposes overloaded setattr() signatures which a single
        # delegating signature cannot match; call the implementation directly.
        apply = cast(Callable[..., None], super().setattr)
        self._assembly_step(description, lambda: apply(target, name, value, raising))

    def delattr(
        self,
        target: object | str,
        name: str | NotSetType = NOTSET,
        raising: bool = True,
    ) -> None:
        __tracebackhide__ = True
        description = self._describe_target(target, "delattr", name)
        apply = super().delattr
        self._assembly_step(description, lambda: apply(target, name, raising=raising))

    def setitem(self, dic: Mapping[K, V], name: K, value: V) -> None:
        description = f"setitem(mapping={_safe_repr(dic)}, key={_safe_repr(name)})"
        apply = super().setitem
        self._assembly_step(description, lambda: apply(dic, name, value))

    def delitem(self, dic: Mapping[K, V], name: K, raising: bool = True) -> None:
        description = f"delitem(mapping={_safe_repr(dic)}, key={_safe_repr(name)})"
        apply = super().delitem
        self._assembly_step(description, lambda: apply(dic, name, raising=raising))

    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        apply = super().setenv
        self._assembly_step(
            f"setenv(name={name!r})",
            lambda: apply(name, value, prepend),
        )

    def delenv(self, name: str, raising: bool = True) -> None:
        apply = super().delenv
        self._assembly_step(
            f"delenv(name={name!r})",
            lambda: apply(name, raising=raising),
        )

    def syspath_prepend(self, path: object) -> None:
        apply = super().syspath_prepend
        self._assembly_step(
            f"syspath_prepend(path={_safe_repr(str(path))})",
            lambda: apply(path),
        )

    def chdir(self, path: str | os.PathLike[str]) -> None:
        apply = super().chdir
        self._assembly_step(
            f"chdir(path={_safe_repr(os.fspath(path))})",
            lambda: apply(path),
        )

    # -- commit / rollback -------------------------------------------------

    def _assembly_commit(self) -> None:
        """Merge the local journal into the parent, keeping parent state."""
        parent = self._assembly_parent
        with parent._assembly_lock:
            parent._setattr.extend(self._setattr)
            parent._setitem.extend(self._setitem)
            # The parent keeps its own earlier snapshots: restoring to the
            # state captured before the parent's first change also undoes
            # everything committed by this unit.
            if self._savesyspath is not None and parent._savesyspath is None:
                parent._savesyspath = self._savesyspath
            if self._cwd is not None and parent._cwd is None:
                parent._cwd = self._cwd
        self._setattr[:] = []
        self._setitem[:] = []
        self._savesyspath = None
        self._cwd = None

    def _assembly_rollback(self) -> list[tuple[str, BaseException]]:
        """Restore local changes in reverse order, never aborting early.

        Every restore failure is captured and returned; remaining changes
        are still restored, mirroring the deletion policy of
        :meth:`MonkeyPatch.undo` for already-removed entries.
        """
        errors: list[tuple[str, BaseException]] = []
        for obj, attr, old_value in reversed(self._setattr):
            action = "delattr" if old_value is NOTSET else "setattr"
            operation = f"restore {action} {_safe_repr(obj)}.{attr!r}"
            try:
                if old_value is not NOTSET:
                    setattr(obj, attr, old_value)
                else:
                    delattr(obj, attr)
            except Exception as exc:
                errors.append((operation, exc))
        self._setattr[:] = []
        for dictionary, key, old_value in reversed(self._setitem):
            action = "delitem" if old_value is NOTSET else "setitem"
            operation = (
                f"restore {action} key={_safe_repr(key)} on {_safe_repr(dictionary)}"
            )
            try:
                if old_value is NOTSET:
                    try:
                        # Not all Mapping types support indexing, but
                        # MutableMapping doesn't support TypedDict
                        del dictionary[key]
                    except KeyError:
                        # Was already deleted, so we have the desired state.
                        pass
                else:
                    # Not all Mapping types support indexing, but MutableMapping doesn't support TypedDict
                    dictionary[key] = old_value
            except Exception as exc:
                errors.append((operation, exc))
        self._setitem[:] = []
        if self._savesyspath is not None:
            try:
                sys.path[:] = self._savesyspath
            except Exception as exc:
                errors.append(("restore sys.path", exc))
            self._savesyspath = None
        if self._cwd is not None:
            try:
                os.chdir(self._cwd)
            except Exception as exc:
                errors.append((f"restore chdir to {self._cwd!r}", exc))
            self._cwd = None
        return errors
