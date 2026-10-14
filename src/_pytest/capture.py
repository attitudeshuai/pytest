# mypy: allow-untyped-defs
"""Per-test stdout/stderr capturing mechanism."""

from __future__ import annotations

import abc
import collections
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
import contextlib
import hashlib
import io
from io import UnsupportedOperation
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryFile
from types import TracebackType
from typing import Any
from typing import AnyStr
from typing import BinaryIO
from typing import cast
from typing import Final
from typing import final
from typing import Generic
from typing import Literal
from typing import NamedTuple
from typing import TextIO
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Self

from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import SubRequest
from _pytest.nodes import Collector
from _pytest.nodes import File
from _pytest.nodes import Item
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.stash import StashKey


_CaptureMethod = Literal["fd", "sys", "no", "tee-sys"]

#: Retention policy for on-disk per-item captures.
_DiskRetainPolicy = Literal["failed", "all"]


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption(
        "--capture",
        action="store",
        default="fd",
        metavar="method",
        choices=["fd", "sys", "no", "tee-sys"],
        help="Per-test capturing method: one of fd|sys|no|tee-sys",
    )
    group._addoption(  # private to use reserved lower-case short option
        "-s",
        action="store_const",
        const="no",
        dest="capture",
        help="Shortcut for --capture=no",
    )
    group.addoption(
        "--capture-to-disk",
        action="store",
        metavar="DIR",
        default=None,
        help="Additionally write each test item's captured output (standard "
        "output, standard error and file-descriptor-level output) to "
        "separate files under DIR. By default only the output of failed "
        "items is kept; see --capture-to-disk-retain and "
        "--capture-to-disk-max-size.",
    )
    group.addoption(
        "--capture-to-disk-retain",
        action="store",
        metavar="policy",
        choices=["failed", "all"],
        default=None,
        help="Which items keep their on-disk captured output: 'failed' "
        "(default) keeps only failed items, 'all' keeps every item.",
    )
    group.addoption(
        "--capture-to-disk-max-size",
        action="store",
        metavar="SIZE",
        default=None,
        help="Maximum total size of retained captured output, e.g. '50MB'. "
        "Oldest items are evicted once the limit is exceeded. 0 (the "
        "default) means unlimited.",
    )
    parser.addini(
        "capture_to_disk",
        "Directory in which to persist each test item's captured output to "
        "separate files (standard output, standard error and "
        "file-descriptor-level output). Empty (the default) disables the "
        "feature; --capture-to-disk takes precedence.",
        default="",
    )
    parser.addini(
        "capture_to_disk_retain",
        "Which items keep their on-disk captured output: 'failed' (default) "
        "or 'all'.",
        default="failed",
    )
    parser.addini(
        "capture_to_disk_max_size",
        "Maximum total size of retained captured output, e.g. '50MB'; "
        "oldest items are evicted when exceeded. 0 (default) is unlimited.",
        default="0",
    )


def _colorama_workaround() -> None:
    """Ensure colorama is imported so that it attaches to the correct stdio
    handles on Windows.

    colorama uses the terminal on import time. So if something does the
    first import of colorama while I/O capture is active, colorama will
    fail in various ways.
    """
    if sys.platform.startswith("win32"):
        try:
            import colorama  # noqa: F401
        except ImportError:
            pass


def _readline_workaround() -> None:
    """Ensure readline is imported early so it attaches to the correct stdio handles.

    This isn't a problem with the default GNU readline implementation, but in
    some configurations, Python uses libedit instead (on macOS, and for prebuilt
    binaries such as used by uv).

    In theory this is only needed if readline.backend == "libedit", but the
    workaround consists of importing readline here, so we already worked around
    the issue by the time we could check if we need to.
    """
    try:
        import readline  # noqa: F401
    except ImportError:
        pass


def _windowsconsoleio_workaround(stream: TextIO) -> None:
    """Workaround for Windows Unicode console handling.

    Python 3.6 implemented Unicode console handling for Windows. This works
    by reading/writing to the raw console handle using
    ``{Read,Write}ConsoleW``.

    The problem is that we are going to ``dup2`` over the stdio file
    descriptors when doing ``FDCapture`` and this will ``CloseHandle`` the
    handles used by Python to write to the console. Though there is still some
    weirdness and the console handle seems to only be closed randomly and not
    on the first call to ``CloseHandle``, or maybe it gets reopened with the
    same handle value when we suspend capturing.

    The workaround in this case will reopen stdio with a different fd which
    also means a different handle by replicating the logic in
    "Py_lifecycle.c:initstdio/create_stdio".

    :param stream:
        In practice ``sys.stdout`` or ``sys.stderr``, but given
        here as parameter for unittesting purposes.

    See https://github.com/pytest-dev/py/issues/103.
    """
    if not sys.platform.startswith("win32") or hasattr(sys, "pypy_version_info"):
        return

    # Bail out if ``stream`` doesn't seem like a proper ``io`` stream (#2666).
    if not hasattr(stream, "buffer"):  # type: ignore[unreachable,unused-ignore]
        return

    raw_stdout = stream.buffer.raw if hasattr(stream.buffer, "raw") else stream.buffer

    if not isinstance(raw_stdout, io._WindowsConsoleIO):  # type: ignore[attr-defined,unused-ignore]
        return

    def _reopen_stdio(f, mode):
        if not hasattr(stream.buffer, "raw") and mode[0] == "w":
            buffering = 0
        else:
            buffering = -1

        return io.TextIOWrapper(
            open(os.dup(f.fileno()), mode, buffering),
            f.encoding,
            f.errors,
            f.newlines,
            f.line_buffering,
        )

    sys.stdin = _reopen_stdio(sys.stdin, "rb")
    sys.stdout = _reopen_stdio(sys.stdout, "wb")
    sys.stderr = _reopen_stdio(sys.stderr, "wb")


@hookimpl(wrapper=True)
def pytest_load_initial_conftests(early_config: Config) -> Generator[None]:
    ns = early_config.known_args_namespace
    if ns.capture == "fd":
        _windowsconsoleio_workaround(sys.stdout)
    _colorama_workaround()
    _readline_workaround()
    pluginmanager = early_config.pluginmanager
    capman = CaptureManager(ns.capture)
    pluginmanager.register(capman, "capturemanager")

    # Make sure that capturemanager is properly reset at final shutdown.
    early_config.add_cleanup(capman.stop_global_capturing)

    # Finally trigger conftest loading but while capturing (issue #93).
    capman.start_global_capturing()
    try:
        try:
            yield
        finally:
            capman.suspend_global_capture()
    except BaseException:
        out, err = capman.read_global_capture()
        sys.stdout.write(out)
        sys.stderr.write(err)
        raise


# IO Helpers.


class EncodedFile(io.TextIOWrapper):
    __slots__ = ()

    @property
    def name(self) -> str:
        # Ensure that file.name is a string. Workaround for a Python bug
        # fixed in >=3.7.4: https://bugs.python.org/issue36015
        return repr(self.buffer)

    @property
    def mode(self) -> str:
        # TextIOWrapper doesn't expose a mode, but at least some of our
        # tests check it.
        assert hasattr(self.buffer, "mode")
        return cast(str, self.buffer.mode.replace("b", ""))


class CaptureIO(io.TextIOWrapper):
    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="UTF-8", newline="", write_through=True)

    def getvalue(self) -> str:
        assert isinstance(self.buffer, io.BytesIO)
        return self.buffer.getvalue().decode("UTF-8")


class TeeCaptureIO(CaptureIO):
    def __init__(self, other: TextIO) -> None:
        self._other = other
        super().__init__()

    def write(self, s: str) -> int:
        super().write(s)
        return self._other.write(s)


class DontReadFromInput(TextIO):
    @property
    def encoding(self) -> str:
        assert sys.__stdin__ is not None
        return sys.__stdin__.encoding

    def read(self, size: int = -1) -> str:
        raise OSError(
            "pytest: reading from stdin while output is captured!  Consider using `-s`."
        )

    readline = read

    def __next__(self) -> str:
        return self.readline()

    def readlines(self, hint: int | None = -1) -> list[str]:
        raise OSError(
            "pytest: reading from stdin while output is captured!  Consider using `-s`."
        )

    def __iter__(self) -> Iterator[str]:
        return self

    def fileno(self) -> int:
        raise UnsupportedOperation("redirected stdin is pseudofile, has no fileno()")

    def flush(self) -> None:
        raise UnsupportedOperation("redirected stdin is pseudofile, has no flush()")

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def readable(self) -> bool:
        return False

    def seek(self, offset: int, whence: int = 0) -> int:
        raise UnsupportedOperation("redirected stdin is pseudofile, has no seek(int)")

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise UnsupportedOperation("redirected stdin is pseudofile, has no tell()")

    def truncate(self, size: int | None = None) -> int:
        raise UnsupportedOperation("cannot truncate stdin")

    def write(self, data: str) -> int:
        raise UnsupportedOperation("cannot write to stdin")

    def writelines(self, lines: Iterable[str]) -> None:
        raise UnsupportedOperation("Cannot write to stdin")

    def writable(self) -> bool:
        return False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    @property
    def buffer(self) -> BinaryIO:
        # The str/bytes doesn't actually matter in this type, so OK to fake.
        return self  # type: ignore[return-value]


# Capture classes.


class CaptureBase(abc.ABC, Generic[AnyStr]):
    EMPTY_BUFFER: AnyStr

    @abc.abstractmethod
    def __init__(self, fd: int) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def suspend(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def resume(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def writeorg(self, data: AnyStr) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def snap(self) -> AnyStr:
        raise NotImplementedError()


patchsysdict = {0: "stdin", 1: "stdout", 2: "stderr"}


class NoCapture(CaptureBase[str]):
    EMPTY_BUFFER = ""

    def __init__(self, fd: int) -> None:
        pass

    def start(self) -> None:
        pass

    def done(self) -> None:
        pass

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def snap(self) -> str:
        return ""

    def writeorg(self, data: str) -> None:
        pass


class SysCaptureBase(CaptureBase[AnyStr]):
    def __init__(
        self, fd: int, tmpfile: TextIO | None = None, *, tee: bool = False
    ) -> None:
        name = patchsysdict[fd]
        self._old: TextIO = getattr(sys, name)
        self.name = name
        if tmpfile is None:
            if name == "stdin":
                tmpfile = DontReadFromInput()
            else:
                tmpfile = CaptureIO() if not tee else TeeCaptureIO(self._old)
        self.tmpfile = tmpfile
        self._state = "initialized"

    def repr(self, class_name: str) -> str:
        return "<{} {} _old={} _state={!r} tmpfile={!r}>".format(
            class_name,
            self.name,
            (hasattr(self, "_old") and repr(self._old)) or "<UNSET>",
            self._state,
            self.tmpfile,
        )

    def __repr__(self) -> str:
        return "<{} {} _old={} _state={!r} tmpfile={!r}>".format(
            self.__class__.__name__,
            self.name,
            (hasattr(self, "_old") and repr(self._old)) or "<UNSET>",
            self._state,
            self.tmpfile,
        )

    def _assert_state(self, op: str, states: tuple[str, ...]) -> None:
        assert self._state in states, (
            "cannot {} in state {!r}: expected one of {}".format(
                op, self._state, ", ".join(states)
            )
        )

    def start(self) -> None:
        self._assert_state("start", ("initialized",))
        setattr(sys, self.name, self.tmpfile)
        self._state = "started"

    def done(self) -> None:
        self._assert_state("done", ("initialized", "started", "suspended", "done"))
        if self._state == "done":
            return
        setattr(sys, self.name, self._old)
        del self._old
        self.tmpfile.close()
        self._state = "done"

    def suspend(self) -> None:
        self._assert_state("suspend", ("started", "suspended"))
        setattr(sys, self.name, self._old)
        self._state = "suspended"

    def resume(self) -> None:
        self._assert_state("resume", ("started", "suspended"))
        if self._state == "started":
            return
        setattr(sys, self.name, self.tmpfile)
        self._state = "started"


class SysCaptureBinary(SysCaptureBase[bytes]):
    EMPTY_BUFFER = b""

    def snap(self) -> bytes:
        self._assert_state("snap", ("started", "suspended"))
        self.tmpfile.seek(0)
        res = self.tmpfile.buffer.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: bytes) -> None:
        self._assert_state("writeorg", ("started", "suspended"))
        self._old.flush()
        self._old.buffer.write(data)
        self._old.buffer.flush()


class SysCapture(SysCaptureBase[str]):
    EMPTY_BUFFER = ""

    def snap(self) -> str:
        self._assert_state("snap", ("started", "suspended"))
        assert isinstance(self.tmpfile, CaptureIO)
        res = self.tmpfile.getvalue()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: str) -> None:
        self._assert_state("writeorg", ("started", "suspended"))
        self._old.write(data)
        self._old.flush()


class FDCaptureBase(CaptureBase[AnyStr]):
    def __init__(self, targetfd: int) -> None:
        self.targetfd = targetfd

        try:
            os.fstat(targetfd)
        except OSError:
            # FD capturing is conceptually simple -- create a temporary file,
            # redirect the FD to it, redirect back when done. But when the
            # target FD is invalid it throws a wrench into this lovely scheme.
            #
            # Tests themselves shouldn't care if the FD is valid, FD capturing
            # should work regardless of external circumstances. So falling back
            # to just sys capturing is not a good option.
            #
            # Further complications are the need to support suspend() and the
            # possibility of FD reuse (e.g. the tmpfile getting the very same
            # target FD). The following approach is robust, I believe.
            self.targetfd_invalid: int | None = os.open(os.devnull, os.O_RDWR)
            os.dup2(self.targetfd_invalid, targetfd)
        else:
            self.targetfd_invalid = None
        self.targetfd_save = os.dup(targetfd)

        if targetfd == 0:
            self.tmpfile = open(os.devnull, encoding="utf-8")
            self.syscapture: CaptureBase[str] = SysCapture(targetfd)
        else:
            self.tmpfile = EncodedFile(
                TemporaryFile(buffering=0),
                encoding="utf-8",
                errors="replace",
                newline="",
                write_through=True,
            )
            if targetfd in patchsysdict:
                self.syscapture = SysCapture(targetfd, self.tmpfile)
            else:
                self.syscapture = NoCapture(targetfd)

        self._state = "initialized"

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.targetfd} oldfd={self.targetfd_save} "
            f"_state={self._state!r} tmpfile={self.tmpfile!r}>"
        )

    def _assert_state(self, op: str, states: tuple[str, ...]) -> None:
        assert self._state in states, (
            "cannot {} in state {!r}: expected one of {}".format(
                op, self._state, ", ".join(states)
            )
        )

    def start(self) -> None:
        """Start capturing on targetfd using memorized tmpfile."""
        self._assert_state("start", ("initialized",))
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self.syscapture.start()
        self._state = "started"

    def done(self) -> None:
        """Stop capturing, restore streams, return original capture file,
        seeked to position zero."""
        self._assert_state("done", ("initialized", "started", "suspended", "done"))
        if self._state == "done":
            return
        os.dup2(self.targetfd_save, self.targetfd)
        os.close(self.targetfd_save)
        if self.targetfd_invalid is not None:
            if self.targetfd_invalid != self.targetfd:
                os.close(self.targetfd)
            os.close(self.targetfd_invalid)
        self.syscapture.done()
        self.tmpfile.close()
        self._state = "done"

    def suspend(self) -> None:
        self._assert_state("suspend", ("started", "suspended"))
        if self._state == "suspended":
            return
        self.syscapture.suspend()
        os.dup2(self.targetfd_save, self.targetfd)
        self._state = "suspended"

    def resume(self) -> None:
        self._assert_state("resume", ("started", "suspended"))
        if self._state == "started":
            return
        self.syscapture.resume()
        os.dup2(self.tmpfile.fileno(), self.targetfd)
        self._state = "started"


class FDCaptureBinary(FDCaptureBase[bytes]):
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces `bytes`.
    """

    EMPTY_BUFFER = b""

    def snap(self) -> bytes:
        self._assert_state("snap", ("started", "suspended"))
        # Avoid seek/read/truncate in the common case of no output.
        if os.fstat(self.tmpfile.fileno()).st_size == 0:
            return self.EMPTY_BUFFER
        self.tmpfile.seek(0)
        res = self.tmpfile.buffer.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res  # type: ignore[return-value]

    def writeorg(self, data: bytes) -> None:
        """Write to original file descriptor."""
        self._assert_state("writeorg", ("started", "suspended"))
        os.write(self.targetfd_save, data)


class FDCapture(FDCaptureBase[str]):
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces text.
    """

    EMPTY_BUFFER = ""

    def snap(self) -> str:
        self._assert_state("snap", ("started", "suspended"))
        # Avoid seek/read/truncate in the common case of no output.
        if os.fstat(self.tmpfile.fileno()).st_size == 0:
            return self.EMPTY_BUFFER
        self.tmpfile.seek(0)
        res = self.tmpfile.read()
        self.tmpfile.seek(0)
        self.tmpfile.truncate()
        return res

    def writeorg(self, data: str) -> None:
        """Write to original file descriptor."""
        self._assert_state("writeorg", ("started", "suspended"))
        # XXX use encoding of original stream
        os.write(self.targetfd_save, data.encode("utf-8"))


# MultiCapture


# Generic NamedTuple only supported since Python 3.11.
if sys.version_info >= (3, 11) or TYPE_CHECKING:

    @final
    class CaptureResult(NamedTuple, Generic[AnyStr]):
        """The result of :method:`caplog.readouterr() <pytest.CaptureFixture.readouterr>`."""

        out: AnyStr
        err: AnyStr

else:

    class CaptureResult(
        collections.namedtuple("CaptureResult", ["out", "err"]),  # noqa: PYI024
        Generic[AnyStr],
    ):
        """The result of :method:`caplog.readouterr() <pytest.CaptureFixture.readouterr>`."""

        __slots__ = ()


class MultiCapture(Generic[AnyStr]):
    _state = None
    _in_suspended = False

    def __init__(
        self,
        in_: CaptureBase[AnyStr] | None,
        out: CaptureBase[AnyStr] | None,
        err: CaptureBase[AnyStr] | None,
    ) -> None:
        self.in_: CaptureBase[AnyStr] | None = in_
        self.out: CaptureBase[AnyStr] | None = out
        self.err: CaptureBase[AnyStr] | None = err

    def __repr__(self) -> str:
        return (
            f"<MultiCapture out={self.out!r} err={self.err!r} in_={self.in_!r} "
            f"_state={self._state!r} _in_suspended={self._in_suspended!r}>"
        )

    def start_capturing(self) -> None:
        self._state = "started"
        if self.in_:
            self.in_.start()
        if self.out:
            self.out.start()
        if self.err:
            self.err.start()

    def pop_outerr_to_orig(self) -> tuple[AnyStr, AnyStr]:
        """Pop current snapshot out/err capture and flush to orig streams."""
        out, err = self.readouterr()
        if out:
            assert self.out is not None
            self.out.writeorg(out)
        if err:
            assert self.err is not None
            self.err.writeorg(err)
        return out, err

    def suspend_capturing(self, in_: bool = False) -> None:
        self._state = "suspended"
        if self.out:
            self.out.suspend()
        if self.err:
            self.err.suspend()
        if in_ and self.in_:
            self.in_.suspend()
            self._in_suspended = True

    def resume_capturing(self) -> None:
        self._state = "started"
        if self.out:
            self.out.resume()
        if self.err:
            self.err.resume()
        if self._in_suspended:
            assert self.in_ is not None
            self.in_.resume()
            self._in_suspended = False

    def stop_capturing(self) -> None:
        """Stop capturing and reset capturing streams."""
        if self._state == "stopped":
            raise ValueError("was already stopped")
        self._state = "stopped"
        if self.out:
            self.out.done()
        if self.err:
            self.err.done()
        if self.in_:
            self.in_.done()

    def is_started(self) -> bool:
        """Whether actively capturing -- not suspended or stopped."""
        return self._state == "started"

    def readouterr(self) -> CaptureResult[AnyStr]:
        out = self.out.snap() if self.out else ""
        err = self.err.snap() if self.err else ""
        # TODO: This type error is real, need to fix.
        return CaptureResult(out, err)  # type: ignore[arg-type]


def _get_multicapture(method: _CaptureMethod) -> MultiCapture[str]:
    if method == "fd":
        return MultiCapture(in_=FDCapture(0), out=FDCapture(1), err=FDCapture(2))
    elif method == "sys":
        return MultiCapture(in_=SysCapture(0), out=SysCapture(1), err=SysCapture(2))
    elif method == "no":
        return MultiCapture(in_=None, out=None, err=None)
    elif method == "tee-sys":
        return MultiCapture(
            in_=None, out=SysCapture(1, tee=True), err=SysCapture(2, tee=True)
        )
    raise ValueError(f"unknown capturing method: {method!r}")


# CaptureManager and CaptureFixture


class CaptureManager:
    """The capture plugin.

    Manages that the appropriate capture method is enabled/disabled during
    collection and each test phase (setup, call, teardown). After each of
    those points, the captured output is obtained and attached to the
    collection/runtest report.

    There are two levels of capture:

    * global: enabled by default and can be suppressed by the ``-s``
      option. This is always enabled/disabled during collection and each test
      phase.

    * fixture: when a test function or one of its fixture depend on the
      ``capsys`` or ``capfd`` fixtures. In this case special handling is
      needed to ensure the fixtures take precedence over the global capture.
    """

    def __init__(self, method: _CaptureMethod) -> None:
        self._method: Final = method
        self._global_capturing: MultiCapture[str] | None = None
        self._capture_fixture: CaptureFixture[Any] | None = None
        #: Optional per-item on-disk capture, installed by
        #: :class:`DiskCaptureManager` only when the feature is enabled.
        #: When ``None`` (the default) capture behaves exactly as before.
        self.disk: DiskCaptureManager | None = None

    def __repr__(self) -> str:
        return (
            f"<CaptureManager _method={self._method!r} _global_capturing={self._global_capturing!r} "
            f"_capture_fixture={self._capture_fixture!r}>"
        )

    def is_capturing(self) -> str | bool:
        if self.is_globally_capturing():
            return "global"
        if self._capture_fixture:
            return f"fixture {self._capture_fixture.request.fixturename}"
        return False

    # Global capturing control

    def is_globally_capturing(self) -> bool:
        return self._method != "no"

    def start_global_capturing(self) -> None:
        assert self._global_capturing is None
        self._global_capturing = _get_multicapture(self._method)
        self._global_capturing.start_capturing()

    def stop_global_capturing(self) -> None:
        if self._global_capturing is not None:
            self._global_capturing.pop_outerr_to_orig()
            self._global_capturing.stop_capturing()
            self._global_capturing = None

    def resume_global_capture(self) -> None:
        # During teardown of the python process, and on rare occasions, capture
        # attributes can be `None` while trying to resume global capture.
        if self._global_capturing is not None:
            self._global_capturing.resume_capturing()

    def suspend_global_capture(self, in_: bool = False) -> None:
        if self._global_capturing is not None:
            self._global_capturing.suspend_capturing(in_=in_)

    def suspend(self, in_: bool = False) -> None:
        # Need to undo local capsys-et-al if it exists before disabling global capture.
        self.suspend_fixture()
        self.suspend_global_capture(in_)

    def resume(self) -> None:
        self.resume_global_capture()
        self.resume_fixture()

    def read_global_capture(self) -> CaptureResult[str]:
        assert self._global_capturing is not None
        return self._global_capturing.readouterr()

    # Fixture Control

    def set_fixture(self, capture_fixture: CaptureFixture[Any]) -> None:
        if self._capture_fixture:
            current_fixture = self._capture_fixture.request.fixturename
            requested_fixture = capture_fixture.request.fixturename
            capture_fixture.request.raiseerror(
                f"cannot use {requested_fixture} and {current_fixture} at the same time"
            )
        self._capture_fixture = capture_fixture

    def unset_fixture(self) -> None:
        self._capture_fixture = None

    def activate_fixture(self) -> None:
        """If the current item is using ``capsys`` or ``capfd``, activate
        them so they take precedence over the global capture."""
        if self._capture_fixture:
            self._capture_fixture._start()

    def deactivate_fixture(self) -> None:
        """Deactivate the ``capsys`` or ``capfd`` fixture of this item, if any."""
        if self._capture_fixture:
            self._capture_fixture.close()

    def suspend_fixture(self) -> None:
        if self._capture_fixture:
            self._capture_fixture._suspend()

    def resume_fixture(self) -> None:
        if self._capture_fixture:
            self._capture_fixture._resume()

    # Helper context managers

    @contextlib.contextmanager
    def global_and_fixture_disabled(self) -> Generator[None]:
        """Context manager to temporarily disable global and current fixture capturing."""
        do_fixture = self._capture_fixture and self._capture_fixture._is_started()
        if do_fixture:
            self.suspend_fixture()
        do_global = self._global_capturing and self._global_capturing.is_started()
        if do_global:
            self.suspend_global_capture()
        try:
            yield
        finally:
            if do_global:
                self.resume_global_capture()
            if do_fixture:
                self.resume_fixture()

    @contextlib.contextmanager
    def item_capture(self, when: str, item: Item) -> Generator[None]:
        self.resume_global_capture()
        self.activate_fixture()
        try:
            yield
        finally:
            self.deactivate_fixture()
            self.suspend_global_capture(in_=False)

            # Grab the undecoded, file-descriptor-level bytes before the
            # normal snap() drains the capture buffers.  Best-effort: the
            # in-memory/report capture must never be affected by disk issues.
            disk = self.disk
            raw: tuple[bytes | None, bytes | None] | None = None
            if disk is not None and self._global_capturing is not None:
                try:
                    raw = disk.peek_raw(self._global_capturing)
                except OSError:
                    raw = None

            out, err = self.read_global_capture()
            item.add_report_section(when, "stdout", out)
            item.add_report_section(when, "stderr", err)

            if disk is not None:
                try:
                    disk.record_phase(item, when, out, err, raw)
                except OSError:
                    disk.errors += 1

    # Hooks

    @hookimpl(wrapper=True)
    def pytest_make_collect_report(
        self, collector: Collector
    ) -> Generator[None, CollectReport, CollectReport]:
        if isinstance(collector, File):
            self.resume_global_capture()
            try:
                rep = yield
            finally:
                self.suspend_global_capture()
            out, err = self.read_global_capture()
            if out:
                rep.sections.append(("Captured stdout", out))
            if err:
                rep.sections.append(("Captured stderr", err))
        else:
            rep = yield
        return rep

    @hookimpl(wrapper=True)
    def pytest_runtest_setup(self, item: Item) -> Generator[None]:
        with self.item_capture("setup", item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None]:
        with self.item_capture("call", item):
            return (yield)

    @hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item: Item) -> Generator[None]:
        with self.item_capture("teardown", item):
            return (yield)

    @hookimpl(tryfirst=True)
    def pytest_keyboard_interrupt(self) -> None:
        self.stop_global_capturing()

    @hookimpl(tryfirst=True)
    def pytest_internalerror(self) -> None:
        self.stop_global_capturing()


# Per-test capture persistence ("capture to disk").
#
# When enabled, every item gets its own directory in which the output of
# each phase (setup/call/teardown) is written to independent files:
#
#     <when>.stdout       text, identical to the "Captured stdout <when>"
#                         report section
#     <when>.stderr       text, identical to the "Captured stderr <when>"
#                         report section
#     <when>.stdout.bin   raw file-descriptor-level bytes (undecoded)
#     <when>.stderr.bin   raw file-descriptor-level bytes (undecoded)
#
# The directory tree mirrors the node id, e.g.::
#
#     <root>/test_dir/test_file.py/TestClass/test_item[param]/call.stdout
#
# Files are only created for streams that produced output.  By default the
# output of passing items is removed when the item finishes, so after the
# session only failed items occupy disk.  All filesystem errors are
# swallowed: persistence problems never change the outcome of a test.

disk_key = StashKey["DiskCaptureManager"]()

#: Maximum length of a single path component derived from a node id.
_DISK_MAX_COMPONENT_LEN = 80

#: Characters that are not safe (or not legal on Windows) in a path
#: component.  They are percent-encoded, which keeps the mapping injective.
_DISK_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*%\x00-\x1f\x7f]')

_DISK_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:b)?\s*$", re.IGNORECASE)
_SIZE_UNITS = {
    "": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}

#: Report section key -> human readable label, in listing order.
_DISK_FILE_LABELS: tuple[tuple[str, str], ...] = (
    ("stdout", "standard output"),
    ("stderr", "standard error"),
    ("stdout_fd", "file-descriptor-level standard output (raw bytes)"),
    ("stderr_fd", "file-descriptor-level standard error (raw bytes)"),
)


def _parse_disk_size(value: str) -> int:
    """Parse a size like ``"50MB"``, ``"1.5g"`` or ``"1024"`` into bytes.

    ``"0"`` means unlimited.

    :raises ValueError: if *value* is not a valid non-negative size.
    """
    match = _SIZE_RE.match(value)
    if not match:
        raise ValueError(value)
    number = float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()]
    return int(number)


def _sanitize_disk_component(name: str) -> str:
    """Make *name* safe to use as a single filesystem path component.

    Percent-encoding guarantees injectivity (two distinct node ids can
    never collide on the same path); overly long components get a hash
    suffix instead of being blindly truncated.
    """

    def _encode(match: re.Match[str]) -> str:
        return f"%{ord(match.group()):02X}"

    encoded = _DISK_UNSAFE_CHARS_RE.sub(_encode, name)
    # Windows refuses trailing spaces/dots in path components.
    if encoded[-1:] in (" ", "."):
        encoded = encoded[:-1] + f"%{ord(encoded[-1]):02X}"
    # Windows reserved device names (e.g. "con", "lpt1.py").
    if encoded.split(".", 1)[0].upper() in _DISK_WINDOWS_RESERVED:
        encoded += "_"
    if len(encoded) > _DISK_MAX_COMPONENT_LEN:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        encoded = encoded[: _DISK_MAX_COMPONENT_LEN - 10] + "__" + digest
    return encoded


def _peek_raw_capture(cap: CaptureBase[Any] | None) -> bytes | None:
    """Read raw bytes from a capture sink without draining it.

    Runs immediately before :meth:`CaptureBase.snap`; leaves position and
    contents untouched so the following ``snap()`` behaves exactly as
    usual.  Returns ``None`` for channels that are not captured.
    """
    if cap is None:
        return None
    tmpfile = getattr(cap, "tmpfile", None)
    if tmpfile is None:
        return None
    if isinstance(cap, FDCaptureBase):
        # Same drain as FDCaptureBinary.snap(), minus the truncation (the
        # regular snap() right after performs it).
        if os.fstat(tmpfile.fileno()).st_size == 0:
            return b""
        tmpfile.seek(0)
        data = tmpfile.buffer.read()
        tmpfile.seek(0)
        return data  # type: ignore[no-any-return]
    if isinstance(cap, SysCaptureBase):
        buffer = getattr(tmpfile, "buffer", None)
        if buffer is None or not hasattr(buffer, "getvalue"):
            # E.g. DontReadFromInput for stdin (never out/err).
            return None
        return bytes(buffer.getvalue())
    return None


class _ItemDiskCapture:
    """On-disk capture state for a single test item across its phases."""

    def __init__(self, path: Path) -> None:
        #: Directory holding this item's capture files.
        self.path = path
        #: Whether any of the item's phases reported a failure/error.
        self.failed = False
        #: ``{phase: {stream key: absolute path string}}``.
        self.phase_files: dict[str, dict[str, str]] = {}

    def has_files(self) -> bool:
        return any(self.phase_files.values())


class DiskCaptureManager:
    """Optional plugin persisting per-item captured output to disk.

    Registered only when ``--capture-to-disk`` / the ``capture_to_disk``
    ini option is set, so a non-opted-in session is byte-for-byte
    unaffected.  Cooperates with :class:`CaptureManager`, which calls into
    :meth:`peek_raw`/:meth:`record_phase` from its per-phase capture
    context.
    """

    def __init__(
        self,
        config: Config,
        root: Path,
        retain: _DiskRetainPolicy,
        max_size: int,
        token: str | None,
    ) -> None:
        self.config = config
        self.root = root
        self.retain: _DiskRetainPolicy = retain
        self.max_size = max_size
        #: Per-process sub-directory name used under xdist/distributed
        #: execution, so concurrent processes never write the same files.
        self.token = _sanitize_disk_component(token) if token else None
        self._items: dict[str, _ItemDiskCapture] = {}
        #: Directories of retained items, in retention (oldest first)
        #: order, for the total-size eviction policy.
        self._retained: list[Path] = []
        self._total_size = 0
        #: Number of items pruned because they passed (failed-policy).
        self.removed_items = 0
        #: Number of items evicted to satisfy the total-size limit.
        self.evicted_items = 0
        #: Number of swallowed filesystem errors.
        self.errors = 0

    # -- Path layout -----------------------------------------------------

    def item_path(self, item: Item) -> Path:
        """Return the (token-aware) capture directory for *item*."""
        node_id = item.id
        parts = [part for part in node_id.path.split("/") if part not in ("", ".")]
        names = list(node_id.names)
        if not names:
            # Defensive fallback for non-python items without name segments.
            names = [parts[-1] if parts else str(node_id)]
        if node_id.params is not None:
            names[-1] = f"{names[-1]}[{node_id.params}]"
        parts.extend(names)
        path = self.root.joinpath(
            *(_sanitize_disk_component(part) for part in parts)
        )
        if self.token is not None:
            path = path / self.token
        return path

    def _get_item(self, item: Item) -> _ItemDiskCapture:
        disk_item = self._items.get(item.nodeid)
        if disk_item is None:
            disk_item = _ItemDiskCapture(self.item_path(item))
            self._items[item.nodeid] = disk_item
        return disk_item

    # -- CaptureManager integration --------------------------------------

    def peek_raw(
        self, multicapture: MultiCapture[str]
    ) -> tuple[bytes | None, bytes | None]:
        """Return raw ``(stdout, stderr)`` bytes before buffers drain."""
        return (
            _peek_raw_capture(multicapture.out),
            _peek_raw_capture(multicapture.err),
        )

    def record_phase(
        self,
        item: Item,
        when: str,
        out: str,
        err: str,
        raw: tuple[bytes | None, bytes | None] | None,
    ) -> None:
        """Write one phase's captured streams to independent files."""
        disk_item = self._get_item(item)
        files = disk_item.phase_files.setdefault(when, {})
        if out:
            path = disk_item.path / f"{when}.stdout"
            if self._write(path, out.encode("utf-8")):
                files["stdout"] = str(path)
        if err:
            path = disk_item.path / f"{when}.stderr"
            if self._write(path, err.encode("utf-8")):
                files["stderr"] = str(path)
        if raw is not None:
            raw_out, raw_err = raw
            if raw_out:
                path = disk_item.path / f"{when}.stdout.bin"
                if self._write(path, raw_out):
                    files["stdout_fd"] = str(path)
            if raw_err:
                path = disk_item.path / f"{when}.stderr.bin"
                if self._write(path, raw_err):
                    files["stderr_fd"] = str(path)
        if files:
            item.add_report_section(
                when, "output files", self._format_files_section(files)
            )

    def _write(self, path: Path, data: bytes) -> bool:
        """Write *data* atomically-ish; never raise, report success."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fp:
                fp.write(data)
            return True
        except OSError:
            self.errors += 1
            return False

    @staticmethod
    def _format_files_section(files: dict[str, str]) -> str:
        return "\n".join(
            f"{label}: {files[key]}"
            for key, label in _DISK_FILE_LABELS
            if key in files
        )

    # -- Hooks -----------------------------------------------------------

    @hookimpl(wrapper=True)
    def pytest_runtest_makereport(
        self, item: Item, call: Any
    ) -> Generator[None, TestReport, TestReport]:
        # Attach the capture-file metadata at makereport time, which is
        # before logreport hooks and any report serialization (xdist) can
        # observe the report.
        report: TestReport = yield
        disk_item = self._items.get(item.nodeid)
        if disk_item is not None:
            if report.failed:
                disk_item.failed = True
            report.captured_output_files = dict(
                disk_item.phase_files.get(report.when, ())
            )
            if report.when == "teardown":
                report.captured_output_dir = self._finalize(item.nodeid)
        return report

    @hookimpl
    def pytest_sessionfinish(self, session: Any) -> None:
        # Safety net for items whose teardown report was never logged
        # (e.g. interrupted sessions): apply retention so passing-item
        # output does not linger when the policy says it should not.
        for nodeid in list(self._items):
            self._finalize(nodeid)

    @hookimpl
    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: int, config: Config
    ) -> None:
        lines: list[str] = []
        if self._retained:
            lines.append(
                f"{len(self._retained)} test item(s) with captured output "
                f"kept in: {self.root}"
            )
        if self.evicted_items:
            lines.append(
                f"{self.evicted_items} item(s) evicted to stay within the "
                f"retained-output size limit"
            )
        if self.errors:
            lines.append(
                f"warning: {self.errors} error(s) while writing captured "
                f"output to disk were ignored; test outcomes were not "
                f"affected"
            )
        if lines:
            terminalreporter.write_sep("=", "captured output files")
            for line in lines:
                terminalreporter.write_line(line)

    # -- Retention -------------------------------------------------------

    def _finalize(self, nodeid: str) -> str | None:
        disk_item = self._items.pop(nodeid, None)
        if disk_item is None:
            return None
        try:
            if self.retain == "all" or disk_item.failed:
                if disk_item.has_files():
                    self._retained.append(disk_item.path)
                    self._total_size += self._directory_size(disk_item.path)
                    # The item that just finished is protected against this
                    # round of eviction: its output must survive at least
                    # until later items can rotate it out.
                    self._enforce_limit(disk_item.path)
                    return str(disk_item.path)
                return None
            # Default policy: passing items leave nothing on disk.
            self._remove_tree(disk_item.path)
            self.removed_items += 1
        except OSError:
            self.errors += 1
        return None

    def _enforce_limit(self, protected: Path | None = None) -> None:
        if self.max_size <= 0:
            return
        while self._total_size > self.max_size:
            # Evict the oldest non-protected retained item.
            index = next(
                (i for i, path in enumerate(self._retained) if path != protected),
                None,
            )
            if index is None:
                # Only the just-finished item remains: keep it even if it
                # is larger than the cap on its own.
                break
            oldest = self._retained.pop(index)
            self._total_size -= self._directory_size(oldest)
            self._remove_tree(oldest)
            self.evicted_items += 1
        if self._total_size < 0:
            self._total_size = 0

    def _remove_tree(self, path: Path) -> None:
        """Remove *path* and any directories emptied by the removal, up to
        (but not including) the capture root."""
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        root = self.root
        while parent != root:
            try:
                parent.relative_to(root)
            except ValueError:
                break
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    @staticmethod
    def _directory_size(path: Path) -> int:
        total = 0
        try:
            for dirpath, _dirnames, filenames in os.walk(path):
                for filename in filenames:
                    try:
                        total += (Path(dirpath) / filename).stat().st_size
                    except OSError:
                        pass
        except OSError:
            return 0
        return total


def pytest_configure(config: Config) -> None:
    capman: CaptureManager | None = config.pluginmanager.getplugin(
        "capturemanager"
    )
    if capman is None:
        return

    raw_path = config.option.capture_to_disk or config.getini("capture_to_disk")
    if not raw_path:
        return

    root = Path(str(raw_path))
    if not root.is_absolute():
        root = config.rootpath / root

    retain = config.option.capture_to_disk_retain
    if retain is None:
        retain = config.getini("capture_to_disk_retain")
    if retain not in ("failed", "all"):
        raise UsageError(
            f"invalid capture-to-disk retain policy {retain!r}: "
            "expected 'failed' or 'all'"
        )

    raw_max_size = config.option.capture_to_disk_max_size
    if raw_max_size is None:
        raw_max_size = config.getini("capture_to_disk_max_size")
    try:
        max_size = _parse_disk_size(str(raw_max_size))
    except ValueError:
        raise UsageError(
            f"invalid capture-to-disk max size {raw_max_size!r}: expected a "
            "size like '50MB', '1.5G' or an integer number of bytes"
        ) from None

    # Distinct processes (xdist workers, distributed runs) get their own
    # sub-directory, so file names can never clash.  An explicit override
    # is available for setups xdist does not identify automatically.
    token = (
        os.environ.get("PYTEST_CAPTURE_TO_DISK_WORKER")
        or os.environ.get("PYTEST_XDIST_WORKER")
    )

    disk = DiskCaptureManager(
        config, root, cast(_DiskRetainPolicy, retain), max_size, token
    )
    # Fail fast if the user-provided location cannot be created: this is a
    # configuration error, not a per-test capture write error.
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UsageError(
            f"cannot create capture-to-disk directory {root}: {exc}"
        ) from exc
    config.stash[disk_key] = disk
    config.pluginmanager.register(disk, "diskcapturemanager")
    capman.disk = disk


def pytest_unconfigure(config: Config) -> None:
    disk = config.stash.get(disk_key, None)
    if disk is None:
        return
    capman: CaptureManager | None = config.pluginmanager.getplugin(
        "capturemanager"
    )
    if capman is not None:
        capman.disk = None
    config.pluginmanager.unregister(disk)
    del config.stash[disk_key]


class CaptureFixture(Generic[AnyStr]):
    """Object returned by the :fixture:`capsys`, :fixture:`capsysbinary`,
    :fixture:`capfd` and :fixture:`capfdbinary` fixtures."""

    def __init__(
        self,
        captureclass: type[CaptureBase[AnyStr]],
        request: SubRequest,
        *,
        config: dict[str, Any] | None = None,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        self.captureclass: type[CaptureBase[AnyStr]] = captureclass
        self.request = request
        self._config = config if config else {}
        self._capture: MultiCapture[AnyStr] | None = None
        self._captured_out: AnyStr = self.captureclass.EMPTY_BUFFER
        self._captured_err: AnyStr = self.captureclass.EMPTY_BUFFER

    def _start(self) -> None:
        if self._capture is None:
            self._capture = MultiCapture(
                in_=None,
                out=self.captureclass(1, **self._config),
                err=self.captureclass(2, **self._config),
            )
            self._capture.start_capturing()

    def close(self) -> None:
        if self._capture is not None:
            if self._config.get("tee"):
                # When tee is enabled, output was already written to the
                # original stream in real-time by TeeCaptureIO.  Using
                # pop_outerr_to_orig() would write it a second time via
                # writeorg(), causing doubled output (see #13784).
                out, err = self._capture.readouterr()
            else:
                out, err = self._capture.pop_outerr_to_orig()
            self._captured_out += out
            self._captured_err += err
            self._capture.stop_capturing()
            self._capture = None

    def readouterr(self) -> CaptureResult[AnyStr]:
        """Read and return the captured output so far, resetting the internal
        buffer.

        :returns:
            The captured content as a namedtuple with ``out`` and ``err``
            string attributes.
        """
        captured_out, captured_err = self._captured_out, self._captured_err
        if self._capture is not None:
            out, err = self._capture.readouterr()
            captured_out += out
            captured_err += err
        self._captured_out = self.captureclass.EMPTY_BUFFER
        self._captured_err = self.captureclass.EMPTY_BUFFER
        return CaptureResult(captured_out, captured_err)

    def _suspend(self) -> None:
        """Suspend this fixture's own capturing temporarily."""
        if self._capture is not None:
            self._capture.suspend_capturing()

    def _resume(self) -> None:
        """Resume this fixture's own capturing temporarily."""
        if self._capture is not None:
            self._capture.resume_capturing()

    def _is_started(self) -> bool:
        """Whether actively capturing -- not disabled or closed."""
        if self._capture is not None:
            return self._capture.is_started()
        return False

    @contextlib.contextmanager
    def disabled(self) -> Generator[None]:
        """Temporarily disable capturing while inside the ``with`` block."""
        capmanager: CaptureManager = self.request.config.pluginmanager.getplugin(
            "capturemanager"
        )
        with capmanager.global_and_fixture_disabled():
            yield


# The fixtures.


@fixture
def capsys(request: SubRequest) -> Generator[CaptureFixture[str]]:
    r"""Enable text capturing of writes to ``sys.stdout`` and ``sys.stderr``.

    The captured output is made available via ``capsys.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``text`` objects.

    Returns an instance of :class:`CaptureFixture[str] <pytest.CaptureFixture>`.

    Example:

    .. code-block:: python

        def test_output(capsys):
            print("hello")
            captured = capsys.readouterr()
            assert captured.out == "hello\n"
    """
    capman: CaptureManager = request.config.pluginmanager.getplugin("capturemanager")
    capture_fixture = CaptureFixture(SysCapture, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()


@fixture
def capteesys(request: SubRequest) -> Generator[CaptureFixture[str]]:
    r"""Enable simultaneous text capturing and pass-through of writes
    to ``sys.stdout`` and ``sys.stderr`` as defined by ``--capture=``.


    The captured output is made available via ``capteesys.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``text`` objects.

    The output is also passed-through, allowing it to be "live-printed",
    reported, or both as defined by ``--capture=``.

    Returns an instance of :class:`CaptureFixture[str] <pytest.CaptureFixture>`.

    Example:

    .. code-block:: python

        def test_output(capteesys):
            print("hello")
            captured = capteesys.readouterr()
            assert captured.out == "hello\n"
    """
    capman: CaptureManager = request.config.pluginmanager.getplugin("capturemanager")
    capture_fixture = CaptureFixture(
        SysCapture, request, config=dict(tee=True), _ispytest=True
    )
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()


@fixture
def capsysbinary(request: SubRequest) -> Generator[CaptureFixture[bytes]]:
    r"""Enable bytes capturing of writes to ``sys.stdout`` and ``sys.stderr``.

    The captured output is made available via ``capsysbinary.readouterr()``
    method calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``bytes`` objects.

    Returns an instance of :class:`CaptureFixture[bytes] <pytest.CaptureFixture>`.

    Example:

    .. code-block:: python

        def test_output(capsysbinary):
            print("hello")
            captured = capsysbinary.readouterr()
            assert captured.out == b"hello\n"
    """
    capman: CaptureManager = request.config.pluginmanager.getplugin("capturemanager")
    capture_fixture = CaptureFixture(SysCaptureBinary, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()


@fixture
def capfd(request: SubRequest) -> Generator[CaptureFixture[str]]:
    r"""Enable text capturing of writes to file descriptors ``1`` and ``2``.

    The captured output is made available via ``capfd.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``text`` objects.

    Returns an instance of :class:`CaptureFixture[str] <pytest.CaptureFixture>`.

    Example:

    .. code-block:: python

        def test_system_echo(capfd):
            os.system('echo "hello"')
            captured = capfd.readouterr()
            assert captured.out == "hello\n"
    """
    capman: CaptureManager = request.config.pluginmanager.getplugin("capturemanager")
    capture_fixture = CaptureFixture(FDCapture, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()


@fixture
def capfdbinary(request: SubRequest) -> Generator[CaptureFixture[bytes]]:
    r"""Enable bytes capturing of writes to file descriptors ``1`` and ``2``.

    The captured output is made available via ``capfd.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``byte`` objects.

    Returns an instance of :class:`CaptureFixture[bytes] <pytest.CaptureFixture>`.

    Example:

    .. code-block:: python

        def test_system_echo(capfdbinary):
            os.system('echo "hello"')
            captured = capfdbinary.readouterr()
            assert captured.out == b"hello\n"

    """
    capman: CaptureManager = request.config.pluginmanager.getplugin("capturemanager")
    capture_fixture = CaptureFixture(FDCaptureBinary, request, _ispytest=True)
    capman.set_fixture(capture_fixture)
    capture_fixture._start()
    yield capture_fixture
    capture_fixture.close()
    capman.unset_fixture()
