"""Running agent-written code: static lint, a stripped environment, resource limits and IPC.

Agent code (strategy.py, self_improve.py and any helper modules) is written by an LLM, so it
is never imported into the harness process. It runs in a child process that:
  * receives market data one bar at a time over a pipe, so future bars do not exist in its
    memory (lookahead is impossible by construction, not merely forbidden);
  * gets a minimal environment with no API keys or broker credentials;
  * has CPU, memory, file-size and open-file limits;
  * runs in a fresh network namespace when `unshare` is available (no network at all);
  * has `open`, `eval`, `exec` and friends removed from builtins, on top of the AST lint.

Python cannot be fully sandboxed in-process. For code you do not trust at all, also run the
whole harness inside a container (e.g. `docker run --network none`).
"""
from __future__ import annotations

import ast
import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent
NOBODY = 65534

ALLOWED_IMPORTS = {
    "__future__", "abc", "bisect", "collections", "copy", "dataclasses", "datetime", "difflib",
    "enum", "functools", "heapq", "itertools", "json", "math", "numpy", "operator", "random",
    "re", "statistics", "string", "textwrap", "typing", "typing_extensions",
}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview",
    "__builtins__", "__loader__", "__spec__",
}
BANNED_ATTRS = {
    # numpy / json file and native-code access
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "fromregex", "memmap", "ctypeslib", "f2py", "DataSource", "ctypes", "lib", "dump",
}
ALLOWED_DUNDER_ATTRS = {"__init__", "__name__", "__post_init__", "__class_getitem__"}

ALLOWED_AGENT_FILE_SUFFIXES = (".py", ".md", ".txt", ".json")
MAX_AGENT_FILE_BYTES = 200_000
MAX_AGENT_FILES = 24


def lint_source(src: str, local_modules: set[str] = frozenset()) -> list[str]:
    """Return a list of violations (empty when the source is acceptable)."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    errs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS and root not in local_modules:
                    errs.append(f"line {node.lineno}: import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level and node.level > 0:
                errs.append(f"line {node.lineno}: relative imports are not allowed; use 'import <module>'")
            elif root not in ALLOWED_IMPORTS and root not in local_modules:
                errs.append(f"line {node.lineno}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") and a.endswith("__") and a not in ALLOWED_DUNDER_ATTRS:
                errs.append(f"line {node.lineno}: dunder attribute '{a}' is not allowed")
            elif a.startswith("_") and not a.startswith("__") and isinstance(node.value, ast.Name) \
                    and node.value.id in {"np", "numpy", "random", "json", "re", "math"}:
                errs.append(f"line {node.lineno}: private module attribute '{a}' is not allowed")
            elif a in BANNED_ATTRS:
                errs.append(f"line {node.lineno}: attribute '{a}' is not allowed")
    return errs


def lint_agent_dir(agent_dir: Path) -> dict[str, list[str]]:
    files = [p for p in agent_dir.iterdir() if p.is_file() and p.suffix == ".py"]
    local = {p.stem for p in files}
    out = {}
    for p in files:
        errs = lint_source(p.read_text(), local)
        if errs:
            out[p.name] = errs
    return out


def defines(src: str, name: str, kind: type) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(isinstance(n, kind) and n.name == name for n in tree.body)


def check_agent_files(agent_dir: Path) -> list[str]:
    errs = []
    entries = list(agent_dir.iterdir())
    if len(entries) > MAX_AGENT_FILES:
        errs.append(f"too many files ({len(entries)} > {MAX_AGENT_FILES})")
    for p in entries:
        if p.is_symlink() or p.is_dir():
            errs.append(f"{p.name}: subdirectories and symlinks are not allowed")
        elif not p.name.endswith(ALLOWED_AGENT_FILE_SUFFIXES):
            errs.append(f"{p.name}: file type not allowed")
        elif p.stat().st_size > MAX_AGENT_FILE_BYTES:
            errs.append(f"{p.name}: larger than {MAX_AGENT_FILE_BYTES} bytes")
    for required, sym, kind in (("strategy.py", "Strategy", ast.ClassDef),
                                ("self_improve.py", "improve", ast.FunctionDef)):
        p = agent_dir / required
        if not p.exists():
            errs.append(f"{required} is missing")
        elif not defines(p.read_text(), sym, kind):
            errs.append(f"{required} must define top-level {sym}")
    return errs


# --------------------------------------------------------------------------- child processes

_UNSHARE_OK: bool | None = None


def _unshare_available() -> bool:
    global _UNSHARE_OK
    if _UNSHARE_OK is None:
        exe = shutil.which("unshare")
        _UNSHARE_OK = False
        if exe and os.environ.get("DGM_NO_UNSHARE") != "1":
            try:
                _UNSHARE_OK = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5).returncode == 0
            except Exception:
                _UNSHARE_OK = False
    return _UNSHARE_OK


def child_env() -> dict:
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(HARNESS_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
    }


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        os.setsid()
        if _drops_privileges():
            # Drop to an unprivileged user so the child cannot read the harness's private
            # files (data, ledger, holdout results are written 0600 by the harness).
            os.setgroups([])
            os.setgid(NOBODY)
            os.setuid(NOBODY)
    return apply


SANDBOX_ROOT = Path(os.environ.get("DGM_SANDBOX_ROOT", "/tmp/dgm-sandbox"))


def _drops_privileges() -> bool:
    return os.getuid() == 0 and os.environ.get("DGM_KEEP_ROOT") != "1"


def _sandbox_copy(src: Path) -> Path:
    import tempfile

    SANDBOX_ROOT.mkdir(mode=0o711, parents=True, exist_ok=True)
    SANDBOX_ROOT.chmod(0o711)
    d = Path(tempfile.mkdtemp(prefix="agent-", dir=SANDBOX_ROOT))
    for p in src.iterdir():
        if p.is_file() and not p.is_symlink() and p.name.endswith(ALLOWED_AGENT_FILE_SUFFIXES):
            shutil.copyfile(p, d / p.name)
            (d / p.name).chmod(0o644)
    d.chmod(0o755)
    return d


class ChildError(RuntimeError):
    pass


class Child:
    """A sandboxed child process speaking newline-delimited JSON over stdin/stdout."""

    def __init__(self, module: str, cwd: Path, mem_mb: int = 4096, cpu_s: int = 3600):
        cmd = [sys.executable, "-m", module]
        if _unshare_available():
            cmd = ["unshare", "-rn"] + cmd
        self.tmpdir = None
        if _drops_privileges():
            # The unprivileged child gets a private, world-readable copy of the agent's files,
            # so it never needs access to the run directory (or anything else of ours).
            self.tmpdir = _sandbox_copy(Path(cwd))
            cwd = self.tmpdir
        self.stderr_chunks: list[str] = []
        self.proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=child_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, preexec_fn=_limits(mem_mb, cpu_s), text=True, bufsize=1)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._buf = ""

    def _drain_stderr(self):
        total = 0
        for line in self.proc.stderr:
            if total < 64_000:
                self.stderr_chunks.append(line)
                total += len(line)

    def stderr_tail(self, n: int = 4000) -> str:
        return "".join(self.stderr_chunks)[-n:]

    def send(self, obj) -> None:
        try:
            self.proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ChildError(f"child exited: {e}\n{self.stderr_tail()}") from e

    def recv(self, timeout: float):
        deadline = time.monotonic() + timeout
        out = self.proc.stdout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChildError(f"timed out after {timeout:.1f}s\n{self.stderr_tail()}")
            r, _, _ = select.select([out], [], [], remaining)
            if not r:
                continue
            line = out.readline()
            if not line:
                self.proc.wait(timeout=5)
                raise ChildError(f"child exited with code {self.proc.returncode}\n{self.stderr_tail()}")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                raise ChildError(f"child wrote non-protocol output: {line[:200]!r}")
            if isinstance(msg, dict) and "error" in msg and len(msg) == 1:
                raise ChildError(str(msg["error"])[-6000:])
            return msg

    def request(self, obj, timeout: float):
        self.send(obj)
        return self.recv(timeout)

    def close(self):
        try:
            if self.proc.poll() is None:
                try:
                    self.send({"op": "close"})
                except ChildError:
                    pass
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=3)
        finally:
            for f in (self.proc.stdin, self.proc.stdout):
                try:
                    f.close()
                except Exception:
                    pass
            if self.tmpdir is not None:
                shutil.rmtree(self.tmpdir, ignore_errors=True)
                self.tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- inside the child

SAFE_BUILTIN_REMOVALS = {
    "open", "eval", "exec", "compile", "input", "breakpoint", "help", "exit", "quit", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "__import__", "__loader__",
    "__spec__",
}


def protocol_streams() -> tuple:
    """Called by host modules inside the child. Returns (proto_in, proto_out) text streams.

    Moves the protocol onto private file descriptors so print() in agent code goes to stderr
    instead of corrupting the protocol.
    """
    import io

    proto_out = io.TextIOWrapper(os.fdopen(os.dup(1), "wb", buffering=0), encoding="utf-8",
                                 line_buffering=True, write_through=True)
    proto_in = io.TextIOWrapper(os.fdopen(os.dup(0), "rb"), encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    sys.stdin = io.StringIO("")
    return proto_in, proto_out


class AgentLoader:
    """Loads agent modules with a restricted __builtins__ and a whitelisting __import__."""

    def __init__(self, agent_dir: Path):
        import builtins
        import types

        self.dir = Path(agent_dir)
        self.local = {p.stem for p in self.dir.glob("*.py")}
        self.modules: dict[str, types.ModuleType] = {}
        self._real_import = builtins.__import__
        safe = {k: v for k, v in vars(builtins).items() if k not in SAFE_BUILTIN_REMOVALS}
        safe["__import__"] = self._guarded_import
        self.safe_builtins = safe

    def _guarded_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if level:
            raise ImportError("relative imports are not allowed")
        if root in self.local:
            return self.load(root)
        if root in ALLOWED_IMPORTS:
            return self._real_import(name, globals, locals, fromlist, level)
        raise ImportError(f"import of '{name}' is not allowed in the agent sandbox")

    def load(self, name: str):
        import types

        if name in self.modules:
            return self.modules[name]
        path = self.dir / f"{name}.py"
        src = path.read_text()
        mod = types.ModuleType(f"agent_{name}")
        mod.__dict__["__builtins__"] = self.safe_builtins
        mod.__dict__["__file__"] = str(path)
        self.modules[name] = mod
        code = compile(src, str(path), "exec")
        exec(code, mod.__dict__)
        return mod


def write_msg(fh, obj) -> None:
    fh.write(json.dumps(obj, separators=(",", ":"), allow_nan=True) + "\n")
    fh.flush()


def protect_private(path: Path) -> None:
    """Make a harness-private file unreadable to the sandbox user."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
