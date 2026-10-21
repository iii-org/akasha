"""CPython-only import attribution for the offline startup diagnostic."""

from importlib import _bootstrap, _bootstrap_external
import sys
import time
import os
import threading


class ImportTrace:
    def __init__(self, phase):
        self.phase = phase
        self.files = {}
        self.modules = []
        self.cache_stack = []
        self.patches = []
        self.cache_operations = []
        self.atomic_record = None

    def record(self, loader):
        key = (self.phase(), loader.name, loader.path)
        if key not in self.files:
            self.files[key] = dict(phase=key[0], module=key[1], source=key[2],
                                   compile_s=0.0, cache_s=0.0, atomic_s=0.0,
                                   compile_calls=0, cache_calls=0,
                                   writes=0, attempts=0, bytes=0)
        return self.files[key]

    def patch(self, owner, name, replacement):
        self.patches.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self):
        loader_class = _bootstrap_external.SourceFileLoader
        original_compile = loader_class.source_to_code
        original_cache = loader_class._cache_bytecode
        original_atomic = _bootstrap_external._write_atomic
        original_load = _bootstrap._load_unlocked

        if os.environ.get("AKASHA_STARTUP_CACHE_OPS") == "1":
            self.install_operation_hooks()

        def compile_source(loader, *args, **kwargs):
            record = self.record(loader)
            started = time.perf_counter()
            try:
                return original_compile(loader, *args, **kwargs)
            finally:
                record["compile_s"] += time.perf_counter() - started
                record["compile_calls"] += 1

        def cache_bytecode(loader, *args, **kwargs):
            record = self.record(loader)
            self.cache_stack.append(record)
            started = time.perf_counter()
            try:
                return original_cache(loader, *args, **kwargs)
            finally:
                record["cache_s"] += time.perf_counter() - started
                record["cache_calls"] += 1
                self.cache_stack.pop()

        def write_atomic(path, data, mode=0o666):
            record = self.cache_stack[-1] if self.cache_stack else None
            previous = self.atomic_record
            self.atomic_record = record
            if record is not None:
                record["cache_path"] = str(path)
            started = time.perf_counter()
            try:
                result = original_atomic(path, data, mode)
                if record is not None:
                    record["writes"] += 1
                    record["bytes"] += len(data)
                return result
            except OSError as error:
                if record is not None:
                    record.setdefault("write_errors", []).append({
                        "type": type(error).__name__, "errno": error.errno,
                        "winerror": getattr(error, "winerror", None),
                        "path_length": len(path),
                    })
                raise
            finally:
                self.atomic_record = previous
                if record is not None:
                    record["atomic_s"] += time.perf_counter() - started
                    record["attempts"] += 1

        def load_module(spec):
            frame = sys._getframe(1)
            while frame and frame.f_globals.get("__name__", "").startswith("importlib"):
                frame = frame.f_back
            event = {"module": spec.name, "source": spec.origin,
                     "importer": frame.f_globals.get("__name__", "") if frame else "",
                     "importer_file": frame.f_code.co_filename if frame else "",
                     "importer_line": frame.f_lineno if frame else 0,
                     "phase": self.phase(), "inclusive_s": 0.0}
            self.modules.append(event)
            del frame
            started = time.perf_counter()
            try:
                return original_load(spec)
            finally:
                event["inclusive_s"] = time.perf_counter() - started

        self.patch(loader_class, "source_to_code", compile_source)
        self.patch(loader_class, "_cache_bytecode", cache_bytecode)
        self.patch(_bootstrap_external, "_write_atomic", write_atomic)
        self.patch(_bootstrap, "_load_unlocked", load_module)
        return self

    def timed_operation(self, operation, record, path, function, *args, **kwargs):
        if record is None:
            return function(*args, **kwargs)
        started_utc = time.time_ns()
        started = time.perf_counter_ns()
        error = None
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            ended = time.perf_counter_ns()
            self.cache_operations.append({
                "operation": operation, "module": record["module"],
                "phase": record["phase"], "path": str(path),
                "pid": os.getpid(), "tid": threading.get_native_id(),
                "start_utc_ns": started_utc, "start_qpc_ns": started,
                "end_qpc_ns": ended, "seconds": (ended - started) / 1e9,
                "error": error,
            })

    def install_operation_hooks(self):
        trace = self
        native = _bootstrap_external._os
        original_open, original_replace = native.open, native.replace
        original_mkdir = native.mkdir
        original_file = _bootstrap_external._io.FileIO

        def open_file(path, *args, **kwargs):
            return trace.timed_operation("open", trace.atomic_record, path,
                                         original_open, path, *args, **kwargs)

        def replace_file(source, target, *args, **kwargs):
            return trace.timed_operation("replace", trace.atomic_record, target,
                                         original_replace, source, target, *args, **kwargs)

        def mkdir(path, *args, **kwargs):
            record = trace.cache_stack[-1] if trace.cache_stack else None
            return trace.timed_operation("mkdir", record, path,
                                         original_mkdir, path, *args, **kwargs)

        class TimedFileIO(original_file):
            def __init__(self, *args, **kwargs):
                self._cache_record = trace.atomic_record
                self._cache_path = (self._cache_record or {}).get("cache_path", "")
                trace.timed_operation("fd_wrap", self._cache_record, self._cache_path,
                                      super().__init__, *args, **kwargs)

            def write(self, data):
                return trace.timed_operation("write", self._cache_record, self._cache_path,
                                             super().write, data)

            def close(self):
                record = self._cache_record if not self.closed else None
                return trace.timed_operation("close", record, self._cache_path,
                                             super().close)

        self.patch(native, "open", open_file)
        self.patch(native, "replace", replace_file)
        self.patch(native, "mkdir", mkdir)
        self.patch(_bootstrap_external._io, "FileIO", TimedFileIO)

    def __exit__(self, *exception):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)

    @staticmethod
    def package(module, source):
        if module.startswith("akasha"):
            return "akasha"
        if "site-packages" not in (source or "").lower():
            return "[stdlib]"
        parts = module.split(".")
        return ".".join(parts[:2]) if parts[0] == "google" else parts[0]

    def result(self):
        packages = {}
        for record in self.files.values():
            name = self.package(record["module"], record["source"])
            summary = packages.setdefault(name, dict(package=name, modules=0, compile_s=0.0,
                                                      cache_s=0.0, writes=0, attempts=0, bytes=0))
            summary["modules"] += 1
            for metric in ("compile_s", "cache_s", "writes", "attempts", "bytes"):
                summary[metric] += record[metric]
        for record in self.files.values():
            record["package"] = self.package(record["module"], record["source"])
        return {"packages": sorted(packages.values(), key=lambda p: p["cache_s"], reverse=True),
                "files": sorted(self.files.values(), key=lambda p: p["cache_s"], reverse=True),
                "modules": self.modules,
                "cache_operations": self.cache_operations,
                "process_id": os.getpid(),
                "note": "Cache times are per-file and additive; module inclusive times overlap. Importers identify the caller that first loaded each module."}
