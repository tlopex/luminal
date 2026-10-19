"""Run the SDK builder from a declarative recipe, without provider-specific logic."""
import contextlib
import json
import os
from pathlib import Path
import sys

from tvm_ffi.cpp import build


def expand(value):
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


recipe = Path(sys.argv[1]).resolve()
spec = expand(json.loads(recipe.read_text()))
for key in ("sources", "cpp_files", "cuda_files", "extra_include_paths"):
    if spec.get(key) is not None:
        values = [spec[key]] if isinstance(spec[key], str) else spec[key]
        spec[key] = [str((recipe.parent / value).resolve()) for value in values]
if spec.get("build_directory") is not None:
    spec["build_directory"] = str((recipe.parent / spec["build_directory"]).resolve())
with contextlib.redirect_stdout(sys.stderr):
    artifact = build(**spec)
print(Path(artifact).resolve())
