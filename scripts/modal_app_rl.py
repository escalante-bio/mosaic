import sys
from pathlib import Path
import modal


image = (
    modal.Image.debian_slim(python_version="3.12.0")
    .apt_install("git", "aria2")
    .env({"JAX_PLATFORMS": "cuda"})
    .run_commands(
        # Base tooling
        "python -m pip install -U pip setuptools wheel && "
        # CUDA PyTorch (provides CUDA libs in container)
        "python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 && "
        # JAX + CUDA plugin (pins matching versions used elsewhere)
        "python -m pip install --upgrade jax==0.7.1 && "
        "python -m pip install --upgrade jax-cuda12-plugin==0.7.1 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && "
        # Provide cuDNN shared libs for the JAX CUDA plugin
        "python -m pip install nvidia-cudnn-cu12==8.9.2.26 && "
        # PTX toolchain
        "python -m pip install nvidia-cuda-nvcc-cu12==12.8.93 && "
        # RL stack + Mosaic deps
        "python -m pip install transformers==4.44.2 datasets==2.20.0 accelerate==0.34.2 trl==0.9.6 tqdm optax==0.2.4 equinox==0.13.0 && "
        # Joltz/Boltz (keep parity with other apps)
        "python -m pip install git+https://github.com/adaptyvbio/joltz.git && "
        "python -m pip install git+https://github.com/jwohlwend/boltz.git && "
        # Bake repo for stable imports; latest edits mounted via Mount below
        "git clone --depth 1 https://github.com/adaptyvbio/mosaic_workflows.git /repo"
    )
)


app = modal.App("mosaic-rl", image=image)


local_src_mount = modal.Mount.from_local_dir(
    "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/src", remote_path="/workspace/src"
)
local_examples_mount = modal.Mount.from_local_dir(
    "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/examples", remote_path="/workspace/examples"
)
local_tiny_mount = modal.Mount.from_local_dir(
    "/Users/tudorcotet/Documents/Adaptyv/mosaic_workflows/_external/tiny-clean-test", remote_path="/workspace/_external/tiny-clean-test"
)


def _add_paths():
    # Prefer mounted local source for latest edits; fallback to baked repo
    # Ensure local src wins over baked repo
    paths_front = ["/workspace/src"]
    paths_back = ["/repo/src", "/workspace", "/repo"]
    for p in paths_back:
        if p not in sys.path and Path(p).exists():
            sys.path.append(p)
    for p in reversed(paths_front):
        if p not in sys.path and Path(p).exists():
            sys.path.insert(0, p)


@app.function(gpu="H100", timeout=2 * 60 * 60, mounts=[local_src_mount, local_examples_mount, local_tiny_mount])
def run_rl_tiny(binder_len: int = 50, prompt: str = "M"):
    # Allow JAX to fall back to CPU for tiny test if CUDA plugin cannot load
    import os as _os
    _os.environ["JAX_PLATFORMS"] = ""
    _add_paths()
    import os as _os
    from importlib.machinery import SourceFileLoader as _Loader
    cand = [
        "/workspace/examples/rl_tiny_llama.py",
        "/repo/examples/rl_tiny_llama.py",
    ]
    path = next(p for p in cand if Path(p).exists())
    ex = _Loader("rl_tiny_llama", path).load_module()  # type: ignore
    ex.main()
    return {"status": "ok"}


@app.local_entrypoint()
def main():
    print(run_rl_tiny.remote())


