# Source before any project command: keeps caches inside the repo and restricts GPUs.
export SELFEVOLVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_CACHE_HOME="$SELFEVOLVE_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export XDG_CONFIG_HOME="$SELFEVOLVE_ROOT/.config"  # e.g. FreeCAD user config
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export FLASHINFER_CACHE_DIR="$XDG_CACHE_HOME/flashinfer"
export VLLM_USE_FLASHINFER_SAMPLER=0  # its JIT build fails with the system nvcc 11.5

# PCI order: 0,1 = free RTX PRO 6000; 2 = A1000 (never use); 3,4 = held by another job.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
