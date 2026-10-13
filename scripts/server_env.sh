#!/usr/bin/env bash
# Source before rendering; override paths for a different workspace.
export YEAHPICK_ROOT="${YEAHPICK_ROOT:-/root/YeahPick}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/root/.config/vulkan/nvidia_egl_icd.json}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/root/.cache/xdg-runtime}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export HF_HUB_DISABLE_XET=1
