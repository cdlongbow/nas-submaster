#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 模型预下载脚本

用途:
  1. 首次部署时预下载 Whisper 模型到 /data/models
  2. 在容器 entrypoint 中自动调用，避免运行时下载卡死
  3. 国内网络环境下，建议先在能直连 huggingface.co 的环境预下载

用法:
  python scripts/download_whisper_model.py [model_size]
  python scripts/download_whisper_model.py              # 下载默认 base
  python scripts/download_whisper_model.py medium       # 下载指定大小
  python scripts/download_whisper_model.py base,medium  # 一次性下载多个

支持的大小: tiny, base, small, medium, large-v3

注意:
  - 不要设 HF_ENDPOINT 环境变量，否则会因 huggingface_hub 不 follow
    跨域 308 重定向而失败（参见 README 故障排查章节）
  - 如果已经在 /data/models 下检测到完整模型，会自动跳过
"""

import os
import sys
import shutil
from pathlib import Path

# 关键：清除可能存在的 HF_ENDPOINT，避免触发 huggingface_hub 的 308 重定向 bug
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("hf_endpoint", None)

from faster_whisper import WhisperModel

# 模型大小 → faster-whisper 内部仓库名
VALID_SIZES = ("tiny", "base", "small", "medium", "large-v3")


def get_model_dir() -> str:
    """与 whisper_service.get_model_dir 保持一致"""
    docker_path = "/data/models"
    if os.path.isdir(docker_path):
        return docker_path
    return "./data/models"


def is_model_complete(model_size: str, model_dir: str) -> bool:
    """检查模型文件是否已完整下载（参考 whisper_service._verify_model_files）"""
    required_files = ["config.json", "model.bin", "tokenizer.json"]
    snapshots = Path(model_dir) / f"models--Systran--faster-whisper-{model_size}" / "snapshots"
    if not snapshots.is_dir():
        return False
    for commit_dir in snapshots.iterdir():
        if not commit_dir.is_dir():
            continue
        if all(
            (commit_dir / f).exists() and (commit_dir / f).stat().st_size > 0
            for f in required_files
        ):
            return True
    return False


def download_model(model_size: str, model_dir: str) -> bool:
    """下载并加载（faster-whisper 内部会在 download_root 下创建 HF 缓存结构）"""
    if model_size not in VALID_SIZES:
        print(f"[ERROR] 不支持的模型大小: {model_size}")
        print(f"        支持: {', '.join(VALID_SIZES)}")
        return False

    if is_model_complete(model_size, model_dir):
        print(f"[SKIP] {model_size} 已存在且完整: {model_dir}")
        return True

    print(f"[INFO] 正在下载 Whisper {model_size} -> {model_dir}")
    print(f"       （如遇网络问题，请检查 huggingface.co 连通性）")

    try:
        # device=cpu + compute_type=int8 下载最稳，不依赖 GPU
        # 这一步同时验证模型完整性
        _model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,
        )
        del _model
        print(f"[OK]   {model_size} 下载完成")
        return True
    except Exception as e:
        print(f"[FAIL] {model_size} 下载失败: {e}")
        return False


def main():
    # 解析参数
    if len(sys.argv) > 1:
        sizes = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    else:
        sizes = ["base"]  # 默认下载 base

    model_dir = get_model_dir()
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print(f"=== Whisper 模型预下载 ===")
    print(f"目标目录: {model_dir}")
    print(f"待下载:   {', '.join(sizes)}")
    print(f"HF_ENDPOINT: {'已设置（将被忽略）' if os.environ.get('HF_ENDPOINT') else '未设置（推荐）'}")
    print()

    # 清理可能被错误设置的 HF_ENDPOINT
    if "HF_ENDPOINT" in os.environ:
        print(f"[WARN] 检测到 HF_ENDPOINT={os.environ['HF_ENDPOINT']}，已自动清除")
        print(f"       huggingface_hub 1.18+ 不 follow 跨域 308 重定向，")
        print(f"       设为 hf-mirror.com 会导致 'Hub' 错误。建议直连 huggingface.co。")
        os.environ.pop("HF_ENDPOINT")

    print()

    success = 0
    failed = 0
    for size in sizes:
        if download_model(size, model_dir):
            success += 1
        else:
            failed += 1

    print()
    print(f"=== 完成: 成功 {success}, 失败 {failed} ===")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
