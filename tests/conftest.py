"""测试级 conftest：修复 hermes venv 的 sys.path 屏蔽问题。

问题：hermes-agent 自带 /root/.hermes/hermes-agent/utils.py，
当 pytest 在 /home/dev/nas-submaster 跑时，sys.path 同时包含两边，
import utils 命中 hermes-agent 的 utils.py 而非 nas-submaster 的 utils/ 包。

修复：把 nas-submaster 强制放到 sys.path[0]，并清掉 hermes-agent 这类
"屏蔽者"的优先级。

不影响生产环境（这是测试基础设施问题，CI 用 Docker 跑没这个冲突）。
"""
import sys
import os

# 项目根目录：conftest.py 在 tests/ 下，根目录是父目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1) 清掉所有"屏蔽者"位置（除标准库和 site-packages 外的非项目路径）
_clean_path = []
for p in sys.path:
    if p == PROJECT_ROOT:
        continue  # 跳过，稍后插入到首位
    if p in ("", ".") or p.startswith("/usr/") or "site-packages" in p or "dist-packages" in p:
        _clean_path.append(p)
        continue
    if "hermes" in p or "/tmp/" in p:
        continue
    _clean_path.append(p)

sys.path = [PROJECT_ROOT] + _clean_path
