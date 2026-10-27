# -*- coding: utf-8 -*-
"""
仓库质量门禁 —— 基础可读版（quality_check_simple.py）
=====================================================
这个文件做的事，和原版 quality_check.py 完全一样，对整个仓库跑 4 条检查：
  1. 英文名的文档（.md / .html）里不许出现一个汉字
  2. 中文名的文档（文件名带中文，或文件名以 .zh-CN 结尾）里必须有汉字
  3. README / index 里引用的图片资源（assets/ 下的 png/jpg 等）必须真实存在
  4. 4 个双语文件必须都在：README.md、README.zh-CN.md、index.html、index.zh-CN.html

全部通过打印 "QUALITY OK" 并返回 0；有任何问题打印 "QUALITY FAIL"
和逐条错误，返回 1。

用法（和原版一样）：
    python scripts/quality_check_simple.py [仓库根目录]
不传目录就默认当前目录。
"""

import re
import subprocess
import sys
from pathlib import Path

# 目录：要检查哪个仓库（命令行第一个参数，或当前目录）
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# 汉字匹配：CJK 统一汉字区段 U+4E00~U+9FFF（这是本工具的核心功能，
# 必须用正则，不属于"数据处理里刻意避免正则"的范畴）
CJK = re.compile("[\\u4e00-\\u9fff]")

# 文档里引用图片的写法：assets/文件名.png 之类
ASSET_RE = re.compile(r"assets/[\w\-.]+\.(?:png|jpg|jpeg|gif|svg)")

# 必须存在的 4 个双语文件
REQUIRED = ["README.md", "README.zh-CN.md", "index.html", "index.zh-CN.html"]


def count_cjk(text):
    """数一段文本里有多少个汉字。"""
    return len(CJK.findall(text))


def is_zh_name(name):
    """判断一个文件名是不是"中文文档"：文件名里有汉字，或带 .zh-CN 后缀。"""
    base = Path(name).name
    if CJK.search(base):
        return True
    if ".zh-CN" in base:
        return True
    return False


def main():
    errors = []

    # ---- 1/2. 文档语言一致性 ----
    # 只检查 git 跟踪的 .md/.html（本地未跟踪的草稿文档不算在内，
    # 这样门禁反映的就是"真正会发布出去的内容"）
    r = subprocess.run(
        ["git", "ls-files", "*.md", "*.html"],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    )
    doc_files = []
    for line in r.stdout.splitlines():
        if line:
            doc_files.append(root / line)

    seen = set()
    for fp in doc_files:
        p = fp.resolve()
        # 同一文件只看一次；点开头的隐藏文件跳过
        if p in seen or fp.name.startswith("."):
            continue
        seen.add(p)

        # 读不出来（编码异常等）就跳过这份文件
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        n = count_cjk(text)
        if is_zh_name(fp.name):
            # 中文文档里没有汉字 → 有问题
            if n == 0:
                errors.append("[" + fp.name + "] Chinese-named doc has no Chinese content")
        else:
            # 英文文档里混进了汉字 → 有问题
            if n > 0:
                errors.append("[" + fp.name + "] English-named doc contains " + str(n) + " CJK chars")

    # ---- 3. README/index 里引用的图片资源必须真实存在 ----
    for fname in REQUIRED:
        fp = root / fname
        if not fp.is_file():
            errors.append("missing " + fname)
            continue
        text = fp.read_text(encoding="utf-8", errors="replace")
        for ref in ASSET_RE.findall(text):
            if not (root / ref).is_file():
                errors.append("missing asset referenced in " + fname + ": " + ref)

    # ---- 4. 4 个双语文件都必须存在 ----
    for fname in REQUIRED:
        if not (root / fname).is_file():
            errors.append("missing " + fname)

    # ---- 输出结果 ----
    if errors:
        print("QUALITY FAIL")
        for e in sorted(set(errors)):
            print(" -", e)
        return 1

    print("QUALITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
