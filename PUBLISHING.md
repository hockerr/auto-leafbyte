# 发布说明 / Publishing notes

## 这个目录是什么

`tools/make_release.py` 从开发工作区生成的**可公开发布版本**：

- 只包含源码、测试、文档、工具和**匿名化后的示例结果**；
- 不含 57 MB 的论文原图（用 `tools/fetch_paper_data.py` 按需从上游拉取）；
- 不含 80 MB 的上游仓库快照（`ext/`，仅本地对照用）；
- 审核人姓名已替换为 `reviewer-01`，本机绝对路径已清除。

## 上传到 GitHub（两种方式）

### A. 用脚本（无需本地 git 可用）

```bash
# 首次创建仓库 / 之后更新都用同一条命令；--replace 让远端文件与本地完全一致
python tools/publish_github.py --token <你的 PAT> --repo auto-leafbyte --public --replace
```

- Token 需要 `repo` 权限（fine-grained token：`Administration: read/write` +
  `Contents: read/write`）。
- 会在你的账号下创建仓库（已存在则直接推送），并用 Git Data API 提交。
- 用完请到 GitHub 设置里删除/轮换这个 token。

### B. 本地 git

```bash
cd release/auto-leafbyte
git remote add origin https://github.com/<你的账号>/auto-leafbyte.git
git push -u origin main
```

（仓库里已经 `git init` 并做了首个 commit；`git commit --amend --reset-author`
可把提交者改成你自己。）

## 隐私自检

构建脚本最后会扫描"审核人姓名、本机 Python 安装路径、工作区绝对路径"这几类字符串，
命中即中止构建。你也可以自己再确认一遍：

```bash
# 仓库中不应出现任何本机绝对路径或个人姓名
grep -rIn -E "[A-Za-z]:[\/](Users|dsh)" . --exclude-dir=.git || echo "clean"
```

## 许可

衍生自 LeafByte（GPL-3.0），本仓库同样 GPL-3.0；见 `LICENSE` 与 `NOTICE.md`。
论文原图不随仓库分发。
