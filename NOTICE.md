# NOTICE — 出处与许可 / Attribution

本仓库 **auto-leafbyte** 是 [LeafByte](https://github.com/TheBeruriahIncident/leafbyte)
的衍生作品（derivative work）：

- LeafByte 是一个用于测量叶片被啃食面积的免费开源手机 App（iOS / Android），
  作者 Abigail Getman-Pickering 等，许可证 **GNU GPL v3**。
- 本仓库把 LeafByte 的**计算核心**（Otsu 阈值、4 连通域标注、四点比例尺矫正、
  叶面积/被啃食面积统计、像素→cm² 换算）用 Python 重新实现，并增加了
  批处理、质量评分/置信度、离线人工审核页与标定工具。
- `docs/upstream/` 下保留了少量上游 Swift 源码摘录，仅用于逐行对照算法；
  版权归原作者，同样遵循 GPL-3.0。
- `data/paper/ground_truth.csv` 中的面积真值取自 LeafByte 论文/测试套件
  （`iOS/LeafByteTests/PaperDataTests.swift` 与论文发表值）。
- 论文原图（30 张）版权归原作者，**不在本仓库中分发**；
  `tools/fetch_paper_data.py` 会从上游仓库按需下载。

依 GPL-3.0 要求，本仓库整体同样以 **GPL-3.0** 发布，完整许可证文本见 `LICENSE`。
若你分发本项目的修改版，请保留本 NOTICE 与 `LICENSE`。

---

This project is a **GPL-3.0** derivative work of LeafByte
(https://github.com/TheBeruriahIncident/leafbyte), which is itself GPL-3.0.
Only a few upstream source excerpts are vendored under `docs/upstream/` for
algorithm-by-algorithm comparison; the 30 paper photographs are **not**
redistributed and are fetched on demand from the upstream repository by
`tools/fetch_paper_data.py`.
