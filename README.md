# auto-leafbyte

把 [LeafByte](https://github.com/TheBeruriahIncident/leafbyte)（手机 App，测叶片被啃食面积）的
计算核心移植成**可批处理的 Python 流水线**，并实现了你真正需要的那件事：

> **大部分照片全自动出结果；精度可疑的自动挑出来，进人工审核队列。**

不需要 App、不需要手机、不需要点屏幕。一条命令处理整个文件夹：

```bash
python -m leafbyte run 图片文件夹 --out out/结果夹 --scale-cm 17
```

---

## 1. 5 分钟上手

需要 Python 3.10+，只依赖 `numpy` 和 `Pillow`（无 cv2/scipy，不联网运行）。

```bash
export PYTHONPATH=src
PY=python          # 换成你的解释器路径也可以

# ⓪ 首次运行：拉取上游 LeafByte 的 30 张论文测试图（约 57 MB，之后可离线）
"$PY" tools/fetch_paper_data.py

# ① 批量测量：自动 / 待审核 自动分流
"$PY" -m leafbyte run data/paper/images --out out/paper --scale-cm 17

# ② 看汇总
"$PY" -m leafbyte report --out out/paper -v

# ③ 人工审核：用浏览器打开（双击即可，纯离线，不联网）
#    out/paper/review/index.html
#    逐张点选候选/手填数值/标记弃用，然后点「导出 decisions.json」

# ④ 合并人工结论，出最终表
"$PY" -m leafbyte apply --out out/paper \
     --decisions out/paper/review/decisions.json --reviewer 张三
#  -> out/paper/final_results.csv
```

Windows 下也可以直接双击 `run_demo.cmd`（跑论文 30 张 + 生成审核页）。

---

## 2. 输出物

| 文件 | 内容 |
| --- | --- |
| `out/results.csv` | 主表：每张照片的状态、总叶面积、被啃食面积、百分比、置信度、标记 |
| `out/results.jsonl` | 全量中间结果（阈值、比例尺四点、连通域、稳定性、候选、质量指标），可复查/二次分析 |
| `out/review/index.html` | **离线人工审核页**（双击打开，无服务器、无网络）：原图 + 叠加图 + 候选对比，键盘操作，导出 decisions.json |
| `out/review/queue.json` | 待审队列（含每张的候选方案与原因） |
| `out/overlays/` | 可视化叠加图：红=叶子，蓝=被啃食孔洞，黄=检出的比例尺四点 |
| `out/figures/` | **阈值图**（每张一张）：矫正图 + 子像素叶缘轮廓（绿线）+ 叶片二值掩膜 + 阈值直方图（红线=实际阈值）<br>**矢量轮廓**：`<样本>_contour.svg`（cm 单位，可编辑，含 0–5 cm 比例条与样本编号）；加 `--pdf` 才额外导出单页矢量 PDF<br>**图版**：`<样本>_plate.jpg`（带比例条+编号的位图，可直接拼图）<br>`<样本>_mask.png` 全分辨率二值掩膜 |
| `out/leaf_traits.csv` | **叶片性状表**（每张一行，22 个指标：周长/叶长/叶宽/长宽比/圆度/凸度/矩形度/分裂指数/叶色/GCC/ExG/孔洞等）；填写干重后可加 SLA、LMA |
| `out/final_results.csv` | 合并人工审核后的终表，带 `source`（auto / review:谁）与审核备注，可追溯 |
| `out/calibration.json` | 置信度阈值标定结果与取舍曲线（用已知真值的样本算出） |
| `out/run.json` | 本次运行的配置、耗时、汇总、与真值的误差统计 |

状态只有三种：

- `auto` —— 自动接受，直接用
- `review` —— **精度可疑，必须人工看**（队列里带原因，例如"面积对阈值不稳定""没有找到比例尺""另一块大斑块和叶子竞争"）
- `failed` —— 处理失败/无法给值，需要重拍或手测

---

## 3. 实测精度（30 张论文原图，真值见 `data/paper/ground_truth.csv`）

论文配图 = 白纸 + 叶子 + 四个黑点标出 17 cm×17 cm 方框。用这 30 张做验证：

| 指标 | 结果 |
| --- | --- |
| 比例尺自动检出 | **30 / 30**（上游 App 的自动检测在 9 张上失败，所以 iOS 测试里把它们注释掉了） |
| 自动接受 | **25 / 30**（83%），其余 5 张进人工审核 |
| 自动接受的样本里，误差 ≤1.5% | **24 / 24 = 100%**（0 张"自信地给错值"） |
| 自动接受的平均绝对误差 | **0.122 cm²**（相对 0.50%），最差 0.311 cm²（1.36%） |
| 上游完全测不了的 9 张 | 我们全部测出且误差 ≤1.3%，其中 7 张被自动接受 |
| 速度 | **325 ms / 张**（1200 px，单核），30 张共 11 秒 |

误差分布是**系统性偏小约 0.5%**：上游是把二值图做透视矫正后按 alpha≠0 记前景（重采样会把边缘"涨"出去一点点），
我们是矫正原图再二值化，几何上更干净。这个 0.5% 对"百分比被啃食"几乎没影响（两个面积同向变化）。
如果你们要跟历史 LeafByte 数据严格对齐，可以用 `--threshold` 或加一个全局系数（见 §7 配置）。

审核队列的取舍（`out/paper/calibration.json` 里有完整曲线）：把 `auto_accept_min` 从 0.92 降到 0.85，
自动接受 26 张，但会多放进来 1 张偏差 2.6% 的样本；提到 0.96 则只剩 20 张自动。**默认 0.92 是"零误收"点。**

---

## 4. 人工审核页怎么用

1. 双击 `out/review/index.html`（Edge/Chrome 均可，**不需要联网**）。
2. 每张卡片左边是原图（黄框=自动找到的 17 cm 方框），右边是矫正后的测量图（红=算作叶子的像素，蓝=算作被啃食的孔洞）。
3. 下方是几个候选方案（换阈值各测一遍），点一下就能预览对应数字。
4. 三种决定：
   - **接受某个候选**（多数情况）
   - **手填数值**（你用 ImageJ 之类量过，直接覆盖）
   - **弃用/需重拍**（反光、糊、没比例尺、叶子裂开…）
5. 点「导出 decisions.json」，把文件放回 `out/review/` 再跑 `leafbyte apply`。
6. 详见 [`docs/REVIEW_WORKFLOW.md`](docs/REVIEW_WORKFLOW.md)。

`apply` 会给每条记录打上 `source=review:张三`、原置信度、原阈值、审核备注 —— **哪张是机器给的、哪张是人改的，一目了然。**

---

## 4.2 汇总数据表（推荐出口）

把所有 run 汇总成一张表（多数据集合并）：

```bash
PYTHONPATH=src python tools/make_summary_table.py --run out/yanfumu --run out/mujing --out out/汇总
```

输出 `out/汇总/叶片性状汇总.xlsx`（三个工作表）：

| 工作表 | 内容 |
| --- | --- |
| 样本数据 | 每样本一行，34 列：状态/置信度/面积/取食率/全部性状/孔洞/内部暗斑/分割方式/标志/备注；带筛选与冻结窗格 |
| 性状统计 | 每个指标 × {全部样本, 各数据集}：n、均值、标准差、最小值、中位数、最大值 |
| 说明 | 比例尺、面积与取食率定义、状态与标志含义、干重指标补充方法 |

同名 `叶片性状汇总.csv` 一定会写（UTF-8 BOM，Excel 直接打开）；Excel 版本需要 `openpyxl`
（缺了会提示 `python -m pip install openpyxl`，其余功能不受影响）。

## 4.5 叶片性状指标（每张样本）

| 类别 | 指标 |
| --- | --- |
| 面积 | 剩余叶面积、完整叶面积（含取食孔洞）、凸包面积、等效直径 |
| 尺寸 | 叶周长、叶长、叶宽（最小外接矩形）、最大/最小卡径 |
| 形状（无量纲） | 长宽比、圆度 4πA/P²、圆整度、凸度（实心度 A/凸包）、矩形度、叶缘分裂指数 P/(2√(πA)) |
| 叶色 | 平均 R/G/B、绿色色度坐标 GCC、过绿指数 ExG（仅叶片像素） |
| 取食 | 孔洞数、平均孔洞面积 |
| 干重相关 | 比叶面积 SLA（cm²/g）、比叶重 LMA（g/m²）——需要干重，见下 |

干重相关指标用一条命令补上（CSV 里给 `file,dry_mass_mg`）：

```bash
PYTHONPATH=src python tools/attach_dry_mass.py --run out/yanfumu --masses masses.csv
```

## 5. 算法（与上游逐条对齐，见 `docs/ALGORITHM.md`）

1. 归一化：EXIF 摆正，等比缩到最长边 ≤1200，不放大（上游同）。
2. luma = (0.299R+0.587G+0.114B)/255，256 桶直方图，**Otsu** 取阈值（平局取大桶号，与上游一致）。
3. 白背景二值化：`luma < 阈值` 记为前景（严格小于）。
4. **自适应前景分割**（本版新增，白纸照片的行为不变）：
   - 默认走上面的亮度/Otsu 路径（与上游完全一致）；
   - 当亮度结果明显不对（背景被当成叶子：最大连通域贴边、或出现与叶片竞争的大斑块、或"啃食面积"全是 1–2 px 噪点）时，
     自动切换到 **过绿指数 ExG = 2G−R−B** 的颜色分割——适合灰白/带暗角/欠曝的背景；
   - 比例尺四点若在 Otsu 下被背景吞掉，会用**绝对暗阈值**（亮度 0.15–0.30 依次尝试）再找一遍；
   - 颜色阈值是**自适应**的：先对 ExG 直方图做 Otsu，再取两类均值的**中点**（谷底中心），
     避免 Otsu 在"亮叶子占少数"时把切点推到谷的上沿、进而削掉叶缘；
   - 记录里会写明 `segmentation.mode`（`luma` / `color`）和实际使用的 `color_threshold`。

5. **4 连通**连通域标注：正标签=前景，负标签=背景/孔洞，贴边背景并成 -1；
   用"行程编码 + 并查集"向量化重写，语义与上游 `labelConnectedComponents` 完全一致
   （已用上游 iOS 单测的数值验证：Otsu=139/255、分量尺寸 [3358, 970002] / [1178, 105400] 等）。
5. 比例尺：在小的、圆的、不贴边的斑块里**搜索"近正方形"的四边形**（凸性、边长比、面积占比、四点尺寸一致性打分），
   必要时用 3 点 + 平行四边形补第 4 点（降置信度并标记）。
   比上游"取第 2~5 大斑块"稳健得多——上游那套遇到相框黑边/反光/淡掉的点就废。
6. 透视矫正：四点单应矩阵映射到 S×S（S=min(1200, ⌊min(bbox_w,bbox_h)⌋)），矫正**原图**。
   方框外（图像外）的采样填白，避免把照片边缘的暗像素拉进掩膜。
7. **孔洞真伪校验（颜色模式）**：ExG 把"非绿"都算空，但褐色病斑/褶皱阴影是"组织还在"。
   因此只把**内部颜色接近纸张背景**的孔洞计入"被取食面积"；明显比纸暗的内部斑块
   （luma < 0.55 × 纸张亮度）填回叶片，并在记录里标 `dark_patches_filled` / `filled_dark_px`。
7. 面积：矫正图上重新二值化；叶子=最大前景连通域；被啃食=与叶子相邻、且不与图外相连的孔洞像素；
   总面积=叶子+啃食；`cm² = 像素 × (scale_cm / S)²`。
8. 质量评估（`quality.py`）：比例尺质量、阈值双峰性/Otsu 分离度、**±阈值扰动下的面积稳定性**、
   叶子是否触边、是否有第二块大斑块竞争、孔洞噪声、叶面积占比、被啃食百分比极端值 → 加权成 0~1 置信度 + 硬性拦截规则。
9. 分流：`confidence ≥ auto_accept_min` 自动接受；否则进审核；低于 `review_min` 记为 failed。

本仓库还自带"为什么信任它"的证据：`tests/` 里 139 个测试，包括**复现上游 iOS 测试的数值**与
`data/paper` 全部 30 张的端到端回归。

---

## 6. 目录

```
src/leafbyte/
  imageops.py   载入/缩放/luma/直方图/Otsu/二值化/单应矫正/叠加图   (Lead)
  ccl.py        4 连通连通域（行程+并查集，与上游语义一致）          (Lead)
  scale.py      4 点比例尺检测 + 透视矫正                           (teammate: scale)
  measure.py    叶面积/孔洞统计、像素→cm²                            (teammate: measure-qc)
  quality.py    质量指标 + 置信度 + 自动/审核/失败 分流               (teammate: measure-qc)
  review.py     离线审核页生成 + 人工决策合并                        (teammate: review-ui)
  batch.py      单张流水线 + 批处理编排 + 真值比对                    (Lead)
  cli.py        命令行 (run/review/apply/report/calibrate)          (Lead)
  calibrate.py  用已知真值标定置信度阈值/权重                        (Lead)
  types.py config.py 数据契约与配置                                 (Lead)
tests/          146 个单元/回归测试（unittest，无需 pytest）
tools/          fetch_paper_data.py（拉取上游测试图）、make_demo.py（重建演示劣化图）
data/paper/     ground_truth.csv（论文发表的面积真值）；图片由 tools/fetch_paper_data.py 拉取
data/demo/      演示用劣化图（模糊/反光/撕裂/欠曝），由 tools/make_demo.py 生成
docs/           ALGORITHM.md（算法与契约）、REVIEW_WORKFLOW.md（审核手册）、upstream/（上游关键源码摘录）
```

跑测试：

```bash
PYTHONPATH=src "$PY" -m unittest discover -s tests -t . -v
```

---

## 7. 常用配置

```bash
# 比例尺物理长度（论文协议=17 cm；LeafByte 默认 10 cm）
--scale-cm 17
# 强制阈值（不想用 Otsu 时）
--threshold 0.5
# 黑背景（例如白叶子拍在黑布上）
--background black
# 叶片分割方式：auto（默认）/ luma（纯白纸）/ color（灰背景，ExG）
--leaf-mode auto
# 灰背景照片的 ExG 阈值（默认 5.0）
--color-threshold 5
# 给每张都导出叠加图（默认只给待审核的）
--overlays all
# 自动接受的置信度门槛（越低越省人工，风险越高）
--auto-accept-min 0.85
# 用现有配置重跑
--config out/paper/config.json
# 校准：用已知真值重新标定门槛/权重
"$PY" -m leafbyte calibrate --out out/paper --ground-truth data/paper/ground_truth.csv --write-config
```

---

## 8. 局限与下一步（诚实说明）

- **比例尺协议**：当前自动识别的是论文那套"四个圆点标 17 cm 方框"。如果你现在用的是 App 的
  扫码/单根比例尺协议，需要在 `scale.py` 里加一个"单尺度/已知边长"的检测分支
  （`ScaleDetection.method` 已经预留；其余流水线不用动）。
- **被啃食面积是相对量**：`percent_consumed` 不依赖比例尺，即使没找到比例尺也照样能算（上游也是这个设计），
  这类样本会以"只有相对值"进审核队列。
- **±1.4% 的测量精度**已经接近这套照片的极限（叶子边缘与纸的对比度、JPEG/PNG 解压差异、Otsu 的单阈值假设）。
  要更准就得从拍摄端下手：均匀光、避免反光、叶子别压到方框外。
- **审核页是给人看的**：没有做多用户/权限/数据库；决策以 JSON 文件流转，方便存档和审计。
- **审核人姓名**：仓库里示例结果中的审核人已匿名化为 `reviewer-01`；你自己的运行会记录真实姓名，发布前请自行确认。
- 想接你们的 LIMS/表格：`results.jsonl` 是稳定 schema，`batch.attach_reference` 已经演示了如何按真值算误差。

---

## 9. 许可与出处

本项目是 LeafByte 的**衍生作品**，遵循 **GPL-3.0**（见 `LICENSE`、`NOTICE.md`）。

- 算法与测试数据来自 LeafByte（GPL-3.0，`LICENSE` 保留了原文），
  作者 Abigail Getman-Pickering 等；论文数据取自上仓库 `iOS/LeafByteTests/Images/Paper Data`。
- 本仓库是对其计算核心的独立 Python 重实现 + 批处理/审核层，未改动上游任何文件（`ext/` 仅为阅读用副本）。
