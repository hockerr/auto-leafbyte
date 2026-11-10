# auto-leafbyte — 算法移植与协作契约

本文件是团队协作的**唯一契约**。所有人先读它，再动手。

## 1. 目标

把 LeafByte（iOS/Android 手机 App，用于测量叶片被啃食面积）的**计算核心**移植成
可批处理的 Python 流水线，并在**精度可疑时自动转入人工审核队列**。

最终产物：`leafbyte` CLI

```
leafbyte run   <图片目录>  --scale-cm 17  --out out/      # 自动批处理
leafbyte review --out out/                                      # 生成人工审核页面
leafbyte apply out/ --decisions out/review/decisions.json  # 合并人工结论，出最终表
leafbyte report out/                                      # 汇总统计
```

## 2. 上游代码在哪、哪些是权威

解包在 `ext/`（原始仓库快照），关键文件已复制到 `docs/upstream/`：

| 文件 | 作用 |
| --- | --- |
| `docs/upstream/ImageProcessing.swift` | Otsu 阈值、连通域标注（union-find） |
| `docs/upstream/ThresholdingFilter.coreimage.metal` | 二值化公式（luma 比较） |
| `docs/upstream/ScaleIdentificationViewController.swift` | 自动找 4 个比例尺标记、透视矫正 |
| `docs/upstream/ResultsViewController.swift` | 叶面积 / 被啃食面积统计、像素→cm² 换算 |
| `docs/upstream/ImageUtils.swift` | 缩放、质心、透视变换、flood fill |

测试夹具与真值：`data/paper/images/*.png`（30 张论文原图）、
`data/paper/ground_truth.csv`（论文给出的 total/consumed 面积，cm²，比例尺边长 17 cm）。
上游 `iOS/LeafByteTests/PaperDataTests.swift` 里的 20 条是 App 能自动过的样本，
9 条被注释掉（`app_test_disabled`）——**这些正是"自动失败、需要人工"的真实案例**，
是我们审核队列的天然验证集。

## 3. 上游算法（必须严格对齐的部分）

### 3.1 归一化
- 载入图片，按 EXIF 摆正。
- 等比缩放，最长边 ≤ **1200**，**不放大**；新尺寸 = `floor(原尺寸 * ratio)`。
- 论文图 900×1200 / 1200×900 因此**不缩放**。

### 3.2 luma 与 Otsu
- `luma = (0.299*R + 0.587*G + 0.114*B) / 255`，输入是**未做色彩饱和处理**的 sRGB。
- 256 桶直方图（桶 = `round(luma*255)`，即按 8bit 值分桶）。
- Otsu：类间方差最大者为 `bestCut`（桶索引），`threshold = bestCut / 255`。
  **平局取桶号更大者**（上游用 `>=` 更新）。
- 二值化（白背景）：`occupied = luma < threshold`（**严格小于**）。
  黑背景模式：`occupied = luma > threshold`。

### 3.3 连通域标注（4 连通）
`labelConnectedComponents` 语义：
- 正标签 = occupied 分量；负标签 = empty 分量；初始 `nextOccupiedLabel=1`，`nextEmptyLabel=-2`。
- 4 连通，只回看 west / north，用并查集合并等价标签。
- **贴边的 empty 像素并入背景类**（`BACKGROUND_LABEL = -1`）。
- 每个标签记 `size`（像素数）和 `member_point`（该分量第一次出现、最上方的点）。
- 合并等价类时：`representative = max(labels)`（背景类因此得 -1），
  `member_point` 取类内**最靠上**（y 最小）的点，`size` 累加，
  `empty_to_occupied` 集合并集。
- **注意本流水线没有用户涂鸦层**，所以 `label_to_size` 只统计 standard 部分。

### 3.4 找比例尺（上游做法，我们要更稳）
上游：`occupied` 分量按 size 降序 → 第 0 大是叶子，第 1..4 大当 4 个标记；
用 `getCentroidOfComponent(minSize=19, maxSize=50000)` 取质心（正方形图片里 4 个角点），
再 `CIPerspectiveCorrection` 四点矫正。

已知缺陷（`app_test_disabled` 的 9 张就这么挂的）：图里有更暗的边框 / 反光 / 污点，
"第 2..5 大"就不是 4 个角点了。**我们的替代方案**（`scale.py` 负责）：

1. 候选标记 = occupied 分量，满足 `scale_mark_min_px ≤ size ≤ scale_mark_max_px`、
   长宽比 ≤ `scale_mark_max_aspect`、不贴边、面积占比 < 2%。
2. 在候选里枚举四边形（数量少时全枚举，多则按面积接近/形状预筛，加超时上限）：
   要求四点凸、近似方形（两对边比例、对角线比例在合理范围）、
   四边长度变异小、面积占图比例合理（0.15~0.95）；
   打分 = 方形度 × 面积占比 × 点数，取最高。
3. 允许退化情形：找到可靠 4 点 → `quad`（放宽闸门命中 → `quad_relaxed`，质量打折）；
   只有 3 个可信标记时用平行四边形补第 4 个角 → `method="quad_3dot"`，
   质量 ×0.6 并加 flag `"4th scale mark inferred"`（置信度因此掉到审核区）；
   连 3 个都没有 → `found=False` + 说明性 flags。
4. 顺序约定：输出 TL, TR, BR, BL（图像坐标，y 向下）。

### 3.5 透视矫正（`scale.py`）
- 输入四点 → 真·单应矩阵（numpy 解 8 参数），把四边形映射到正方形。
- 输出边长 `S = min(1200, floor(min(bbox_w, bbox_h)))`——与上游
  `sizeToAdjustTo` 一致；`scale_px_length = S`。
- **矫正的是原始 RGB 图**（上游矫正的是二值图，会二次量化；我们矫正原图后
  用同一个阈值再二值化，等价且更准）。
- 用双线性插值采样（`numpy` 反向映射；`PIL.Image.transform(QUAD)` 可作对照）。

### 3.6 面积统计（`measure.py`）
在矫正后的 S×S 图上：`binarize` → `label_components` →
- `leaf = 最大 occupied`；`leaf_px = size(leaf)`。
- `consumed = Σ size(empty)`，条件：`empty != BACKGROUND_LABEL` **且**
  它邻接的 occupied 集合与叶子的等价类有交集（即被叶子包住的洞）。
  上游就是这么判定的——**不**要求"完全被叶子包围"，只要邻接叶子。
- 排除区域（人工标注的排除点）对应分量不计入 consumed（本流水线用
  `excluded_mask` 位图表达，可为 None）。
- `total_px = leaf_px + consumed_px`；
  `percent_consumed = consumed_px / total_px * 100`。
- cm² 换算：`units_per_pixel = scale_cm / scale_px_length`，
  `area_cm2 = units_per_pixel**2 * pixels`。

## 4. 目录与文件归属（**不要写别人的文件**）

| 路径 | 负责人 | 内容 |
| --- | --- | --- |
| `src/leafbyte/types.py`, `config.py`, `imageops.py`, `ccl.py`, `batch.py`, `cli.py`, `calibrate.py` | Lead | 契约、IO、二值化、连通域、编排、标定 |
| `src/leafbyte/scale.py`, `tests/test_scale.py` | **scale** | 4 点检测 + 单应矫正 |
| `src/leafbyte/measure.py`, `src/leafbyte/quality.py`, `tests/test_measure.py`, `tests/test_quality.py` | **measure-qc** | 面积统计 + 质量指标/置信度 |
| `src/leafbyte/review.py`, `tests/test_review.py`, `docs/REVIEW_WORKFLOW.md` | **review-ui** | 审核队列 JSON + 离线 HTML 审核页 + 决策合并 |
| `docs/ALGORITHM.md` | Lead | 本文件 |
| `data/`, `out/` | Lead 统筹 | 数据与产物（各人可写 `out/<自己的临时子目录>`） |

需要改契约（`types.py`/`config.py`）就先给 Lead 发消息，别直接改。

## 5. 运行环境（重要）

- Python：`python`
- **只有 numpy 2.3.5 + Pillow 12.3.0**（没有 cv2/scipy/skimage/matplotlib，也不能联网装）。
  连通域、形态学、单应、可视化全部自己用 numpy 写。
- 无 pytest，用标准库 `unittest`。
- 工作目录 ``，Bash 里用 ``。
- 跑测试（在仓库根目录）：

```bash
PY="python"
PYTHONPATH=src "$PY" -m unittest discover -s tests -t . -v
```

- 相对导入一律用绝对包路径 `from leafbyte import ...`（依赖 `PYTHONPATH=src`）。
- 大图处理约 1M 像素，纯 numpy 要保证单张 < 3 s；连通域用"行程编码 + 并查集"向量化实现。

## 6. 模块 API（照此实现，签名不要改）

### 6.1 `imageops.py`（Lead）
```python
load_image(path) -> np.ndarray            # (H,W,3) uint8, RGB, EXIF 已摆正
resize_within(img, max_dim=1200) -> np.ndarray   # 等比、不放大、floor
resize_to(img, size) -> np.ndarray        # size=(w,h)，双线性
luma(img) -> np.ndarray                   # float64 (H,W)，[0,1]
luma_histogram(luma_arr) -> np.ndarray    # int64 (256,)
otsu_threshold(hist) -> float             # [0,1]
binarize(img, threshold, background="white") -> np.ndarray  # bool，(H,W)，True=occupied
warp_quad(img, quad, out_size) -> np.ndarray   # quad=[TL,TR,BR,BL]，输出 (S,S,3) uint8
overlay_png(img, mask, holes, quad, ...) -> bytes  # 可视化辅助（可用）
```

### 6.2 `ccl.py`（Lead）
```python
label_components(mask: np.ndarray) -> ComponentsInfo   # 4 连通，语义见 3.3
```

### 6.3 `scale.py`（scale）
```python
detect(img_normalized, cfg) -> ScaleDetection
# 内部：Otsu -> binarize -> label_components -> 候选 -> 四边形搜索
rectify(img_normalized, detection, cfg) -> RectifiedImage | None
detect_and_rectify(img_normalized, cfg) -> tuple[ScaleDetection, RectifiedImage | None]
```

### 6.4 `measure.py`（measure-qc）
```python
analyze(rect_img, cfg, threshold=None, excluded_mask=None) -> MeasureAnalysis
# MeasureAnalysis dataclass: threshold: float, mask: np.ndarray(bool),
#   components: ComponentsInfo, result: MeasureResult
compute_results(ci, cfg, excluded_mask=None) -> MeasureResult
to_cm2(result, cfg, scale_px_length) -> tuple[float, float, float]  # total, consumed, percent
```

### 6.5 `quality.py`（measure-qc）
```python
evaluate(record: dict, cfg, weights: dict | None = None) -> dict
# 在 batch 生成的 record(dict) 上就地补充/返回：
#   "metrics": {...}, "confidence": float, "status": "auto"|"review"|"failed",
#   "flags": [str], "reason": str
# weights 让 calibrate.py 扫描权重；默认用模块级 WEIGHTS。
```

`ComponentsInfo.empty_to_occupied` 的键和值都是**规范化后的代表标签**，
所以"洞是否邻接叶子"就是 `leaf_label in ci.empty_to_occupied.get(hole_label, set())`。
指标至少包含（都可从 record 里算，不重新读图）：
`scale_quality`、`threshold_quality`（Otsu 类间方差分离度/双峰性）、
`stability`（±δ 阈值下 total_cm2 的相对变化）、`leaf_border_touch`、
`competing_leaf`（第二大 occupied / 最大）、`hole_noise`（近阈值小洞占比）、
`percent_extremes`。置信度 = 各指标加权几何/算术平均，落到 [0,1]。
权重和阈值必须在 `calibrate.py` 里可调（暴露成模块级常量 `WEIGHTS`/`DEFAULTS`）。

### 6.6 `review.py`（review-ui）
```python
build(out_dir, records, cfg) -> Path        # 写 out/review/queue.json + out/review/index.html
apply(records, decisions: dict, cfg) -> list[dict]   # 合并人工决策，返回新 records
```
- 队列只收 `status in {"review","failed"}` 的样本。
- 每个待审样本带预计算的候选（batch 已生成，见 §7 `candidates`）：
  候选 = 若干阈值偏移下的完整测量结果 + 叠加图 PNG 路径。
- `index.html` **完全离线**（图片 base64 内联 / 相对路径），双击即看；
  支持：左右对比原图与叠加图、点选候选、键盘快捷键、标记"弃用-需重拍"、
  导出 `decisions.json`（浏览器下载）。
- `decisions.json` 结构：
```json
{"reviewer": "zs", "reviewed_at": "2026-11-10T12:00:00",
 "decisions": {
   "CO1.png": {"action": "accept", "threshold": 0.53, "total_cm2": 23.95, "consumed_cm2": 1.0, "note": ""},
   "AI2.png": {"action": "override", "total_cm2": 22.565, "consumed_cm2": 2.279, "note": "ImageJ 手测"},
   "CA1.png": {"action": "reject", "note": "反光，需重拍"}
 }}
```
  `action`: `accept`（采纳某个候选，用它的 threshold/数值）、
  `override`（人工直接给数值）、`reject`（作废）。
- `apply` 必须写出**审计信息**：`source: "auto"|"review:<reviewer>"`、原始置信度、原阈值。

## 7. 分析记录 schema（batch → quality → review 的唯一数据形态）

`out/results.jsonl` 每行一个 dict：

```json
{
  "file": "CO1.png", "path": "...", "ok": true, "stage": "done",
  "image_size": [900, 1200], "normalized_size": [900, 1200],
  "threshold": 0.53, "otsu_threshold": 0.55,
  "threshold_metrics": {"separation": 0.71, "near_threshold_fraction": 0.004, "bimodality": 0.62},
  "scale": {"found": true, "method": "quad", "quality": 0.92,
            "points": [[x,y],[x,y],[x,y],[x,y]],
            "scale_px_length": 806, "flags": [], "details": {"side_ratios": [...]}},
  "measure": {"total_px": 53400, "consumed_px": 2210, "leaf_px": 51190,
              "holes": [{"label": -3, "area_px": 2210, "centroid": [x, y]}],
              "flags": []},
  "total_cm2": 23.951, "consumed_cm2": 0.992, "percent_consumed": 4.141,
  "stability": {"deltas": [-0.02, 0.02], "total_cm2": [23.9, 24.0], "rel_range": 0.004},
  "candidates": [{"threshold": 0.51, "total_cm2": 23.8, "consumed_cm2": 0.9,
                  "percent_consumed": 3.8, "overlay": "review/candidates/CO1__t0.510.png"}],
  "metrics": {}, "confidence": 0.0, "status": "auto", "flags": [],
  "timing_ms": 1234, "error": null
}
```

数值统一保留 6 位有效数字写入 JSON；CSV 保留 3 位小数。

## 8. 验收标准

1. `tests/` 全绿（`PYTHONPATH=src python -m unittest discover -s tests -t .`）。
2. 复现上游 iOS 单测数值（Lead 在 `tests/test_ccl.py` 里把关）：
   - `leafWithScale.png` Otsu 阈值 = 139/255；
   - 900×1200 归一化后：empty 分量 size 末两位 = [3358, 970002]，
     occupied 末两位 = [1178, 105400]。
3. 论文集 30 张全部跑通（允许 `status=review`），其中
   `app_test` 的 20 张里 ≥ 18 张被判 `auto`，且 |Δtotal| ≤ 0.1 cm² 或 ≤1.5%；
   `app_test_disabled` 的 9 张里 ≥ 7 张被正确转入 `review`（不能静默给错值）。
4. 人工审核闭环可跑：`run → review → (人工改 decisions.json) → apply → final_results.csv`。
5. 无网络、无第三方依赖（numpy+Pillow 以外）。

## 9. 沟通

- 完成/卡住/需要改契约，`send_message` 给 `lead`，一句话说明状态 + 需要的动作。
- 不要动别人的文件；不要 `git` 操作；不要改 `data/`。
- 你的实现要能被别人在不读你代码的情况下调用：docstring 写清输入输出。
