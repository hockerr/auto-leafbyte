# 输出示例 / Example outputs

仓库里带一份用**上游 LeafByte 论文的 30 张测试照片**跑出来的完整示例
（`out/paper/`），方便你直接看到每类文件长什么样。论文照片本身不随仓库分发，
用 `python tools/fetch_paper_data.py` 下载到 `data/paper/images/` 后可以完全复现：

```bash
export PYTHONPATH=src
python tools/fetch_paper_data.py
python -m leafbyte run data/paper/images --out out/paper --scale-cm 17 --overlays review
```

## 一个 run 目录里有什么

```
out/paper/
├── results.csv               主表（状态 / 置信度 / 面积 / 取食率 / 标志）
├── results.jsonl             全量中间量（阈值、比例尺四点、稳定性、候选、性状）
├── leaf_traits.csv           22 个叶片性状（周长/长叶宽/圆度/凸度/叶色…）
├── config.json               本次运行的完整配置（可原样重放）
├── run.json                  汇总、耗时、与真值的误差统计
├── overlays/                 可视化叠加图：红=叶片、蓝=孔洞、黄=比例尺四点
│   └── *_rectified.jpg       run 时只为**待审核**样本生成（--overlays all 则每张都有）；
│                             本仓库只保留 CO1 / SH2 的示例
├── review/decisions.applied.json   （可选）人工审核后归档的决策文件
├── calibration.json           （可选）跑 `leafbyte calibrate` 才生成
└── FINAL_REPORT.md            （可选）`leafbyte report`/人工审核流程产出的可读报告
├── figures/                  每张样本 4 个文件（示例只保留 CO1、SH2）
│   ├── CO1_..._threshold.png  阈值图：矫正图+绿线叶缘 / 掩膜 / 直方图+阈值线
│   ├── CO1_..._mask.png       全分辨率二值掩膜（面积按它计算）
│   ├── CO1_..._contour.svg    矢量轮廓（cm 单位、比例条、样本编号）
│   └── CO1_..._plate.jpg      图版：叶片填充+轮廓+比例条+编号
└── review/
    ├── index.html            离线人工审核页（双击打开）
    └── queue.json            待审队列（含每个候选阈值的结果）
```

## 汇总表

多个 run 合并成一张统计表：

```bash
python tools/make_summary_table.py --run out/paper --out out/汇总
```

得到 `叶片性状汇总.xlsx`（样本数据 / 性状统计 / 说明 三个工作表）和同名 CSV。

> 说明：本仓库的示例 run 只保留了 CO1 / SH2 的图文件，因此 `results.jsonl` 里
> 其他样本的 `figure_*` 路径指向本地未包含的文件；在你自己的完整 run 里它们都存在。
