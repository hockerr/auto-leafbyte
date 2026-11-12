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
├── calibration.json          置信度门槛标定曲线
├── FINAL_REPORT.md           人可以读的最终报告（含审核明细）
├── overlays/                 可视化叠加图：红=叶片、蓝=孔洞、黄=比例尺四点
│   ├── CO1_rectified.jpg     （本仓库只为 CO1 / SH2 保留示例图）
│   └── SH2_rectified.jpg
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
