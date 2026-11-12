"""Turn a finished run folder into a Chinese summary table + notes.

    python tools/summarize_run.py --out out/yanfumu
    python tools/summarize_run.py --out out/yanfumu --title "盐肤木 140 号样本"

Writes ``叶片测量结果.csv`` (UTF-8 with BOM, opens in Excel) and ``说明.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leafbyte.batch import natural_key  # noqa: E402

STATUS_ZH = {"auto": "自动通过", "review": "待人工复核", "failed": "失败",
             "done": "人工已确认", "rejected": "已弃用"}
FLAG_ZH = {
    "area_unstable": "面积对分割阈值敏感（边缘偏软/虚焦）",
    "tiny_hole_noise": "啃食孔洞只有噪点级小点",
    "scale_not_found": "未找到比例尺",
    "scale_invalid_length": "比例尺长度异常",
    "competing_leaf": "存在与叶片竞争的大斑块",
    "leaf_touches_border": "叶片被画幅裁切",
    "relative_only": "只有相对比例，无绝对面积",
    "percent_implausible": "取食比例不合理",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="run folder containing results.jsonl")
    parser.add_argument("--title", default=None, help="heading for the notes file")
    args = parser.parse_args(argv)

    out = Path(args.out)
    records = [json.loads(line) for line in
               (out / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda record: natural_key(record["file"]))
    title = args.title or out.name

    csv_path = out / "叶片测量结果.csv"
    if csv_path.exists():
        try:
            csv_path.unlink()
        except PermissionError:
            csv_path = out / "叶片测量结果_新.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["序号", "文件名", "状态", "置信度", "叶面积_cm2", "取食面积_cm2",
                         "取食率_%", "完整叶面积_cm2", "叶周长_cm", "叶长_cm", "叶宽_cm",
                         "长宽比", "圆度", "凸度", "叶色GCC", "比例尺边长_px", "分割方式",
                         "内部暗斑(像素)", "阈值图", "轮廓SVG", "轮廓PDF", "图版PNG", "说明"])
        for index, record in enumerate(records, start=1):
            measure = record.get("measure") or {}
            segmentation = record.get("segmentation") or {}
            flags = "；".join(FLAG_ZH.get(flag, flag) for flag in (record.get("flags") or []))
            writer.writerow([
                index, record["file"], STATUS_ZH.get(record["status"], record["status"]),
                round(float(record.get("confidence") or 0), 3),
                round(record["leaf_area_cm2"], 3) if record.get("leaf_area_cm2") is not None
                else (round(record["total_cm2"], 3) if record.get("total_cm2") is not None else ""),
                round(record["consumed_cm2"], 3) if record.get("consumed_cm2") is not None else "",
                round(record["percent_consumed"], 2) if record.get("percent_consumed") is not None else "",
                (record.get("traits") or {}).get("intact_area_cm2"),
                (record.get("traits") or {}).get("perimeter_cm"),
                (record.get("traits") or {}).get("length_cm"),
                (record.get("traits") or {}).get("width_cm"),
                (record.get("traits") or {}).get("aspect_ratio"),
                (record.get("traits") or {}).get("circularity"),
                (record.get("traits") or {}).get("solidity"),
                (record.get("traits") or {}).get("gcc"),
                int((record.get("scale") or {}).get("scale_px_length") or 0),
                "彩色(ExG)" if measure.get("mode") == "color" else "亮度",
                measure.get("filled_dark_px") or 0,
                record.get("figure_threshold") or "",
                record.get("figure_contour") or "",
                record.get("figure_pdf") or "",
                record.get("figure_plate") or "",
                flags or "测量正常",
            ])

    auto = [r for r in records if r.get("status") == "auto"]
    review = [r for r in records if r.get("status") in {"review", "failed"}]
    areas = [r["total_cm2"] for r in auto if r.get("total_cm2") is not None]
    pcts = [r["percent_consumed"] for r in records if r.get("percent_consumed") is not None]
    scale_cm = 17.0
    config_path = out / "config.json"
    if config_path.exists():
        try:
            scale_cm = float(json.loads(config_path.read_text(encoding="utf-8")).get("scale_cm", 17.0))
        except Exception:  # noqa: BLE001
            pass

    lines = [f"# {title} 测量说明\n",
             f"- 样本数：{len(records)}（自动通过 {len(auto)}，待人工复核 {len(review)}）",
             f"- 比例尺：四角圆点方框，边长按 **{scale_cm:g} cm** 计算"]
    if areas:
        lines.append(f"- 自动通过样本叶面积：{min(areas):.2f} ~ {max(areas):.2f} cm²，"
                     f"均值 {sum(areas)/len(areas):.2f} cm²")
    if pcts:
        lines.append(f"- 取食率：0 ~ {max(pcts):.2f}%（均值 {sum(pcts)/len(pcts):.2f}%）")
    modes = sorted({(r.get("measure") or {}).get("mode") for r in records if r.get("measure")})
    lines.append(f"- 分割方式：{', '.join(m for m in modes if m)}（自适应：白纸走亮度 Otsu，灰背景走 ExG 颜色）\n")

    if review:
        lines.append("## 待人工复核\n")
        for record in review:
            flags = "；".join(FLAG_ZH.get(flag, flag) for flag in (record.get("flags") or [])) or "置信度低于门槛"
            area = record.get("total_cm2")
            area_text = f"{float(area):.2f} cm²" if area is not None else "未测出面积"
            confidence = record.get("confidence")
            conf_text = f"{float(confidence):.2f}" if confidence is not None else "-"
            lines.append(f"- `{record['file']}`：叶面积 {area_text}，置信度 {conf_text}；{flags}")
        lines.append("")
    else:
        lines.append("## 待人工复核\n\n无 —— 全部样本置信度达到自动接受门槛。\n")

    figures = [r for r in records if r.get("figure_threshold")]
    if figures:
        lines += ["## 阈值图\n",
                  "每张样本的「原图 / 叶片掩膜 / 阈值直方图」三联图保存在 `figures/` 下，"
                  "例如：",
                  f"```\n{figures[0]['figure_threshold']}\n```\n"]
    lines += [
        "## 复核流程\n",
        "双击 `review/index.html`，红色 = 自动识别的叶片，可与原图对照；确认后导出 decisions.json，再执行：",
        "```",
        f"PYTHONPATH=src python -m leafbyte apply --out {out.as_posix()} --decisions <decisions.json>",
        "```\n",
        "## 说明\n",
        "1. 本算法统计**叶片内部**的啃食孔洞；叶缘缺刻不在自动测量范围内。",
        "2. 面积单位换算：`cm² = 像素 × (比例尺边长 / 校正后图像边长)²`，比例尺边长默认 17 cm。",
        "3. 置信度 < `auto_accept_min`（默认 0.92）的样本进入人工审核队列；这是有意设计，不是失败。\n",
    ]
    (out / "说明.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"written: {csv_path}")
    print(f"written: {out / '说明.md'}")
    print(f"auto {len(auto)} / review {len(review)} / total {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
