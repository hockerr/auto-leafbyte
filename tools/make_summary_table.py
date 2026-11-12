"""Build one summary data table (CSV + optional Excel) from finished runs.

    python tools/make_summary_table.py --run out/yanfumu --run out/mujing --out out/汇总
    python tools/make_summary_table.py --run out/yanfumu --sheets-only

Outputs (in ``--out``):

* ``叶片性状汇总.xlsx``  - three sheets: 样本数据 / 性状统计 / 说明
* ``叶片性状汇总.csv``   - the same per-sample table, flat (UTF-8 with BOM)

Excel needs ``openpyxl``; the CSV is always written, so the tool works with a
bare numpy/Pillow install and you can open the CSV straight in Excel.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leafbyte import traits  # noqa: E402
from leafbyte.batch import natural_key  # noqa: E402

STATUS_ZH = {"auto": "自动通过", "review": "待人工复核", "failed": "失败",
             "done": "人工已确认", "rejected": "已弃用"}
FLAG_ZH = {
    "area_unstable": "面积对分割阈值敏感",
    "tiny_hole_noise": "孔洞仅噪点级",
    "scale_not_found": "未找到比例尺",
    "scale_invalid_length": "比例尺长度异常",
    "competing_leaf": "存在竞争斑块",
    "leaf_touches_border": "叶片被画幅裁切",
    "relative_only": "仅相对比例",
    "dark_patches_filled": "内部暗斑已填回叶片",
    "percent_implausible": "取食比例异常",
    "weak_threshold_separation": "阈值分离度弱",
}

#: (csv column, chinese header, getter) - keep the order stable for downstream tools
SAMPLE_COLUMNS: list[tuple[str, str]] = [
    ("index", "序号"),
    ("dataset", "数据集"),
    ("file", "文件名"),
    ("status", "状态"),
    ("confidence", "置信度"),
    ("total_cm2", "叶面积_cm2"),
    ("intact_cm2", "完整叶面积_cm2"),
    ("consumed_cm2", "取食面积_cm2"),
    ("percent_consumed", "取食率_%"),
    ("perimeter_cm", "叶周长_cm"),
    ("length_cm", "叶长_cm"),
    ("width_cm", "叶宽_cm"),
    ("aspect_ratio", "长宽比"),
    ("circularity", "圆度"),
    ("roundness", "圆整度"),
    ("solidity", "凸度"),
    ("rectangularity", "矩形度"),
    ("dissection_index", "叶缘分裂指数"),
    ("max_feret_cm", "最大卡径_cm"),
    ("min_feret_cm", "最小卡径_cm"),
    ("convex_area_cm2", "凸包面积_cm2"),
    ("equiv_diameter_cm", "等效直径_cm"),
    ("mean_r", "平均R"),
    ("mean_g", "平均G"),
    ("mean_b", "平均B"),
    ("gcc", "GCC"),
    ("exg", "ExG"),
    ("hole_count", "孔洞数"),
    ("mean_hole_cm2", "平均孔洞面积_cm2"),
    ("filled_dark_px", "内部暗斑_px"),
    ("mode", "分割方式"),
    ("scale_px", "比例尺边长_px"),
    ("flags_zh", "标志"),
    ("note", "备注"),
]

#: traits summarised in the statistics sheet (chinese label)
STAT_COLUMNS = [
    ("total_cm2", "叶面积_cm2"), ("intact_cm2", "完整叶面积_cm2"),
    ("consumed_cm2", "取食面积_cm2"), ("percent_consumed", "取食率_%"),
    ("perimeter_cm", "叶周长_cm"), ("length_cm", "叶长_cm"), ("width_cm", "叶宽_cm"),
    ("aspect_ratio", "长宽比"), ("circularity", "圆度"), ("solidity", "凸度"),
    ("rectangularity", "矩形度"), ("dissection_index", "叶缘分裂指数"),
    ("max_feret_cm", "最大卡径_cm"), ("min_feret_cm", "最小卡径_cm"),
    ("convex_area_cm2", "凸包面积_cm2"), ("equiv_diameter_cm", "等效直径_cm"),
    ("gcc", "GCC"), ("exg", "ExG"), ("hole_count", "孔洞数"),
]


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _dataset_name(filename: str, fallback: str) -> str:
    for species in ("盐肤木", "牡荆", "女贞", "樟", "枫", "杨", "柳"):
        if species in filename:
            return species
    return fallback


def load_run(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"no results.jsonl in {run_dir}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    records.sort(key=lambda record: natural_key(record["file"]))
    return records


def sample_row(record: dict, index: int, dataset: str) -> dict:
    measure = record.get("measure") or {}
    trait_values = record.get("traits") or {}
    scale = record.get("scale") or {}
    flags = record.get("flags") or []
    return {
        "index": index,
        "dataset": dataset,
        "file": record["file"],
        "status": STATUS_ZH.get(record.get("status"), record.get("status")),
        "status_raw": record.get("status"),
        "confidence": _num(record.get("confidence")),
        "total_cm2": _num(record.get("total_cm2")),
        "intact_cm2": _num(trait_values.get("intact_area_cm2")),
        "consumed_cm2": _num(record.get("consumed_cm2")),
        "percent_consumed": _num(record.get("percent_consumed")),
        "filled_dark_px": measure.get("filled_dark_px") or 0,
        "mode": "彩色(ExG)" if measure.get("mode") == "color" else "亮度",
        "scale_px": int(scale.get("scale_px_length") or 0),
        "flags_zh": "；".join(FLAG_ZH.get(flag, flag) for flag in flags),
        "note": record.get("reason") or "",
        **{key: _num(trait_values.get(key)) for key in traits.TRAIT_COLUMNS},
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([label for _, label in SAMPLE_COLUMNS])
        for row in rows:
            writer.writerow([row.get(key) for key, _ in SAMPLE_COLUMNS])


def statistics_rows(rows: list[dict], subset_label: str) -> list[list]:
    out: list[list] = []
    for key, label in STAT_COLUMNS:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            out.append([subset_label, label, 0, None, None, None, None, None])
            continue
        out.append([
            subset_label, label, len(values),
            round(statistics.mean(values), 4),
            round(statistics.stdev(values), 4) if len(values) > 1 else None,
            round(min(values), 4),
            round(statistics.median(values), 4),
            round(max(values), 4),
        ])
    return out


def write_xlsx(rows: list[dict], datasets: list[tuple[str, list[dict]]], path: Path,
               meta: dict) -> str:
    """Write the workbook; returns "" on success or a reason it was skipped."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except Exception:  # noqa: BLE001
        return "openpyxl not installed"

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "样本数据"
    header_fill = PatternFill("solid", fgColor="1F4E3D")
    header_font = Font(bold=True, color="FFFFFF")
    for column, (_, label) in enumerate(SAMPLE_COLUMNS, start=1):
        cell = sheet.cell(row=1, column=column, value=label)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row_index, row in enumerate(rows, start=2):
        for column, (key, _) in enumerate(SAMPLE_COLUMNS, start=1):
            value = row.get(key)
            cell = sheet.cell(row=row_index, column=column, value=value)
            if isinstance(value, float):
                cell.number_format = "0.000" if abs(value) >= 0.01 else "0.0000"
    widths = {"file": 46, "note": 40, "flags_zh": 26, "dataset": 10, "status": 12}
    for column, (key, label) in enumerate(SAMPLE_COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(column)].width = widths.get(
            key, max(9, len(label) * 1.6))
    sheet.freeze_panes = "D2"
    sheet.auto_filter.ref = sheet.dimensions

    stats = workbook.create_sheet("性状统计")
    stats.append(["子集", "指标", "n", "均值", "标准差", "最小值", "中位数", "最大值"])
    for cell in stats[1]:
        cell.fill = header_fill
        cell.font = header_font
    for subset_label, subset_rows in [("全部样本", rows)] + datasets:
        for line in statistics_rows(subset_rows, subset_label):
            stats.append(line)
    for column in range(1, 9):
        stats.column_dimensions[get_column_letter(column)].width = (14 if column > 2 else 12)
    stats.freeze_panes = "A2"

    notes = workbook.create_sheet("说明")
    notes.column_dimensions["A"].width = 110
    for row_index, line in enumerate(meta["notes"], start=1):
        cell = notes.cell(row=row_index, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if row_index == 1:
            cell.font = Font(bold=True, size=12)

    workbook.save(path)
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True,
                        help="run folder (repeat for several datasets)")
    parser.add_argument("--out", default=str(ROOT / "out" / "汇总"), help="output folder")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    datasets: list[tuple[str, list[dict]]] = []
    index = 0
    for run in args.run:
        run_dir = Path(run)
        records = load_run(run_dir)
        subset: list[dict] = []
        for record in records:
            index += 1
            row = sample_row(record, index, _dataset_name(record["file"], run_dir.name))
            rows.append(row)
            subset.append(row)
        label = subset[0]["dataset"] if subset else run_dir.name
        datasets.append((f"{label}（{len(subset)} 张）", subset))

    csv_path = out_dir / "叶片性状汇总.csv"
    write_csv(rows, csv_path)

    auto_rows = [row for row in rows if row.get("status_raw") == "auto"]
    notes = [
        f"叶片性状汇总（{time.strftime('%Y-%m-%d %H:%M')}）",
        "",
        f"样本数：{len(rows)}（自动通过 {len(auto_rows)}，待人工复核 "
        f"{sum(1 for row in rows if row.get('status_raw') == 'review')}）",
        "比例尺：四角圆点方框，边长按 17 cm 计算；面积 = 像素 × (17 / 校正图像边长)²。",
        "叶面积 = 剩余叶片组织（不含内部孔洞）；完整叶面积 = 剩余 + 被取食孔洞；"
        "取食率 = 取食面积 / 完整叶面积 × 100%。",
        "内部暗斑_px：颜色明显比纸暗的内部斑块（病斑/褶皱阴影），已填回叶片、不计入取食。",
        "状态：自动通过 = 置信度达到门槛；待人工复核 = 面积对阈值敏感 / 比例尺可疑等。",
        "性状定义见 README.md 与 docs/ALGORITHM.md；干重相关 SLA/LMA 需另用 "
        "tools/attach_dry_mass.py 补充。",
        "每个样本的阈值图 / 矢量轮廓 / 图版在各自 run 的 figures/ 目录下。",
    ]
    xlsx_path = out_dir / "叶片性状汇总.xlsx"
    skipped = write_xlsx(rows, datasets, xlsx_path, {"notes": notes})

    print(f"rows    : {len(rows)}")
    print(f"csv     : {csv_path}")
    if skipped:
        print(f"xlsx    : skipped ({skipped}); install with: python -m pip install openpyxl")
    else:
        print(f"xlsx    : {xlsx_path}")
        print(f"sheets  : 样本数据 / 性状统计 / 说明")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
