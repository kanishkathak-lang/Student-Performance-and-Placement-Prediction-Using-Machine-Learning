#!/usr/bin/env python3
"""Student Success Predictor: an end-to-end ML project using Python's stdlib only."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FEATURES = ("attendance", "cgpa", "aptitude", "communication", "projects", "internships", "coding", "study_hours")
LABEL = "placement_ready"
SEED = 2026


@dataclass
class Model:
    means: list[float]
    scales: list[float]
    weights: list[float]

    def probability(self, row: dict[str, float]) -> float:
        z = self.weights[0]
        for index, name in enumerate(FEATURES):
            z += self.weights[index + 1] * ((row[name] - self.means[index]) / self.scales[index])
        z = max(-35, min(35, z))
        return 1 / (1 + math.exp(-z))


def create_demo_data(path: Path, count: int = 320) -> None:
    """Create a deterministic, explicitly synthetic dataset for demonstration."""
    rng = random.Random(SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("student_id",) + FEATURES + (LABEL,))
        writer.writeheader()
        for student in range(1, count + 1):
            attendance = round(rng.uniform(52, 100), 1)
            cgpa = round(rng.uniform(5.5, 10.0), 2)
            aptitude = round(rng.uniform(30, 100), 1)
            communication = round(rng.uniform(35, 100), 1)
            projects = rng.choices([0, 1, 2, 3, 4], weights=[8, 25, 34, 23, 10])[0]
            internships = rng.choices([0, 1, 2], weights=[54, 38, 8])[0]
            coding = round(rng.uniform(25, 100), 1)
            study_hours = round(rng.uniform(1, 8), 1)
            score = (-18.0 + 0.055 * attendance + 0.82 * cgpa + 0.030 * aptitude +
                     0.025 * communication + 0.52 * projects + 0.95 * internships +
                     0.034 * coding + 0.11 * study_hours + rng.gauss(0, 1.15))
            probability = 1 / (1 + math.exp(-score))
            ready = int(rng.random() < probability)
            writer.writerow({"student_id": f"STU{student:03d}", "attendance": attendance, "cgpa": cgpa,
                             "aptitude": aptitude, "communication": communication, "projects": projects,
                             "internships": internships, "coding": coding, "study_hours": study_hours,
                             LABEL: ready})


def read_data(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise ValueError(f"dataset not found: {path}. Run bootstrap-data first.")
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = set(FEATURES) | {LABEL}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError("CSV must contain " + ", ".join((*FEATURES, LABEL)))
        for line, raw in enumerate(reader, 2):
            try:
                row = {name: float(raw[name]) for name in FEATURES}
                row[LABEL] = float(int(raw[LABEL]))
                if row[LABEL] not in (0, 1): raise ValueError("label must be 0 or 1")
                if not all(math.isfinite(value) for value in row.values()): raise ValueError("non-finite number")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid data at CSV line {line}: {exc}") from exc
            rows.append(row)
    if len(rows) < 30: raise ValueError("at least 30 rows are required")
    return rows


def split(rows: list[dict[str, float]], ratio: float = .8) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    groups = {0: [], 1: []}
    for row in rows: groups[int(row[LABEL])].append(row)
    rng = random.Random(SEED)
    train, test = [], []
    for group in groups.values():
        rng.shuffle(group)
        boundary = max(1, int(len(group) * ratio))
        train.extend(group[:boundary]); test.extend(group[boundary:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test


def train(rows: list[dict[str, float]], epochs: int = 1800, learning_rate: float = .075) -> Model:
    means = [statistics.fmean(row[name] for row in rows) for name in FEATURES]
    scales = [max(statistics.pstdev(row[name] for row in rows), 1e-8) for name in FEATURES]
    weights = [0.0] * (len(FEATURES) + 1)
    for _ in range(epochs):
        gradient = [0.0] * len(weights)
        for row in rows:
            values = [(row[name] - means[i]) / scales[i] for i, name in enumerate(FEATURES)]
            z = weights[0] + sum(weight * value for weight, value in zip(weights[1:], values))
            probability = 1 / (1 + math.exp(-max(-35, min(35, z))))
            error = probability - row[LABEL]
            gradient[0] += error
            for i, value in enumerate(values): gradient[i + 1] += error * value
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradient[i] / len(rows)
    return Model(means, scales, weights)


def evaluate(model: Model, rows: Iterable[dict[str, float]]) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in rows:
        predicted = int(model.probability(row) >= .5)
        actual = int(row[LABEL])
        if predicted and actual: tp += 1
        elif predicted and not actual: fp += 1
        elif not predicted and actual: fn += 1
        else: tn += 1
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"samples": total, "accuracy": round((tp + tn) / total, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0,
            "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}


def importance(model: Model) -> list[tuple[str, float]]:
    return sorted(((name, weight) for name, weight in zip(FEATURES, model.weights[1:])), key=lambda item: abs(item[1]), reverse=True)


def summary(rows: list[dict[str, float]]) -> dict[str, object]:
    ready = [row for row in rows if row[LABEL] == 1]
    not_ready = [row for row in rows if row[LABEL] == 0]
    return {"students": len(rows), "placement_ready": len(ready), "not_ready": len(not_ready),
            "ready_rate": round(100 * len(ready) / len(rows), 1),
            "mean_by_group": {name: {"ready": round(statistics.fmean(r[name] for r in ready), 2),
                                      "not_ready": round(statistics.fmean(r[name] for r in not_ready), 2)} for name in FEATURES}}


def report_html(stats: dict[str, object], metrics: dict[str, float | int], ranked: list[tuple[str, float]]) -> str:
    bars = "".join(f'<div class="bar"><span>{html.escape(name.replace("_", " ").title())}</span><i style="width:{min(100, abs(weight)*55):.1f}%"></i><b>{weight:+.3f}</b></div>' for name, weight in ranked)
    metric_cards = "".join(f'<article><strong>{str(key).replace("_", " ").title()}</strong><em>{value}</em></article>' for key, value in metrics.items() if key in {"accuracy", "precision", "recall", "f1", "samples"})
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Student Success Predictor</title><style>
body{{font:16px system-ui,sans-serif;background:#f5f7fb;color:#172033;max-width:960px;margin:40px auto;padding:0 20px}}h1{{margin-bottom:4px}}.sub{{color:#52627b}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:22px 0}}article{{background:white;border-radius:12px;padding:16px;box-shadow:0 2px 10px #dfe5f0}}strong,em{{display:block}}em{{font-size:28px;font-style:normal;font-weight:700;color:#2855c5;margin-top:8px}}section{{background:white;border-radius:12px;padding:22px;margin-top:16px;box-shadow:0 2px 10px #dfe5f0}}.bar{{display:grid;grid-template-columns:180px 1fr 70px;gap:10px;align-items:center;margin:11px 0}}i{{height:16px;border-radius:9px;background:#4d7cff}}b{{font-variant-numeric:tabular-nums}}footer{{color:#62708a;margin:28px 0;font-size:13px}}</style></head><body>
<h1>Student Success & Placement Readiness Predictor</h1><p class=\"sub\">Exploratory analysis and custom logistic-regression evaluation</p>
<div class=\"grid\"><article><strong>Dataset</strong><em>{stats['students']} students</em></article><article><strong>Ready rate</strong><em>{stats['ready_rate']}%</em></article>{metric_cards}</div>
<section><h2>Most influential model features</h2>{bars}<p>Positive weights increase estimated placement readiness; magnitudes are on standardized features.</p></section>
<section><h2>Responsible-use note</h2><p>This bundled dataset is synthetic and exists only to demonstrate the pipeline. A real deployment must use consented, representative data and should support—not replace—human academic or career guidance.</p></section>
<footer>Generated locally with Python standard library only.</footer></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Student placement-readiness ML pipeline")
    parser.add_argument("--data", default="data/student_records.csv", help="CSV dataset path")
    commands = parser.add_subparsers(dest="command", required=True)
    boot = commands.add_parser("bootstrap-data", help="create deterministic synthetic demo data"); boot.add_argument("--count", type=int, default=320)
    commands.add_parser("analyze", help="print dataset summary")
    commands.add_parser("train", help="train and evaluate the model")
    predict = commands.add_parser("predict", help="estimate readiness for one student")
    for name in FEATURES: predict.add_argument("--" + name.replace("_", "-"), type=float, required=True)
    report = commands.add_parser("generate-report", help="write a static HTML analysis report"); report.add_argument("--output", default="artifacts/analysis_report.html")
    args = parser.parse_args(); data_path = Path(args.data)
    if args.command == "bootstrap-data":
        create_demo_data(data_path, args.count); print(f"Created {args.count} synthetic records at {data_path}"); return 0
    try:
        rows = read_data(data_path)
        if args.command == "analyze": print(json.dumps(summary(rows), indent=2)); return 0
        train_rows, test_rows = split(rows); model = train(train_rows); metrics = evaluate(model, test_rows); ranked = importance(model)
        if args.command == "train": print(json.dumps({"test_metrics": metrics, "feature_importance": ranked}, indent=2)); return 0
        if args.command == "predict":
            row = {name: getattr(args, name) for name in FEATURES}; chance = model.probability(row)
            print(json.dumps({"placement_readiness_probability": round(chance, 4), "classification": "ready" if chance >= .5 else "needs_support", "threshold": .5}, indent=2)); return 0
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(report_html(summary(rows), metrics, ranked), encoding="utf-8"); print(f"Report written to {output}")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
