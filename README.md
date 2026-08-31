# Student Success & Placement Readiness Predictor

An end-to-end Data Science and Machine Learning project that estimates whether a student is placement-ready from academic, technical, and experiential indicators.

> **Important:** The included dataset is deterministic synthetic data. It is for demonstrating the data-science pipeline and must not be presented as real student or placement records.

## Problem statement

Career cells often need an early, explainable signal of where students may benefit from additional support. This project explores factors such as attendance, CGPA, aptitude, coding score, projects, internships, communication score, and study hours; it uses logistic regression to estimate readiness.

The output is intended to support mentoring conversations, never make an automatic hiring or academic decision.

## Features

- Data generation and CSV ingestion
- Data validation and exploratory group-level analysis
- Stratified train/test split
- Logistic regression built from first principles
- Accuracy, precision, recall, F1 score, and confusion-matrix values
- Feature-weight explanation
- Individual student prediction through the command line
- A shareable static HTML analysis report

## Run it

Requires Python 3.10+; no package installation is needed.

```sh
python3 student_success_ml.py bootstrap-data
python3 student_success_ml.py analyze
python3 student_success_ml.py train
python3 student_success_ml.py generate-report
```

Open `artifacts/analysis_report.html` in a browser after generating it.

Example individual prediction:

```sh
python3 student_success_ml.py predict --attendance 88 --cgpa 8.1 --aptitude 76 --communication 72 --projects 2 --internships 1 --coding 82 --study-hours 4
```

## Method

1. Validate numeric values and binary target labels in the CSV.
2. Stratify records into an 80/20 train/test split to preserve target balance.
3. Standardize features using statistics calculated from training data only.
4. Train a logistic-regression classifier with batch gradient descent.
5. Evaluate on the held-out test set and rank standardized model weights by magnitude.

## Dataset schema

| Column | Meaning |
| --- | --- |
| `attendance` | Attendance percentage |
| `cgpa` | Cumulative grade point average |
| `aptitude`, `coding`, `communication` | Assessment scores out of 100 |
| `projects`, `internships` | Completed count |
| `study_hours` | Typical daily study hours |
| `placement_ready` | Demonstration target: 1 = ready, 0 = needs support |

## Responsible use

This is an educational decision-support prototype. Before using real data, obtain consent, check for bias and missing groups, secure the data, validate with domain experts, and ensure a human remains responsible for decisions.

## Suggested form entries

**Project Name:** Student Success and Placement Readiness Predictor Using Machine Learning

**Short Summary:** An end-to-end machine-learning project that analyzes academic, technical, and experiential factors to estimate student placement readiness. It includes CSV data validation, exploratory analysis, logistic-regression training, accuracy/precision/recall/F1 evaluation, interpretable feature importance, individual prediction, and an automatically generated HTML report.
