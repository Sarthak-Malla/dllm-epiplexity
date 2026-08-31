"""Analyze GSM8K and HumanEval path-selection ablations.

Run from any directory with:
	python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation_study.py [dataset]

	where dataset is either 'gsm8k' (default) or 'humaneval'
"""

from itertools import combinations
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/sarthak.malla/dllm-selection-ensemble")
RESULTS_ROOT = ROOT / "eval_results/path_selection"
METHODS = ["greedy", "entropy_drop", "risk_reduction"]

# Dataset configuration
DATASET = sys.argv[1].lower() if len(sys.argv) > 1 else "gsm8k"
if DATASET not in {"gsm8k", "humaneval"}:
	raise ValueError(f"Dataset must be 'gsm8k' or 'humaneval', got '{DATASET}'")

DATASET_CONFIG = {
	"gsm8k": {
		"prefix": "gsm8k_full",
		"sample_pattern": "samples_gsm8k_cot_*.jsonl",
		"expected_count": 1319,
		"filters": {"strict-match", "flexible-extract"},
		"correctness_key": "exact_match",
	},
	"humaneval": {
		"prefix": "humaneval_full",
		"sample_pattern": "samples_humaneval_instruct_*.jsonl",
		"expected_count": 164,
		"filters": {"create_test"},
		"correctness_key": "pass@1",
	},
}


def normalize_string(value):
	value = str(value).lower().strip()
	value = value.replace(",", "").replace("$", "")
	value = re.sub(r"(?s).*#### ", "", value)
	return value.removesuffix(".").strip()


def numeric_value(value):
	value = normalize_string(value)
	try:
		return Decimal(value)
	except (InvalidOperation, ValueError):
		return None


def same_answer(left, right, numeric=False):
	if numeric:
		left_number = numeric_value(left)
		right_number = numeric_value(right)
		return left_number is not None and left_number == right_number
	return normalize_string(left) == normalize_string(right)


def majority_value(values, numeric=False):
	normalized = [numeric_value(value) if numeric else normalize_string(value) for value in values]
	if numeric:
		normalized = [value for value in normalized if value is not None]
	if not normalized:
		return None
	counts = pd.Series(normalized).value_counts()
	return counts.index[0]


def accuracy_and_stderr(correct):
	values = np.asarray(correct, dtype=float)
	accuracy = values.mean()
	stderr = np.sqrt(accuracy * (1 - accuracy) / len(values))
	return accuracy, stderr


def sample_path(method):
	config = DATASET_CONFIG[DATASET]
	folders = sorted((RESULTS_ROOT).glob(f"{config['prefix']}_{method}/**/{config['sample_pattern']}"))
	if len(folders) != 1:
		raise FileNotFoundError(
			f"Expected one {DATASET.upper()} sample file for {method}, found {len(folders)}"
		)
	return folders[0]


def load_method(method):
	config = DATASET_CONFIG[DATASET]
	rows = [json.loads(line) for line in sample_path(method).open()]
	
	if DATASET == "gsm8k":
		return load_gsm8k_method(rows, config)
	else:  # humaneval
		return load_humaneval_method(rows, config)


def load_gsm8k_method(rows, config):
	records = {}
	for row in rows:
		filter_name = row["filter"]
		if filter_name not in config["filters"]:
			continue
		doc_id = row["doc_id"]
		records[(doc_id, filter_name)] = {
			"target": row["target"],
			"prediction": row["filtered_resps"][0],
			"correct": bool(row["exact_match"]),
		}

	doc_ids = sorted({doc_id for doc_id, _ in records})
	data = pd.DataFrame(index=doc_ids)
	for filter_name, column in {
		"strict-match": "exact",
		"flexible-extract": "flexible",
	}.items():
		selected = pd.DataFrame.from_dict(
			{doc_id: records[(doc_id, filter_name)] for doc_id in doc_ids},
			orient="index",
		)
		data["target"] = selected["target"]
		data[f"{column}_prediction"] = selected["prediction"]
		data[f"{column}_correct"] = selected["correct"]
	return data


def load_humaneval_method(rows, config):
	records = {}
	for row in rows:
		filter_name = row["filter"]
		if filter_name not in config["filters"]:
			continue
		doc_id = row["doc_id"]
		records[doc_id] = {
			"target": row["target"],
			"prediction": row["filtered_resps"][0],
			"correct": bool(row["pass@1"]),
		}

	doc_ids = sorted(records.keys())
	data = pd.DataFrame.from_dict(records, orient="index")
	# HumanEval uses a single filter, so exact and flexible use the same predictions
	data["exact_prediction"] = data["prediction"]
	data["exact_correct"] = data["correct"]
	data["flexible_prediction"] = data["prediction"]
	data["flexible_correct"] = data["correct"]
	return data[["target", "exact_prediction", "exact_correct", "flexible_prediction", "flexible_correct"]]


def pairwise_table(data, method_a, method_b, metric):
	left = data[f"{method_a}_{metric}_correct"]
	right = data[f"{method_b}_{metric}_correct"]
	table = pd.DataFrame(
		{
			"Only A correct": np.sum(left & ~right),
			"Only B correct": np.sum(~left & right),
			"Both correct": np.sum(left & right),
			"Both wrong": np.sum(~left & ~right),
		},
		index=[f"{method_a} vs {method_b}"],
	)
	return table


def diversity_table(data, metric="flexible", include_numeric_vote=True, include_majority_vote=True):
	methods = list(METHODS)
	correct = data[[f"{method}_{metric}_correct" for method in methods]].to_numpy(bool)
	all_correct = correct.all(axis=1)
	all_wrong = ~correct.any(axis=1)
	mixed = ~(all_correct | all_wrong)
	at_least_one = correct.any(axis=1)

	counts = [
		("All three methods correct", all_correct.sum()),
		("All three methods wrong", all_wrong.sum()),
		("Mixed correctness", mixed.sum()),
		("At least one method correct", at_least_one.sum()),
	]
	
	if include_majority_vote:
		predictions = data[[f"{method}_{metric}_prediction" for method in methods]].to_numpy()
		string_votes = np.array([majority_value(row) for row in predictions])
		targets = data["target"].to_numpy()
		counts.append(
			("Majority vote, string exact", sum(same_answer(vote, target) for vote, target in zip(string_votes, targets)))
		)
		if include_numeric_vote:
			numeric_votes = np.array([majority_value(row, numeric=True) for row in predictions])
			counts.append(
				("Majority vote, numeric equivalent", sum(
					same_answer(vote, target, numeric=True)
					for vote, target in zip(numeric_votes, targets)
				))
			)
	
	total = len(data)
	return pd.DataFrame(
		[(label, count, f"{count} / {total} = {count / total:.2%}") for label, count in counts],
		columns=["Quantity", "Count", "Value"],
	).set_index("Quantity")


def main():
	config = DATASET_CONFIG[DATASET]
	method_data = {method: load_method(method) for method in METHODS}
	data = pd.concat(method_data, axis=1)
	data.columns = [f"{method}_{column}" for method, column in data.columns]
	data["target"] = data["greedy_target"]
	if data.index.duplicated().any() or len(data) != config["expected_count"]:
		raise ValueError(f"Expected {config['expected_count']} aligned {DATASET.upper()} questions, found {len(data)}")

	print(f"\n{'='*60}")
	print(f"Ablation Study: {DATASET.upper()}")
	print(f"{'='*60}\n")

	print("Totals")
	totals = {}
	for method in METHODS:
		exact_accuracy, exact_stderr = accuracy_and_stderr(data[f"{method}_exact_correct"])
		flexible_accuracy, flexible_stderr = accuracy_and_stderr(data[f"{method}_flexible_correct"])
		totals[method] = {
			"Exact correct": int(data[f"{method}_exact_correct"].sum()),
			"Exact accuracy": f"{exact_accuracy:.2%}",
			"Exact stderr": f"{exact_stderr:.2%}",
			"Flexible correct": int(data[f"{method}_flexible_correct"].sum()),
			"Flexible accuracy": f"{flexible_accuracy:.2%}",
			"Flexible stderr": f"{flexible_stderr:.2%}",
		}
	print(pd.DataFrame.from_dict(totals, orient="index").to_string())

	for method_a, method_b in combinations(METHODS, 2):
		print(f"\nTwo-method comparison: {method_a} vs {method_b}")
		for metric in ["exact", "flexible"]:
			print(f"\n{metric.title()} match")
			print(pairwise_table(data, method_a, method_b, metric).to_string())

	# Skip majority voting for HumanEval (coding benchmark)
	include_majority = (DATASET == "gsm8k")
	
	print("\nTable 4: Flexible-match diversity analysis across three cached non-baseline methods")
	print(diversity_table(data, metric="flexible", include_majority_vote=include_majority).to_string())

	if include_majority:
		print("\nTable 5: Flexible-match diversity analysis without numeric equivalence")
		print(diversity_table(data, metric="flexible", include_numeric_vote=False, include_majority_vote=True).to_string())

	print("\nTable 6: Strict-match diversity analysis across three cached non-baseline methods")
	print(diversity_table(data, metric="exact", include_majority_vote=include_majority).to_string())


if __name__ == "__main__":
	main()
