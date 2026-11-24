"""Rescore saved responses with python -m dllm.pipelines.tie.eval --model tie."""

import json
from pathlib import Path

from .action_fusion import numeric_answer_span, score_responses


MODELS = ("llada", "dream")


def load_saved_answers(trace_path, samples_path, doc_ids):
    """Bind the independent responses to the original lm-eval prompts."""
    answers, prompts = {}, {}
    with Path(trace_path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc = row["doc_id"]
            if row["mode"] != "shared_audit" or doc in answers:
                raise ValueError(f"Unexpected or repeated saved trace for doc {doc}")
            answers[doc] = {
                model: {field: row["owners"][model]["baseline"][field]
                        for field in ("response", "correct")}
                for model in MODELS
            }
    with Path(samples_path).open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            if sample["filter"] != "flexible-extract":
                continue
            doc = sample["doc_id"]
            if doc in prompts or doc not in answers:
                raise ValueError(f"Unexpected or repeated saved sample for doc {doc}")
            if sample["exact_match"] != answers[doc]["llada"]["correct"]:
                raise ValueError(f"Saved LLaDA correctness differs for doc {doc}")
            prompts[doc] = sample["arguments"]["gen_args_0"]
    if sorted(answers) != list(doc_ids) or sorted(prompts) != list(doc_ids):
        raise ValueError("Saved traces and sample prompts must match the declared split")
    return {doc: (prompts[doc], answers[doc]) for doc in doc_ids}


def score_saved_answers(members, prompts, marker, saved_answers, *, scoring_texts=None):
    spans = None
    if scoring_texts is None:
        scoring_texts = {model: candidate["response"] for model, candidate in saved_answers.items()}
    else:
        spans = {text: numeric_answer_span(text) for text in scoring_texts.values()}
    scores, costs = score_responses(
        members, prompts, marker, scoring_texts.values(), answer_spans=spans
    )
    candidates = {
        model: {**candidate, "scores": scores[scoring_texts[model]]}
        for model, candidate in saved_answers.items()
    }
    if spans is not None:
        for model, text in scoring_texts.items():
            span = spans[text]
            candidates[model]["extracted_answer"] = None if span is None else text[span[0]:span[1]]
    return {"candidates": candidates, "generation_forwards": 0, "scoring_cost": costs}
