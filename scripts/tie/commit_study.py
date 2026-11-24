"""Run python scripts/tie/commit_study.py --help for study analysis commands."""

import argparse
import json
import random
from pathlib import Path


FRESH_GSM8K_TASK = "gsm8k_cot_train_dev"
MODEL_BILLIONS = {"llada": 8.016, "dream": 7.616}
SPLITS = {"pilot": range(32), "fit": range(32, 200),
          "validation": range(200, 300), "heldout": range(300, 1319),
          "fresh": sorted(random.Random(20261122).sample(range(7473), 192))}
SPLITS["support"] = sorted(random.Random(20261123).sample(
    sorted(set(range(7473)) - set(SPLITS["fresh"])), 32
))
SPLITS["replication"] = sorted(random.Random(20261124).sample(
    sorted(set(range(7473)).difference(SPLITS["fresh"], SPLITS["support"])), 768
))
SPLITS["periodic_js"] = sorted(random.Random(202611241).sample(
    sorted(set(range(7473)).difference(SPLITS["fresh"], SPLITS["support"], SPLITS["replication"])), 32
))
SPLITS["selective_js"] = sorted(random.Random(202611242).sample(
    sorted(set(range(7473)).difference(
        SPLITS["fresh"], SPLITS["support"], SPLITS["replication"], SPLITS["periodic_js"],
    )), 64
))


def read(paths, expected, mode=None):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if mode in {"native_llada", "native_dream"}:
                if row["filter"] != "flexible-extract":
                    continue
                ids = json.loads((Path(path).parent.parent / "split.json").read_text())
                row = {"mode": mode, "doc_id": ids["gsm8k_cot"][row["doc_id"]],
                       "correct": bool(row["exact_match"])}
            doc = row["doc_id"]
            if doc in rows or (mode and row["mode"] != mode):
                raise ValueError(f"Duplicate document or unexpected mode at {path}, doc {doc}")
            rows[doc] = row
    if sorted(rows) != list(expected):
        raise ValueError("Trace documents do not exactly match the frozen split")
    return rows


def events(rows, branch):
    records = []
    for row in rows.values():
        base = row["branches"]["keep"]["correct"]
        alternative = row["branches"].get(branch)
        event = next(iter(row["attempts"]), {})
        delta = 0 if alternative is None else int(alternative["correct"]) - int(base)
        records.append((event, delta, alternative is not None))
    return records


def evaluate(records, gate):
    effects = [delta for event, delta, changed in records if changed and gate(event)]
    return {"accepted": len(effects), "wins": effects.count(1),
            "losses": effects.count(-1), "net": sum(effects), "questions": len(records),
            "attempted": sum(bool(event) for event, _, _ in records),
            "eligible": sum("ids_b" in event for event, _, _ in records),
            "changed": sum(changed for _, _, changed in records)}


def fit(args):
    scheduling = args.command == "fit-schedule"
    mode = "schedule_diagnostic" if scheduling else "diagnostic"
    branch = "positions" if scheduling else "replace"
    fit_rows = events(read([args.fit], SPLITS["fit"], mode), branch)
    val_rows = events(read([args.validation], SPLITS["validation"], mode), branch)
    caps = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    output = {"threshold_grid": caps}
    if scheduling:
        output = json.loads(Path(args.span_policy).read_text())
        if not output["span_pass"]:
            raise ValueError("The span gate must pass before scheduling")
        grid = [-0.30, -0.15, 0.0, 0.15, 0.30]
        gate = lambda e, threshold: e["schedule_margin"] >= threshold
        threshold = max(grid, key=lambda t: objective(evaluate(fit_rows, lambda e: gate(e, t))))
        output["schedule"] = threshold
        output["schedule_grid"] = grid
        output["schedule_validation"] = evaluate(val_rows, lambda e: gate(e, threshold))
    else:
        gate = lambda e, d, v: e["dream_margin"] >= d and e["llada_margin"] >= -v
        grid = [(d, v) for d in caps for v in caps]
        d, v = max(grid, key=lambda p: objective(evaluate(fit_rows, lambda e: gate(e, *p))))
        llada = max(caps, key=lambda t: objective(evaluate(
            fit_rows, lambda e: e["llada_margin"] >= t
        )))
        combined = evaluate(val_rows, lambda e: gate(e, d, v))
        receiver = evaluate(val_rows, lambda e: e["llada_margin"] >= llada)
        output.update(dream=d, veto=v, llada_only=llada, span_validation=combined,
                      llada_validation=receiver,
                      span_pass=combined["net"] > max(0, receiver["net"]))
    return output


def objective(result):
    return result["net"], -result["accepted"]


def cost(row):
    costs = row.get("scoring_cost", {})
    if "branches" in row:
        winner = row["branches"].get(row["mode"], row["branches"]["keep"])
        items = [winner]
    else:
        items = row.get("completions", [])
    return sum(MODEL_BILLIONS[item["member"]] * item["processed_tokens"] for item in items) + sum(
        MODEL_BILLIONS[model] * work[1] for model, work in costs.items()
    )


def comparison(candidate, baseline):
    changes = [int(candidate[d]["correct"]) - int(baseline[d]["correct"]) for d in candidate]
    rng = random.Random(1234)
    n = len(changes)
    totals = sorted(sum(rng.choices(changes, k=n)) / n for _ in range(5000))
    return {"wins": changes.count(1), "losses": changes.count(-1),
            "gain": sum(changes) / n, "paired_95_ci": [totals[124], totals[4874]]}


def pilot_check(paths):
    llada, dream, post, relay = [
        read([path], SPLITS["pilot"], mode)
        for path, mode in zip(paths, ("llada", "dream", "post_generation", "relay"))
    ]
    parity = [
        doc for doc, row in post.items() if row["completions"] != [
            llada[doc]["completions"][0], dream[doc]["completions"][0]
        ]
    ]
    directions = {"llada": 0, "dream": 0}
    skipped = 0
    for row in relay.values():
        for event in row["relays"]:
            if event["applied"]:
                directions[event["leader"]] += 1
            else:
                skipped += 1
    return {"parity_mismatches": parity, "successful_relays_by_donor": directions,
            "skipped_relays": skipped, "pass": not parity and all(directions.values())}


def summarize(args):
    traces = {}
    for spec in args.trace:
        arm, path = spec.split("=", 1)
        if arm in traces:
            raise ValueError(f"Duplicate arm {arm}")
        traces[arm] = read(path.split(","), SPLITS[args.split], arm)
    summary = {arm: {"accuracy": sum(r["correct"] for r in rows.values()) / len(rows),
                     "weighted_tokens": (None if arm.startswith("native_")
                                         else sum(map(cost, rows.values()))),
                     "interventions": sum(bool(r.get("attempts") and
                         r["attempts"][0].get("writes", {}).get(arm))
                         for r in rows.values())}
               for arm, rows in traces.items()}
    if args.command == "freeze":
        required = {"llada", "text", "positions", "joint"}
        if not required <= traces.keys():
            raise ValueError("Freeze requires all four factorial arms")
        policy = json.loads(Path(args.policy).read_text())
        if not policy.get("span_pass") or "schedule" not in policy:
            raise ValueError("Freeze requires calibrated content and scheduling gates")
        selected = max(sorted(required), key=lambda arm: (summary[arm]["accuracy"],
                       -summary[arm]["weighted_tokens"], -summary[arm]["interventions"]))
        return policy | {"selected": selected, "validation": summary}
    selected = json.loads(Path(args.policy).read_text())["selected"]
    required = {selected, "llada", "dream", "ctca", "relay", "multi",
                "native_llada", "native_dream"}
    if not required <= traces.keys():
        raise ValueError(f"Missing held-out arms: {sorted(required - traces.keys())}")
    paired = {arm: comparison(traces[selected], rows)
              for arm, rows in traces.items() if arm != selected}
    multi_matched = (
        abs(summary[selected]["weighted_tokens"] / summary["multi"]["weighted_tokens"] - 1)
        <= 0.05
    )
    strongest_native = max(
        ("native_llada", "native_dream"), key=lambda arm: summary[arm]["accuracy"]
    )
    controls = (strongest_native, "ctca", "relay", "multi")
    return {"selected": selected, "arms": summary, "paired": paired,
            "multi_cost_matched": multi_matched, "strongest_native": strongest_native,
            "system_win": multi_matched and all(
                summary[selected]["weighted_tokens"] <= 1.05 * summary[arm]["weighted_tokens"]
                for arm in ("ctca", "relay")
            ) and all(
                arm in paired and paired[arm]["paired_95_ci"][0] > 0 for arm in controls
            ),
            "joint_gain_supported": selected == "joint" and all(
                arm in paired and paired[arm]["paired_95_ci"][0] > 0
                for arm in ("text", "positions")
            )}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("name", choices=SPLITS)
    split.add_argument("--shard", type=int, choices=range(4))
    commands.add_parser("pilot-check").add_argument(
        "trace", nargs=4, metavar="TRACE",
        help="Supply LLaDA, Dream, post-generation and relay traces in that order.",
    )
    for name in ("fit-span", "fit-schedule"):
        command = commands.add_parser(name)
        command.add_argument("fit")
        command.add_argument("validation")
        if name == "fit-schedule":
            command.add_argument("span_policy")
    for name in ("freeze", "report"):
        command = commands.add_parser(name)
        command.add_argument("policy")
        command.add_argument("trace", nargs="+", metavar="ARM=TRACE")
        command.set_defaults(split="validation" if name == "freeze" else "heldout")
    args = parser.parse_args()
    if args.command == "split":
        ids = list(SPLITS[args.name])
        if args.shard is not None:
            if args.name != "heldout":
                parser.error("Only the held-out split is sharded")
            ids = ids[255 * args.shard:255 * (args.shard + 1)]
        train_splits = {"fresh", "support", "replication", "periodic_js", "selective_js"}
        task = FRESH_GSM8K_TASK if args.name in train_splits else "gsm8k_cot"
        out = {task: ids}
    elif args.command.startswith("fit-"):
        out = fit(args)
    elif args.command == "pilot-check":
        out = pilot_check(args.trace)
    else:
        out = summarize(args)
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
