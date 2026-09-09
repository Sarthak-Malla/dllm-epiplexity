"""Audit completed Ablation 4 runs against the entropy-budget reference on CPU.

Run after sourcing ~/.zshrc (or /apps/local/conda_init.sh if absent) and
activating the dllm conda environment:
    python /home/sarthak.malla/dllm-selection-ensemble/examples/path_selection/ablation/4_marginal_utility/analyze_results.py
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np


ROOT = Path('/home/sarthak.malla/dllm-selection-ensemble')
ABLATION = ROOT / 'eval_results/path_selection/ablation'
OUTPUT = ABLATION / 'analysis_marginal_utility_20260908'


def distribution(values):
    """Summarize an observed scalar distribution without rounding stored values."""
    values = np.asarray(values, dtype=float)
    return dict(zip(('mean', 'min', 'p50', 'p90', 'max'), (
        float(values.mean()), float(values.min()),
        float(np.quantile(values, .5)), float(np.quantile(values, .9)),
        float(values.max()),
    )))


def load_run(directory):
    """Validate sample metrics and reduce diagnostics, excluding distributed padding."""
    result_path, = directory.glob('results_*.json')
    sample_path, = directory.glob('samples*.jsonl')
    result = json.loads(result_path.read_text())
    samples = {}
    with sample_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            samples.setdefault(row['filter'], {})[row['doc_id']] = row
    for name, rows in samples.items():
        assert len(rows) == 1319
        assert np.isclose(np.mean([r['exact_match'] for r in rows.values()]),
                          result['results']['gsm8k_cot'][f'exact_match,{name}'])
    flexible = samples['flexible-extract']
    manifest = json.loads((directory / 'results.json_entropy_drop_diagnostics_manifest.json').read_text())
    sizes, costs, calls, reasons, first_rows, example_rows = [], [], [], Counter(), [], []
    fallback = 0
    token_high_cost = 0
    padding = 0
    step_counts = []
    timing = Counter()
    for rank, shard in enumerate(manifest['shards']):
        records = json.loads(Path(shard).read_text())
        assert len(records) == 660
        assert [r['example_index'] for r in records] == list(range(660))
        # task.doc_iterator uses rank-strided documents; generate_until preserves
        # request order. The harness appends one padding request to rank 1.
        for record in records:
            doc_id = manifest['world_size'] * record['example_index'] + rank
            if doc_id not in flexible:
                padding += 1
                continue
            steps = record['steps']
            selected = [s['selected_candidate'] for s in steps]
            assert sum(s['action_size'] for s in selected) == 256
            assert steps[0]['remaining_response_masks'] == 256
            for current, following in zip(steps, steps[1:]):
                assert current['remaining_response_masks'] - current['commit_k'] == following['remaining_response_masks']
            step_counts.append(len(steps))
            calls.append(sum(s['captured_base_forward_count'] + s['lookahead_model_calls'] for s in steps))
            for step, action in zip(steps, selected):
                size, cost = action['action_size'], action['immediate_action_cost']
                sizes.append(size)
                costs.append(cost)
                reasons[action['stopping_reason']] += 1
                fallback += bool(action['fallback'])
                token_high_cost += size if cost / size > 1 else 0
                timing.update(step['timing_seconds'])
            first = selected[0]
            next_step = steps[1]
            first_row = {
                'doc_id': doc_id, 'size': first['action_size'],
                'entropy_sum': first['immediate_action_cost'],
                'entropy_per_token': first['immediate_action_cost'] / first['action_size'],
                'reliable_count': steps[0]['reliable_anchor_count_after'],
                'proposal_score': first['proposal_score'],
                'lookahead_score_per_token': first['verifier_score'],
                'stop_reason': first['stopping_reason'],
                'next_consistency_count': next_step['immediate_token_consistency_count'],
                'next_consistency_total': next_step['immediate_token_consistency_total'],
                'correct': flexible[doc_id]['exact_match'],
            }
            first_rows.append(first_row)
            example_rows.append({'doc_id': doc_id, 'steps': len(steps), 'calls': calls[-1],
                                 'first_action': first_row})
    assert len(first_rows) == 1319 and padding == 1
    n, tokens = len(sizes), sum(sizes)
    runtime = json.loads((directory / 'results.json_entropy_drop_runtime.json').read_text())
    summary = {
        'directory': str(directory), 'result_path': str(result_path),
        'sample_path': str(sample_path), 'config': result['config'],
        'metrics': result['results']['gsm8k_cot'], 'documents': 1319,
        'padding_excluded': padding,
        'correct': int(sum(r['exact_match'] for r in flexible.values())),
        'actions_per_example': distribution(step_counts),
        'model_calls_per_example': distribution(calls),
        'action_size': distribution(sizes), 'action_size_histogram': dict(Counter(sizes)),
        'singleton_action_fraction': sizes.count(1) / n,
        'token_fraction_in_actions_gt32': sum(s for s in sizes if s > 32) / tokens,
        'token_fraction_in_actions_mean_entropy_gt1': token_high_cost / tokens,
        'committed_entropy_per_token': sum(costs) / tokens,
        'selected_stop_reasons': dict(reasons), 'selected_fallback_fraction': fallback / n,
        'first_action': {key: distribution([r[key] for r in first_rows]) for key in (
            'size', 'entropy_sum', 'entropy_per_token', 'reliable_count', 'proposal_score',
            'lookahead_score_per_token')},
        'first_action_size_gt32_fraction': sum(r['size'] > 32 for r in first_rows) / 1319,
        'first_action_stop_reasons': dict(Counter(r['stop_reason'] for r in first_rows)),
        'first_action_reliable_token_fraction': sum(r['reliable_count'] for r in first_rows) / sum(r['size'] for r in first_rows),
        'first_action_next_consistency': sum(r['next_consistency_count'] for r in first_rows) / sum(r['next_consistency_total'] for r in first_rows),
        'generation_hours': runtime['generation_total_seconds'] / 3600,
        'summed_step_timing_seconds': dict(timing),
        'invalid_flexible_count': sum(r['filtered_resps'] == ['[invalid]'] for r in flexible.values()),
        'examples': example_rows,
    }
    return summary, samples


def main():
    """Write validated aggregate measurements, paired comparisons, and examples."""
    directories = {'entropy_B2': ABLATION / '2_entropy_budget_without_k_limit/vectorized_soft_full_v1/gsm8k_cot/entropy_budget2.0/seed42'}
    for tau in ('0.0', '0.25', '0.5'):
        directories[f'tau{tau}'] = ABLATION / f'4_marginal_utility/marginal_utility_v1/gsm8k_cot/marginal_utility_tau{tau}/seed42'
    summaries, all_samples = {}, {}
    for name, directory in directories.items():
        summaries[name], all_samples[name] = load_run(directory)
        s = summaries[name]
        print(name, json.dumps({k: s[k] for k in ('correct', 'action_size', 'model_calls_per_example', 'first_action', 'first_action_next_consistency', 'selected_stop_reasons', 'generation_hours')}), flush=True)
    reference = all_samples['entropy_B2']['flexible-extract']
    paired = {}
    for name, samples in all_samples.items():
        rows = samples['flexible-extract']
        assert rows.keys() == reference.keys()
        for doc_id in reference:
            for key in ('doc', 'arguments', 'target', 'prompt_hash'):
                assert rows[doc_id][key] == reference[doc_id][key]
        if name == 'entropy_B2':
            continue
        delta = np.array([rows[i]['exact_match'] - reference[i]['exact_match'] for i in sorted(reference)])
        paired[name] = {'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum()),
                        'difference_pp': float(delta.mean() * 100)}
    first = all_samples['tau0.0']['flexible-extract']
    second = all_samples['tau0.25']['flexible-extract']
    pair_identical = sum(first[i]['resps'] == second[i]['resps'] for i in first)
    config_differences = {}
    base_args = dict(item.split('=', 1) for item in summaries['entropy_B2']['config']['model_args'].split(','))
    for name, summary in summaries.items():
        args = dict(item.split('=', 1) for item in summary['config']['model_args'].split(','))
        config_differences[name] = {key: [base_args.get(key), args.get(key)]
                                    for key in base_args.keys() | args.keys()
                                    if base_args.get(key) != args.get(key)}
    output = {'runs': summaries, 'paired': paired, 'config_differences': config_differences,
              'tau0_vs_tau025_identical_responses': pair_identical,
              'checks': '1319 matching documents, prompts and targets; per-filter metrics reproduced; 256 commits per trace; one padding trace excluded per run',
              'doc0_responses': {name: samples['flexible-extract'][0]['resps'] for name, samples in all_samples.items()}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print('PAIRED', paired, 'IDENTICAL_TAU0_TAU025', pair_identical)
    print('CONFIG_DIFFERENCES', config_differences)
    print('OUTPUT', OUTPUT / 'summary.json')


if __name__ == '__main__':
    main()
