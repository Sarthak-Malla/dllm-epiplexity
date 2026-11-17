"""Replay selected benchmark cases through the existing full-trace logger.

Run inside a two-GPU allocation: python scripts/tse/ctca/benchmark_replay.py --study-id ID
Use --case DOC:MASTER to choose a case; omit it for the original four losses.
Use --condition ensemble to replay only the ensemble trajectory.
Use --projection-mode and --num-anchors for a matched projection comparison.
Use --prepare-only to validate sources and write prompts without loading models.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CASES = [(23, 'b'), (18, 'b'), (9, 'b'), (11, 'a')]
JOBS = {'a': '352346', 'b': '352347'}
ENDPOINTS = {'a': '350508', 'b': '352165'}


def samples(job):
    paths = list(ROOT.glob(f'.logs/**/{job}/samples_*.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected one sample file for {job}, found {paths}')
    return {int(x['doc_id']): x for x in map(json.loads, paths[0].read_text().splitlines())
            if x['filter'] == 'flexible-extract'}


def prompt(row):
    return row['arguments']['gen_args_0']['arg_0']


def normalize(text):
    for stop in ('Q:', '</s>', '<|im_end|>'):
        text = text.split(stop)[0]
    return re.sub(r'<\|[^>]*\|>', '', text).strip()


def parse_case(value):
    match = re.fullmatch(r'(\d+):([ab])', value)
    if not match:
        raise argparse.ArgumentTypeError('case must be DOC:a or DOC:b')
    return int(match.group(1)), match.group(2)


def parse_anchor_count(value):
    if value == 'auto':
        return value
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('num-anchors must be positive or auto') from error
    if count < 1:
        raise argparse.ArgumentTypeError('num-anchors must be positive or auto')
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--case', type=parse_case, action='append',
                        help='Document and master role, e.g. 3:a; repeat for more cases')
    parser.add_argument('--condition', choices=('ensemble', 'endpoint'), action='append',
                        help='Replay condition; defaults to both')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--projection-mode', choices=('exact', 'sparse_topk', 'relative_distribution'))
    parser.add_argument('--num-anchors', type=parse_anchor_count)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.study_id):
        parser.error('study-id must be a simple directory name')
    cases = args.case or CASES
    conditions = args.condition or ('ensemble', 'endpoint')
    mode_suffix = f'-{args.projection_mode}' if args.projection_mode else ''
    if args.num_anchors is not None:
        mode_suffix += f'-{args.num_anchors}'
    if len(conditions) != len(set(conditions)):
        parser.error('each condition must appear once')
    if len(cases) != len(set(cases)) or len({doc for doc, _ in cases}) != len(cases):
        parser.error('each document must appear once')
    rows = {job: samples(job) for job in set(JOBS.values()) | set(ENDPOINTS.values())}
    job_id = os.environ.get('SLURM_JOB_ID', 'manual')
    root = ROOT / '.logs' / args.study_id
    out = root / job_id
    out.mkdir(parents=True, exist_ok=True)
    for doc, master in cases:
        if not all(doc in rows[job] for job in set(JOBS.values()) | set(ENDPOINTS.values())):
            parser.error(f'doc {doc} is absent from one or more benchmark sample files')
        sources = {m: rows[j][doc] for m, j in JOBS.items()}
        assert sources['a']['doc_hash'] == sources['b']['doc_hash']
        for m in ('a', 'b'):
            (out / f'doc{doc}-prompt-{m}.txt').write_text(prompt(sources[m]))
        traces = []
        checks = []
        for condition in conditions:
            source_job = JOBS[master] if condition == 'ensemble' else ENDPOINTS[master]
            source = rows[source_job][doc]
            assert source['doc_hash'] == sources[master]['doc_hash']
            assert prompt(source) == prompt(sources[master]), 'Endpoint prompt differs'
            name = f'doc{doc}-{master}-{condition}{mode_suffix}'
            trace = out / f'{name}.jsonl'
            command = [sys.executable, str(ROOT / 'scripts/tse/smoke.py'),
                '--model-a', 'GSAI-ML/LLaDA-8B-Instruct',
                '--seed', '1234',
                '--model-b', 'Dream-org/Dream-v0-Instruct-7B', '--ctca',
                '--master-model', master, '--fusion-device', 'cuda:0' if master == 'a' else 'cuda:1',
                '--max-new-tokens', '512', '--steps', '128', '--block-size', '32',
                '--selection-mode', 'tse', '--weighting-mode', 'static',
                '--alpha', '0.5' if condition == 'ensemble' else ('1.0' if master == 'a' else '0.0'),
                '--temperature-a', '1.0', '--temperature-b', '1.0', '--temperature', '0.0',
                '--ctca-shared-token-policy', 'exact_shared', '--ctca-chunk-size', '512',
                '--trace-top-k', '64', '--trace-jsonl', str(trace),
                '--prompt-a-file', str(out / f'doc{doc}-prompt-a.txt'),
                '--prompt-b-file', str(out / f'doc{doc}-prompt-b.txt'),
                '--question', source['doc']['question'], '--question-label', name,
                '--gold-answer', str(source['target']), '--source-doc-id', str(doc),
                '--source-job-id', source_job]
            if master == 'a':
                command += ['--master-begin-suppress-tokens', '126081', '126348']
            if args.projection_mode:
                command += ['--ctca-projection-mode', args.projection_mode]
            if args.num_anchors is not None:
                command += ['--ctca-num-anchors', str(args.num_anchors)]
            (out / f'{name}-command.json').write_text(json.dumps(command, indent=2))
            if args.prepare_only:
                continue
            with (out / f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = next(x for x in map(json.loads, trace.read_text().splitlines())
                          if x['event'] == 'question_result')
            checks.append({'condition': condition, 'source_job': source_job,
                           'matches_original_response': normalize(result['response']) == normalize(source['resps'][0][0]),
                           'original_response': source['resps'][0][0], 'replay_response': result['response']})
            traces.append('/' + str(trace.relative_to(ROOT)))
        if not args.prepare_only:
            (out / f'doc{doc}{mode_suffix}-reproduction.json').write_text(json.dumps(checks, indent=2))
            manifest = {'job_id': job_id, 'traces': traces, 'reproduction': checks,
                        'projection_mode': args.projection_mode, 'num_anchors': args.num_anchors}
            (root / f'doc{doc}{mode_suffix}.json').write_text(json.dumps(manifest))
            (root / 'latest.json').write_text(json.dumps(manifest))
        if args.prepare_only:
            print(f'Prepared doc {doc}, master {master}', flush=True)
        else:
            print(f'Completed doc {doc}: ' + ', '.join(
                f'{check["condition"]}={check["matches_original_response"]}'
                for check in checks), flush=True)


if __name__ == '__main__':
    main()
