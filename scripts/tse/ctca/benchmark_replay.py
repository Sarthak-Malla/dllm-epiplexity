"""Replay benchmark losses through the existing full-trace logger.

Run inside a two-GPU allocation: python scripts/tse/ctca/benchmark_replay.py --study-id ID
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.study_id):
        parser.error('study-id must be a simple directory name')
    rows = {job: samples(job) for job in set(JOBS.values()) | set(ENDPOINTS.values())}
    job_id = os.environ.get('SLURM_JOB_ID', 'manual')
    root = ROOT / '.logs' / args.study_id
    out = root / job_id
    out.mkdir(parents=True, exist_ok=True)
    for doc, master in CASES:
        sources = {m: rows[j][doc] for m, j in JOBS.items()}
        assert sources['a']['doc_hash'] == sources['b']['doc_hash']
        for m in ('a', 'b'):
            (out / f'doc{doc}-prompt-{m}.txt').write_text(prompt(sources[m]))
        traces = []
        checks = []
        for condition in ('ensemble', 'endpoint'):
            source_job = JOBS[master] if condition == 'ensemble' else ENDPOINTS[master]
            source = rows[source_job][doc]
            assert source['doc_hash'] == sources[master]['doc_hash']
            assert prompt(source) == prompt(sources[master]), 'Endpoint prompt differs'
            name = f'doc{doc}-{master}-{condition}'
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
            (out / f'doc{doc}-reproduction.json').write_text(json.dumps(checks, indent=2))
            manifest = {'job_id': job_id, 'traces': traces, 'reproduction': checks}
            (root / f'doc{doc}.json').write_text(json.dumps(manifest))
            (root / 'latest.json').write_text(json.dumps(manifest))
        print(f'Prepared doc {doc}, master {master}' if args.prepare_only else f'Completed doc {doc}: {checks[0]["matches_original_response"]}, {checks[1]["matches_original_response"]}', flush=True)


if __name__ == '__main__':
    main()
