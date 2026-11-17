"""Write a complete paired GSM8K question audit.

Run: python /home/juan.guevara/naruto/dllm-epiplexity/scripts/tse/ctca/question_audit.py
Activate the dllm conda environment first. The script only reads saved samples.
"""

import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.logs' / 'ctca-gsm8k300-question-audit-20261104'
PAIRS = (
    ('LLaDA master', '350508', '352346'),
    ('Dream master', '352165', '352347'),
)


def load_samples(job_id):
    paths = list((ROOT / '.logs').glob(f'**/{job_id}/samples_*.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected one sample file for {job_id}; found {paths}')
    rows = {}
    for line in paths[0].read_text().splitlines():
        sample = json.loads(line)
        if sample['filter'] == 'flexible-extract':
            doc_id = int(sample['doc_id'])
            if doc_id in rows:
                raise ValueError(f'Duplicate question {doc_id} in {job_id}')
            rows[doc_id] = sample
    if set(rows) != set(range(300)):
        raise ValueError(f'{job_id} does not contain exactly docs 0–299')
    return rows


def one_line(value):
    return str(value).replace('\n', ' ').strip()


def numeric_value(value):
    value = str(value).strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def write_section(lines, title, rows):
    lines.extend((f'## {title}', '', f'{len(rows)} questions.', ''))
    for row in rows:
        lines.extend((
            f"### Question {row['doc_id']}", '',
            row['question'], '',
            f"- Comparison: `{row['outcome']}`",
            f"- Gold: `{row['gold']}`",
            f"- Baseline {row['baseline_job']}: `{one_line(row['baseline_prediction'])}`",
            f"- Ensemble {row['ensemble_job']}: `{one_line(row['ensemble_prediction'])}`",
            '- Numeric-equivalent to gold despite a failed benchmark score: '
            + ', '.join(name for name, key in (
                ('baseline', 'baseline_numeric_equivalent'),
                ('ensemble', 'ensemble_numeric_equivalent'),
            ) if row[key] and not row[f'{name}_correct'])
            if any(row[key] and not row[f'{name}_correct'] for name, key in (
                ('baseline', 'baseline_numeric_equivalent'),
                ('ensemble', 'ensemble_numeric_equivalent'),
            )) else '',
            '',
        ))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    report = [
        '# GSM8K-300 ensemble question audit', '',
        'Question IDs are zero-based GSM8K document IDs. Correctness is the saved '
        '`flexible-extract` exact-match score. The predicted answers below are the '
        f'saved filtered responses; `{OUT / "all_questions.jsonl"}` also contains '
        'the full raw model responses and reference solutions. Each pair uses the same 300 '
        'questions and matched prompts. A numeric-equivalent note marks a scored '
        'failure whose extracted number equals the gold number after removing '
        'currency symbols, commas, and trailing periods; this is an audit aid, '
        'not a replacement score.', '',
    ]
    sections = []
    for master, baseline_job, ensemble_job in PAIRS:
        baseline = load_samples(baseline_job)
        ensemble = load_samples(ensemble_job)
        rows = []
        for doc_id in sorted(baseline):
            b, e = baseline[doc_id], ensemble[doc_id]
            if b['doc_hash'] != e['doc_hash'] or b['target_hash'] != e['target_hash']:
                raise ValueError(f'Mismatched question or target at doc {doc_id}')
            if b['arguments']['gen_args_0']['arg_0'] != e['arguments']['gen_args_0']['arg_0']:
                raise ValueError(f'Mismatched prompt at doc {doc_id}')
            b_correct, e_correct = bool(b['exact_match']), bool(e['exact_match'])
            if not b_correct and e_correct:
                outcome = 'corrected'
            elif b_correct and not e_correct:
                outcome = 'regressed'
            elif not e_correct:
                outcome = 'both_wrong'
            else:
                outcome = 'both_correct'
            row = {
                'master': master,
                'doc_id': doc_id,
                'outcome': outcome,
                'baseline_job': baseline_job,
                'ensemble_job': ensemble_job,
                'question': b['doc']['question'],
                'gold': b['target'],
                'reference_solution': b['doc']['answer'],
                'baseline_prediction': b['filtered_resps'][0],
                'ensemble_prediction': e['filtered_resps'][0],
                'baseline_correct': b_correct,
                'ensemble_correct': e_correct,
                'baseline_numeric_equivalent':
                    numeric_value(b['filtered_resps'][0]) == numeric_value(b['target']),
                'ensemble_numeric_equivalent':
                    numeric_value(e['filtered_resps'][0]) == numeric_value(e['target']),
                'baseline_response': b['resps'][0][0],
                'ensemble_response': e['resps'][0][0],
            }
            rows.append(row)
            all_rows.append(row)
        wrong = [x for x in rows if not x['ensemble_correct']]
        corrected = [x for x in rows if x['outcome'] == 'corrected']
        regressed = [x for x in rows if x['outcome'] == 'regressed']
        both_wrong = [x for x in rows if x['outcome'] == 'both_wrong']
        sections.append((master, baseline_job, ensemble_job, wrong, corrected))
        report.extend((
            f'## {master}: {ensemble_job} vs {baseline_job}', '',
            f"Baseline correct: {sum(x['baseline_correct'] for x in rows)}/300; "
            f"ensemble correct: {sum(x['ensemble_correct'] for x in rows)}/300. "
            f'Ensemble wrong: {len(wrong)} = {len(regressed)} regressions + '
            f'{len(both_wrong)} questions both missed. '
            f'Ensemble corrections: {len(corrected)}.', '',
        ))
    report.extend((
        '## Patterns visible in the saved answers', '',
        '- Multi-step arithmetic and quantity interpretation can fail even when '
        'a baseline succeeds: question 5 (alternating discount), question 11 '
        '(pastry costs), and question 16 (distance per train) are examples.',
        '- Time and unit handling can shift: question 18 uses days per week; '
        'question 23 changes the candle unit and start time in the Dream ensemble.',
        '- Some responses are visibly corrupted: question 9 contains a long '
        'run of repeated digits in both ensembles.',
        '- The ensembles also repair multi-step questions: LLaDA-master '
        'question 3 correctly multiplies sprints by days; Dream-master '
        'question 14 correctly handles a percentage of the remaining students.',
        '- Some apparent changes are scoring-format changes. For example, '
        'LLaDA baseline question 27 extracts `$16.00.` against gold `16` and '
        'is scored wrong, while its ensemble extracts `$16.` and is scored correct. '
        'The numeric-equivalent notes below expose these cases.', '',
    ))
    for master, baseline_job, ensemble_job, wrong, corrected in sections:
        write_section(report, f'{master} ensemble {ensemble_job}: every wrong question', wrong)
        write_section(report, f'{master} ensemble {ensemble_job}: every correction over {baseline_job}', corrected)
    (OUT / 'report.md').write_text('\n'.join(report) + '\n')
    with (OUT / 'all_questions.jsonl').open('w') as output:
        for row in all_rows:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    fields = [key for key in all_rows[0] if key not in
              ('reference_solution', 'baseline_response', 'ensemble_response')]
    with (OUT / 'all_questions.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in all_rows)
    browser_rows = [{key: value for key, value in row.items()
                     if key not in ('reference_solution',)} for row in all_rows]
    browser_data = json.dumps(browser_rows, ensure_ascii=False).replace('</', '<\\/')
    browser = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GSM8K ensemble question audit</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}
header{position:sticky;top:0;background:#fff;padding:1rem 0;border-bottom:1px solid #ddd}
label{margin-right:1rem}select,input{font:inherit;padding:.4rem}article{border:1px solid #ddd;border-radius:8px;margin:1rem 0;padding:1rem}
.answers{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.answer{background:#f6f7f8;padding:.75rem;border-radius:6px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.45 ui-monospace,monospace;max-height:28rem;overflow:auto}
.meta{color:#555;font-size:.9rem}@media(max-width:700px){.answers{grid-template-columns:1fr}}
</style>
<header><h1>GSM8K ensemble question audit</h1>
<label>Ensemble <select id="master"><option value="all">Both</option><option>LLaDA master</option><option>Dream master</option></select></label>
<label>Outcome <select id="outcome"><option value="wrong">Ensemble wrong</option><option value="regressed">Baseline correct, ensemble wrong</option><option value="both_wrong">Both wrong</option><option value="corrected">Ensemble corrected baseline</option><option value="all">All questions</option></select></label>
<label>Question ID <input id="search" type="search" placeholder="Any" size="8"></label><span id="count"></span></header>
<main id="results"></main>
<script>
const rows=DATA;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const m=document.querySelector('#master').value,o=document.querySelector('#outcome').value,q=document.querySelector('#search').value.trim();
let shown=rows.filter(r=>(m==='all'||r.master===m)&&(!q||String(r.doc_id)===q)&&(o==='all'||(o==='wrong'?!r.ensemble_correct:r.outcome===o)));
document.querySelector('#count').textContent=` ${shown.length} questions`;
document.querySelector('#results').innerHTML=shown.map(r=>`<article><h2>${esc(r.master)} · Question ${r.doc_id}</h2>
<div class="meta">${esc(r.outcome)} · gold ${esc(r.gold)} · baseline ${esc(r.baseline_job)} extracted: ${esc(r.baseline_prediction)} · ensemble ${esc(r.ensemble_job)} extracted: ${esc(r.ensemble_prediction)}</div>
<h3>Question</h3><p>${esc(r.question).replace(/\\n/g,'<br>')}</p>
<div class="answers"><section class="answer"><h3>Baseline ${esc(r.baseline_job)} · ${r.baseline_correct?'correct':'wrong'}</h3><pre>${esc(r.baseline_response)}</pre></section>
<section class="answer"><h3>Ensemble ${esc(r.ensemble_job)} · ${r.ensemble_correct?'correct':'wrong'}</h3><pre>${esc(r.ensemble_response)}</pre></section></div></article>`).join('')||'<p>No matching questions.</p>'}
for(const id of ['master','outcome','search'])document.getElementById(id).addEventListener('input',render);
render();
</script></html>'''.replace('DATA', browser_data)
    (OUT / 'browse.html').write_text(browser)
    print(OUT / 'report.md')
    print(OUT / 'browse.html')
    for master, baseline_job, ensemble_job, wrong, corrected in sections:
        print(f'{master} {ensemble_job} vs {baseline_job}: '
              f'{len(wrong)} wrong, {len(corrected)} corrected')


if __name__ == '__main__':
    main()
