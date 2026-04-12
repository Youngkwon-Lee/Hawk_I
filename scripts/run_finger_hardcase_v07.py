#!/usr/bin/env python3
import json, os, re, time, csv
from pathlib import Path
from datetime import datetime

from google import genai


def safe_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def parse_answer(text: str):
    if not text:
        return None, False
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            a = safe_int(o.get("answer"))
            return a, a is not None
        except Exception:
            pass
    m2 = re.search(r"\b([0-4])\b", text)
    if m2:
        return int(m2.group(1)), True
    return None, False


def parse_gate(text: str):
    t = (text or "").lower()
    if "moderate_or_more" in t or "high" in t:
        return "high", True
    if "mild_or_less" in t or "low" in t:
        return "low", True
    return None, False


def gen_text(client, uploaded, prompt, retries=5):
    last = None
    for i in range(retries):
        try:
            r = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[uploaded, prompt],
                config={"max_output_tokens": 512, "temperature": 0},
            )
            return r.text or ""
        except Exception as e:
            last = e
            time.sleep(min(20, 2 * (i + 1)))
    raise last


def resolve(repo: Path, video_name: str):
    for p in repo.rglob(Path(video_name).name):
        return p
    return None


def main():
    repo = Path('/home/yk/.openclaw/workspace/Hawkeye')
    src = repo / 'rationale_results/finger_hardcase10_20260411_2308/selected_10_samples.json'
    samples = json.loads(src.read_text())

    run_id = 'finger_hardcase10_v07_' + datetime.now().strftime('%Y%m%d_%H%M')
    out = repo / 'rationale_results' / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'selected_10_samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding='utf-8')

    api = os.getenv('GEMINI_API_KEY')
    if not api:
        raise RuntimeError('GEMINI_API_KEY missing')
    client = genai.Client(api_key=api)

    gate_prompt = (
        "Finger tapping triage. Decide ONLY one label: mild_or_less OR moderate_or_more. "
        "Use sustained evidence only. Return JSON: {\"gate\":\"mild_or_less|moderate_or_more\"}."
    )
    low_prompt = (
        "Finger tapping scoring for LOW band only. Choose ONLY 0 or 1. "
        "Return JSON: {\"answer\":0|1}."
    )
    high_prompt = (
        "Finger tapping scoring for HIGH band only. Choose ONLY 2 or 3 or 4. "
        "Return JSON: {\"answer\":2|3|4}."
    )

    rows = []
    for s in samples:
        vp = resolve(repo, s['video_path'])
        if not vp:
            rows.append({**s, 'status': 'video_not_found'})
            continue
        try:
            up = client.files.upload(file=str(vp))
            active = False
            for _ in range(90):
                st = str(client.files.get(name=up.name).state)
                if 'ACTIVE' in st:
                    active = True
                    break
                if 'FAILED' in st:
                    break
                time.sleep(2)
            if not active:
                rows.append({**s, 'status': 'file_not_active'})
                continue

            gtxt = gen_text(client, up, gate_prompt)
            gate, g_ok = parse_gate(gtxt)
            if gate == 'high':
                atxt = gen_text(client, up, high_prompt)
            else:
                atxt = gen_text(client, up, low_prompt)
            ans, a_ok = parse_answer(atxt)

            # hard guardrail: no strong evidence => cap to 2
            if gate == 'high' and ans is not None and ans >= 3 and ('sustained' not in (gtxt or '').lower()):
                ans = 2

            rows.append({
                **s,
                'status': 'ok',
                'gate': gate,
                'gate_ok': g_ok,
                'answer': ans,
                'parse_ok': a_ok,
                'match': (ans == s.get('gt_score')) if ans is not None else False,
            })
        except Exception as e:
            rows.append({**s, 'status': 'error', 'error': str(e)})

    n = len(rows)
    m = sum(1 for r in rows if r.get('match'))
    ok = sum(1 for r in rows if r.get('status') == 'ok')

    with (out / 'results.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (out / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id','gt_score','status','gate','gate_ok','answer','parse_ok','match'])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ['sample_id','gt_score','status','gate','gate_ok','answer','parse_ok','match']})

    summary = '\n'.join([
        '# finger hardcase10 v0.7 live',
        '',
        f'- Samples: {n}',
        f'- OK: {ok}/{n}',
        f'- Match: {m}/{n} ({round((m/n)*100,1) if n else 0}%)',
    ])
    (out / 'summary.md').write_text(summary, encoding='utf-8')
    print('DONE', run_id)


if __name__ == '__main__':
    main()
