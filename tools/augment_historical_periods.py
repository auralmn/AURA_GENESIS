import argparse
import json
from typing import Any, Dict, Optional

from historical_features import parse_year, get_historical_period
from historical_teacher import HistoricalTeacher


CANON_PERIODS = {"Neolithic","Ancient","Medieval","Early_Modern","Industrial","Modern"}


def canon_period(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    t = str(label).strip().lower().replace('-', '_').replace(' ', '_')
    mapping = {
        'neolithic': 'Neolithic',
        'ancient': 'Ancient',
        'classical': 'Ancient',
        'medieval': 'Medieval',
        'middle_ages': 'Medieval',
        'early_modern': 'Early_Modern',
        'earlymodern': 'Early_Modern',
        'industrial': 'Industrial',
        'modern': 'Modern',
    }
    return mapping.get(t, None)


def choose_period(ev: Dict[str, Any], teacher: HistoricalTeacher) -> Dict[str, Any]:
    # 1) If a period/era exists, normalize it
    raw = ev.get('period') or ev.get('era') or ev.get('time_period')
    canon = canon_period(raw)
    if canon:
        return {'period': canon, 'source': 'existing', 'confidence': 1.0}

    # 2) Teacher hint from text fields
    text_blob = ' '.join([
        str(ev.get('text','')),
        str(ev.get('title','')),
        str(ev.get('summary','')),
        str(ev.get('source_text','')),
    ])
    hint, conf = teacher.hint_with_confidence(text_blob)
    if hint:
        return {'period': hint, 'source': 'teacher', 'confidence': float(conf)}

    # 3) Year-based fallback
    ys = parse_year(ev.get('year_start'))
    ye = parse_year(ev.get('year_end'))
    if ys is not None:
        return {'period': get_historical_period(int(ys)), 'source': 'years', 'confidence': 0.6}
    if ye is not None:
        return {'period': get_historical_period(int(ye)), 'source': 'years', 'confidence': 0.6}

    return {'period': None, 'source': 'unknown', 'confidence': 0.0}


def run(input_path: str, output_path: str, teacher_path: str) -> Dict[str, Any]:
    teacher = HistoricalTeacher(teacher_path)
    total = 0
    updated = 0
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            total += 1
            choice = choose_period(ev, teacher)
            if choice['period']:
                ev['period'] = choice['period']
                ev['period_source'] = choice['source']
                ev['period_confidence'] = choice['confidence']
                updated += 1
            fout.write(json.dumps(ev, ensure_ascii=False) + '\n')
    return {'total': total, 'updated': updated, 'coverage': (updated / max(1,total))}


def main():
    ap = argparse.ArgumentParser(description='Augment historical JSONL with canonical period names')
    ap.add_argument('--input', required=True, help='Path to input JSONL')
    ap.add_argument('--output', required=True, help='Path to output JSONL')
    ap.add_argument('--teacher', type=str, default='historical_teacher.md')
    args = ap.parse_args()
    res = run(args.input, args.output, args.teacher)
    print(res)


if __name__ == '__main__':
    main()

