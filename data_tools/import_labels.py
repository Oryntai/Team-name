"""Import local XLSX labels without reading video frames or inferring event types."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import time as clock_time
import json
import math
from pathlib import Path
import re

from data_tools.inventory import _key, inventory

HEADERS = ('наименование файла', 'таймстемп', 'нарушение / без нарушения')


def parse_timestamp(value):
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError('boolean timestamp')
    if isinstance(value, clock_time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        value = value.strip().lower().replace(',', '.')
        if re.fullmatch(r'\d+:\d{2}(?:\.\d+)?', value):
            minute, second = value.split(':')
            if float(second) >= 60:
                raise ValueError('invalid seconds')
            result = int(minute) * 60 + float(second)
        else:
            match = re.fullmatch(r'(?:(\d+)\s*(?:минут[аы]?|мин\.?)\s*)?(\d+(?:\.\d+)?)\s*(?:секунд[аы]?|сек\.?|с)?', value)
            if not match:
                raise ValueError('unsupported timestamp')
            minute, second = match.groups()
            if minute is not None and float(second) >= 60:
                raise ValueError('invalid seconds')
            result = int(minute or 0) * 60 + float(second)
    else:
        raise ValueError('unsupported timestamp type')
    if not math.isfinite(result) or result < 0:
        raise ValueError('invalid timestamp')
    return result


def import_rows(rows, videos: Path, durations=None):
    rows = iter(rows)
    headers = next(rows, None)
    if headers is None or tuple(str(x).strip().lower() for x in headers[:3]) != HEADERS:
        raise ValueError('unexpected label headers')
    paths = [p for p in videos.rglob('*') if p.is_file() and p.suffix.lower() == '.mov']
    names = {}
    for path in paths:
        relative = _key(path.relative_to(videos).as_posix())
        for key in {relative, _key(str(Path(relative).with_suffix('')))}:
            if key in names and names[key] != relative:
                raise ValueError('ambiguous video filename')
            names[key] = relative
    annotations = []
    for row_number, row in enumerate(rows, start=2):
        if all(value is None for value in row):
            continue
        if len(row) < 3 or not isinstance(row[0], str):
            raise ValueError('invalid annotation row')
        key = _key(row[0].strip())
        if key not in names:
            raise ValueError('annotation filename not found')
        label_text = str(row[2]).strip().lower()
        if label_text not in ('нарушение', 'без нарушения'):
            raise ValueError('unrecognized binary label')
        label = int(label_text == 'нарушение')
        timestamp = parse_timestamp(row[1])
        if label and timestamp is None:
            raise ValueError('positive label without timestamp')
        if not label and timestamp is not None:
            raise ValueError('negative label with event timestamp needs review')
        if durations is not None:
            duration = durations.get(names[key])
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError('missing or invalid duration')
            if timestamp is not None and timestamp > duration:
                raise ValueError('annotation timestamp exceeds duration')
        annotations.append({'file':names[key], 'label':label, 'timestamp_sec':timestamp, 'source_row':row_number})
    result = inventory(videos, annotations)
    if result['annotated_file_count'] != result['file_count']:
        raise ValueError('not every video has a label')
    # A single point is not an exhaustive event annotation. Do not invent types.
    gold = [{'video_id':a['file'], 'verdict':'violation' if a['label'] else 'no_violation',
             'exhaustive':False,
             'events':([{'timecode_sec':a['timestamp_sec']}] if a['label'] else [])} for a in annotations]
    review = [{'video_id':a['file'], 'timestamp_sec':a['timestamp_sec'], 'event_type':None,
               'human_verified':False, 'exhaustive':False} for a in annotations if a['label']]
    counts = Counter(a['label'] for a in annotations)
    summary = {k:result[k] for k in ('file_count','unique_content_count','annotated_file_count')}
    summary.update(positive=counts[1], negative=counts[0],
                   conflicting_duplicate_groups=sum(g['annotation_conflict'] for g in result['duplicate_groups']),
                   timestamps_checked=durations is not None)
    return annotations, result, gold, review, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xlsx', type=Path)
    parser.add_argument('--videos', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--sheet', default='labels')
    parser.add_argument('--metadata', type=Path, help='Optional array of file/duration metadata')
    args = parser.parse_args(argv)
    from openpyxl import load_workbook
    durations = None
    if args.metadata:
        durations = {_key(row['file']):row['duration'] for row in json.loads(args.metadata.read_text(encoding='utf-8'))}
    workbook = load_workbook(args.xlsx, read_only=True, data_only=False)
    try:
        result = import_rows(workbook[args.sheet].values, args.videos, durations)
    finally:
        workbook.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in zip(('annotations.json','inventory.json','gold.json','review_queue.json','import_summary.json'), result):
        (args.output_dir/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result[-1]))


if __name__ == '__main__':
    main()
