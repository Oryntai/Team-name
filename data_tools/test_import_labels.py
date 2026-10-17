from datetime import time
import pytest

from data_tools.import_labels import HEADERS, import_rows, parse_timestamp


@pytest.mark.parametrize('value, expected', [
    ('1минута 8 секунда', 68), ('22 секунда', 22), ('1:08',68),
    ('2 мин 3,5 сек',123.5), (12.5,12.5), (time(0,1,8),68), (None,None),
])
def test_timestamp_formats(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize('value', [True, -1, float('nan'), '1:88', '1 минута 80 секунда', '=SUM(A1)', 'some text'])
def test_invalid_or_formula_timestamp(value):
    with pytest.raises(ValueError):
        parse_timestamp(value)


def test_all_labels_import_without_guessing_type_and_duplicate_conflict(tmp_path):
    (tmp_path/'X-1.mov').write_bytes(b'syntheticduplicate')
    (tmp_path/'Х-2.mov').write_bytes(b'syntheticduplicate')
    rows=[HEADERS, ('X-1','5 секунда','нарушение'),('Х-2',None,'без нарушения')]
    annotations, inventory, gold, review, summary = import_rows(rows,tmp_path,{'X-1.mov':8,'Х-2.mov':8})
    assert summary == {'file_count':2,'unique_content_count':1,'annotated_file_count':2,
                       'positive':1,'negative':1,'conflicting_duplicate_groups':1,'timestamps_checked':True}
    assert gold[0]['events']==[{'timecode_sec':5.0}]
    assert not gold[0]['exhaustive'] and not gold[1]['exhaustive']
    assert review[0]['event_type'] is None and not review[0]['human_verified']
    assert inventory['duplicate_groups'][0]['review_required']


def test_missing_label_and_latin_cyrillic_mismatch_fail(tmp_path):
    (tmp_path/'Х-1.mov').write_bytes(b'synthetic')
    with pytest.raises(ValueError):
        import_rows([HEADERS,('X-1','5 секунда','нарушение')],tmp_path)
    with pytest.raises(ValueError):
        import_rows([HEADERS],tmp_path)


def test_timestamp_outside_duration_fails(tmp_path):
    (tmp_path/'X-1.mov').write_bytes(b'synthetic')
    with pytest.raises(ValueError):
        import_rows([HEADERS,('X-1','5 секунда','нарушение')],tmp_path,{'X-1.mov':4})


def test_duplicate_rows_fail(tmp_path):
    (tmp_path/'X-1.mov').write_bytes(b'synthetic')
    row=('X-1','5 секунда','нарушение')
    with pytest.raises(ValueError):
        import_rows([HEADERS,row,row],tmp_path)
