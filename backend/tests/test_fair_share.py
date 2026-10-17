"""Ролики, загруженные разом, делят одно окно в 5 минут: worker обслуживает очередь по одному."""

import time

import pytest

from backend.store import Store
from backend.worker import SHARE_FLOOR_SEC, SHARE_TAIL_SEC, fair_share_end


def test_lone_job_keeps_its_whole_budget():
    assert fair_share_end(1000.0, 1300.0, []) == 1300.0


def test_burst_of_five_splits_the_window_evenly():
    now, hard = 1000.0, 1300.0
    queued = [1300.0, 1300.5, 1301.0, 1301.5]  # остальные четыре загружены в ту же секунду
    end = fair_share_end(now, hard, queued)
    # доля первого: самый тесный префикс очереди — все пять до последнего срока
    assert end == pytest.approx(min(now + (d - SHARE_TAIL_SEC - now) / (k + 1)
                                    for k, d in enumerate(queued, start=1)))
    assert end - now == pytest.approx((1301.5 - SHARE_TAIL_SEC - now) / 5, abs=0.5)


def test_simulated_burst_every_job_finishes_before_its_deadline():
    # пять роликов, каждому для полного разбора нужно 104 с (Х-14), все загружены в 0 с
    need, deadlines = 104.0, [300.0] * 5
    now = 0.0
    for i, hard in enumerate(deadlines):
        end = fair_share_end(now, hard, deadlines[i + 1:])
        now = min(now + need, end, hard - 20.0) + SHARE_TAIL_SEC  # разбор (анализатор держит 20 с запаса) + кадры
        assert now <= hard + 1e-6, f"ролик {i + 1} не успел: {now:.0f} с"


def test_later_upload_leaves_the_running_job_more_time():
    now = 1040.0
    early = fair_share_end(now, 1300.0, [1301.0])  # второй загружен почти одновременно
    late = fair_share_end(now, 1300.0, [1400.0])  # второй загружен через 100 с
    assert late > early


def test_share_never_drops_below_floor_but_never_passes_the_hard_deadline():
    now = 1000.0
    assert fair_share_end(now, 1300.0, [1001.0] * 10) == now + SHARE_FLOOR_SEC
    assert fair_share_end(now, now + 5.0, [now + 6.0]) == now + 5.0


def test_store_lists_only_waiting_jobs_earliest_first(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    store.initialize()
    now = time.time()
    for name, arrival in (("b", now + 2), ("a", now + 1), ("c", now + 3)):
        store.enqueue(name, tmp_path / f"{name}.mov", arrival, arrival + 300, "standard", 20)
    assert store.queued_deadlines() == sorted([now + 301, now + 302, now + 303])
    store.claim()  # "a" ушёл в работу и больше не ждёт
    assert store.queued_deadlines() == [now + 302, now + 303]


def test_recomputing_later_moves_the_end_so_the_analyzer_keeps_the_first_one():
    # «пятая часть остатка» через 40 с снова даёт почти столько же: конец уезжает вперёд. Поэтому анализатор
    # берёт минимум из пересчётов (forte_vision.anchored.analyzer, share_end), а не последний пересчёт.
    queued = [300.0] * 4
    first = fair_share_end(15.0, 300.0, queued)
    later = fair_share_end(55.0, 300.0, queued)
    assert later > first
    assert min(first, later) == first
