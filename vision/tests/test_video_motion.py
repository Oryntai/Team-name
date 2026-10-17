from conftest import DURATION, EVENT
from forte_vision import motion
from forte_vision.video import frame_id, grab_at, probe, sample


def test_probe_uses_real_timeline(synthetic_video):
    info = probe(synthetic_video)
    assert abs(info.duration_sec - DURATION) < 0.1
    assert info.width == 464 and info.height == 848
    assert abs(info.fps - 30) < 0.5


def test_sample_step_and_size(synthetic_video):
    frames = list(sample(synthetic_video, fps=2.0, size=(116, 212)))
    assert 38 <= len(frames) <= 41
    assert frames[0][1].size == (116, 212)
    ts = [t for t, _ in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_grab_at_returns_actual_pts(synthetic_video):
    got = grab_at(synthetic_video, [5.01, 5.02, 30.0])
    ts = [t for t, _ in got]
    assert ts[0] >= 5.01 - 1e-6
    assert ts[-1] <= DURATION  # время за концом ролика -> последний кадр
    assert len(ts) == len(set(ts))


def test_frame_id_stable():
    assert frame_id(9.0333) == frame_id(9.0333) == "pts_000009033"


def test_motion_peak_inside_event(synthetic_video):
    times, energy = motion.motion_energy(synthetic_video, fps=5.0)
    peaks = motion.find_peaks(times, energy, DURATION, top_k=3)
    assert peaks, "движение в событии должно дать пик"
    assert any(EVENT[0] - 0.5 <= p.main_sec <= EVENT[1] + 0.5 for p in peaks)
    for p in peaks:
        assert 0 <= p.start_sec <= p.main_sec <= p.end_sec <= DURATION
