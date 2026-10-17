"""Public VideoIO batches used by the local observer, synthetic input only."""
import time
import pytest
from backend.video_io import VideoIO
from backend.tests.video_fixtures import ffmpeg_tools, make_video


def test_full_video_iterator_exceeds_old_256_cap_and_reuses_evidence(tmp_path):
    ffmpeg, ffprobe = ffmpeg_tools()
    path = make_video(tmp_path/'synthetic.mov', ffmpeg, duration=3, rate=100)
    video = VideoIO(path, ffmpeg_bin=ffmpeg, ffprobe_bin=ffprobe, deadline_at=time.time()+60)
    meta = video.validate()
    frames = list(video.iter_frames(.01, batch_size=64))
    assert len(frames) == len(meta.frame_pts) == 300
    assert [f.timestamp_sec for f in frames] == list(meta.frame_pts)
    selected = video.frames_at([meta.frame_pts[-1], 0, 0])
    assert selected[0].jpeg_bytes == frames[-1].jpeg_bytes
    assert selected[1] is selected[2] and selected[1].jpeg_bytes == frames[0].jpeg_bytes


@pytest.mark.parametrize('interval', [True,0,-1,float('nan'),float('inf'),1e-320])
def test_invalid_sampling_interval_before_media_read(tmp_path, interval):
    with pytest.raises(ValueError):
        list(VideoIO(tmp_path/'not-opened.mov', deadline_at=time.time()+60).iter_frames(interval))


@pytest.mark.parametrize('points', [[True],[0]*257,'0'])
def test_invalid_batch_before_media_read(tmp_path, points):
    with pytest.raises(ValueError):
        VideoIO(tmp_path/'not-opened.mov', deadline_at=time.time()+60).frames_at(points)


def test_decimal_grid_does_not_add_frame_after_exact_boundary(tmp_path, monkeypatch):
    from backend.video_io import VideoMetadata
    video = VideoIO(tmp_path/'unused.mov', deadline_at=time.time()+60)
    pts = (0., .1, .2, .3, .333333, .4, .5)
    monkeypatch.setattr(video, 'probe', lambda: VideoMetadata(.6, 1, 1, pts))
    monkeypatch.setattr(video, '_extract', lambda indices: [pts[i] for i in indices])
    assert list(video.iter_frames(.1)) == [0., .1, .2, .3, .4, .5]


def test_deadline_checked_between_cached_frames(tmp_path, monkeypatch):
    from backend.video_io import VideoMetadata, VideoDeadlineExceeded
    video = VideoIO(tmp_path/'unused.mov', deadline_at=time.time()+60)
    monkeypatch.setattr(video, 'probe', lambda: VideoMetadata(1., 1, 1, (0., .5)))
    monkeypatch.setattr(video, '_extract', lambda indices: indices)
    iterator = video.iter_frames(.5)
    assert next(iterator) == 0
    video.deadline_at = time.time()-1
    with pytest.raises(VideoDeadlineExceeded):
        next(iterator)
