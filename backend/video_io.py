"""Local, timestamp-aware video access shared by analyzers and evidence export.

Timestamps are presentation times in seconds, relative to the first video
stream's declared start time. FFmpeg performs its usual display-matrix rotation;
neither evidence extraction nor validation adds crops, overlays, or audio work.
"""

from __future__ import annotations

import bisect
import ctypes
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from PIL import Image


class VideoError(ValueError):
    """The uploaded file is not a usable video or could not be decoded."""


class VideoDeadlineExceeded(TimeoutError):
    """No video operation may continue beyond the job's absolute deadline."""


@dataclass(frozen=True)
class VideoMetadata:
    duration_sec: float
    width: int
    height: int
    frame_pts: tuple[float, ...]
    start_time_sec: float = 0.0
    rotation_degrees: float = 0.0


@dataclass(frozen=True)
class VideoFrame:
    timestamp_sec: float
    jpeg_bytes: bytes
    width: int
    height: int


def _number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


class _WindowsChildJob:
    """Close-on-owner-exit job: killing an ML process also kills its FFmpeg.

    Windows process groups alone do not kill descendants. Modern Windows allows
    nested jobs, so this also works inside a supervisor-owned job object.
    """

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise VideoError("Cannot create decoder process containment")
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            kernel.CloseHandle(handle)
            raise VideoError("Cannot configure decoder process containment")
        if not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):  # type: ignore[attr-defined]
            kernel.CloseHandle(handle)
            raise VideoError("Cannot contain decoder process")
        self.handle = handle

    def close(self) -> None:
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _linux_parent_guard(parent_pid: int):
    # The supervisor also kills the owner's process group. PDEATHSIG covers an
    # unexpected death of the owner before its supervisor can run cleanup.
    library = ctypes.CDLL(None, use_errno=True)

    def guard() -> None:
        if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
            os._exit(127)

    return guard


class VideoIO:
    def __init__(
        self,
        path: Path,
        *,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        deadline_at: float,
    ) -> None:
        self.path = Path(path).resolve()
        self.ffmpeg_bin = str(ffmpeg_bin)
        self.ffprobe_bin = str(ffprobe_bin)
        self.deadline_at = float(deadline_at)
        if not math.isfinite(self.deadline_at):
            raise ValueError("deadline_at must be a finite UTC Unix timestamp")
        self._metadata: VideoMetadata | None = None
        self._validated = False
        self._declared_frames: int | None = None
        self._frames_cache: dict[int, VideoFrame] = {}

    def _remaining(self) -> float:
        remaining = self.deadline_at - time.time()
        if remaining <= 0:
            raise VideoDeadlineExceeded("Video operation exceeded the job deadline")
        return remaining

    def _run(self, arguments: list[str]) -> bytes:
        self._remaining()
        if not self.path.is_file():
            raise VideoError("Video file is missing")
        extra: dict = {}
        if os.name == "nt":
            extra["creationflags"] = subprocess.CREATE_NO_WINDOW
        elif sys.platform.startswith("linux"):
            extra["preexec_fn"] = _linux_parent_guard(os.getpid())
        process = None
        containment = None
        try:
            process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **extra)
            containment = _WindowsChildJob(process)
            stdout, stderr = process.communicate(timeout=self._remaining())
            self._remaining()
            if process.returncode != 0 or stderr.strip():
                raise VideoError("Video stream could not be decoded completely")
            return stdout
        except subprocess.TimeoutExpired as exc:
            raise VideoDeadlineExceeded("Decoder exceeded the job deadline") from exc
        except OSError as exc:
            raise VideoError("Local FFmpeg or ffprobe is unavailable") from exc
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            if containment is not None:
                containment.close()

    def probe(self) -> VideoMetadata:
        self._remaining()
        if self._metadata is not None:
            return self._metadata
        output = self._run([
            self.ffprobe_bin, "-v", "error", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "mov", "-enable_drefs", "0", "-use_absolute_path", "0",
            "-err_detect", "explode", "-select_streams", "v:0",
            "-show_streams", "-show_frames", "-show_entries",
            "stream=index,width,height,start_time,duration,nb_frames:stream_tags=rotate:stream_side_data=rotation:frame=best_effort_timestamp_time,pts_time,duration_time,pkt_duration_time",
            "-of", "json", str(self.path),
        ])
        try:
            data = json.loads(output)
            stream = data["streams"][0]
            frames = data["frames"]
            width, height = int(stream["width"]), int(stream["height"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VideoError("No decodable video stream was found") from exc
        if not frames or min(width, height) < 1 or max(width, height) > 16384:
            raise VideoError("Video dimensions or frame count are unsupported")
        times = [_number(frame.get("best_effort_timestamp_time", frame.get("pts_time"))) for frame in frames]
        if any(value is None for value in times):
            raise VideoError("Video frames must have presentation timestamps")
        raw_pts = tuple(float(value) for value in times)  # type: ignore[arg-type]
        start = _number(stream.get("start_time"))
        start = raw_pts[0] if start is None else start
        points = tuple(value - start for value in raw_pts)
        if any(value < -1e-6 for value in points) or any(b < a for a, b in zip(points, points[1:])):
            raise VideoError("Video presentation timestamps are inconsistent")
        points = tuple(max(0.0, value) for value in points)
        last_duration = _number(frames[-1].get("duration_time", frames[-1].get("pkt_duration_time")))
        if last_duration is None or last_duration <= 0:
            last_duration = points[-1] - points[-2] if len(points) > 1 else 0.0
        duration = _number(stream.get("duration"))
        duration = max(duration or 0.0, points[-1] + last_duration)
        if duration <= 0:
            raise VideoError("Video duration is unavailable")
        declared = _number(stream.get("nb_frames"))
        self._declared_frames = int(declared) if declared is not None and declared > 0 else None
        if self._declared_frames is not None and len(points) != self._declared_frames:
            raise VideoError("Video is incomplete: declared and decoded frame counts differ")
        rotation = _number(stream.get("tags", {}).get("rotate")) or 0.0
        for item in stream.get("side_data_list", []):
            rotation = _number(item.get("rotation")) if "rotation" in item else rotation
        rotation = rotation or 0.0
        if abs(rotation % 180 - 90) < 1e-6:
            width, height = height, width
        self._metadata = VideoMetadata(duration, width, height, points, start, rotation)
        return self._metadata

    def validate(self) -> VideoMetadata:
        metadata = self.probe()
        if not self._validated:
            self._run([
                self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-xerror", "-err_detect", "explode", "-protocol_whitelist", "file,pipe",
                "-format_whitelist", "mov", "-enable_drefs", "0", "-use_absolute_path", "0",
                "-i", str(self.path), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-fps_mode", "passthrough", "-f", "null", "-",
            ])
            self._validated = True
        return metadata

    def _nearest_index(self, timestamp_sec: float) -> int:
        metadata = self.probe()
        if not math.isfinite(timestamp_sec) or timestamp_sec < 0 or timestamp_sec > metadata.duration_sec + 1e-6:
            raise ValueError("Requested timestamp must lie within the video duration")
        index = bisect.bisect_left(metadata.frame_pts, timestamp_sec)
        if index == len(metadata.frame_pts):
            return index - 1
        if index and timestamp_sec - metadata.frame_pts[index - 1] <= metadata.frame_pts[index] - timestamp_sec:
            return index - 1
        return index

    def _extract(self, indices: list[int]) -> list[VideoFrame]:
        metadata = self.probe()
        missing = sorted(set(index for index in indices if index not in self._frames_cache))
        if missing:
            # Frame ordinal is used only to select the frame whose actual PTS was
            # enumerated by ffprobe; ordinal / assumed FPS is never a timecode.
            expression = "+".join(f"eq(n\\,{index})" for index in missing)
            output = self._run([
                self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-xerror", "-err_detect", "explode", "-protocol_whitelist", "file,pipe",
                "-format_whitelist", "mov", "-enable_drefs", "0", "-use_absolute_path", "0",
                "-i", str(self.path), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", f"select={expression}", "-fps_mode", "passthrough",
                "-frames:v", str(len(missing)), "-c:v", "mjpeg", "-q:v", "2", "-f", "image2pipe", "pipe:1",
            ])
            images: list[bytes] = []
            offset = 0
            while offset < len(output):
                if output[offset:offset + 2] != b"\xff\xd8":
                    raise VideoError("Decoder returned invalid JPEG evidence")
                end = output.find(b"\xff\xd9", offset + 2)
                if end < 0:
                    raise VideoError("Decoder returned a truncated JPEG")
                images.append(output[offset:end + 2])
                offset = end + 2
            if len(images) != len(missing):
                raise VideoError("Decoder did not return every requested frame")
            for index, jpeg in zip(missing, images):
                try:
                    with Image.open(io.BytesIO(jpeg)) as image:
                        image.load()
                        if image.format != "JPEG":
                            raise VideoError("Evidence must be a JPEG image")
                        width, height = image.size
                except (OSError, ValueError) as exc:
                    raise VideoError("Evidence JPEG could not be decoded") from exc
                self._frames_cache[index] = VideoFrame(metadata.frame_pts[index], jpeg, width, height)
        self._remaining()
        return [self._frames_cache[index] for index in indices]

    def frame_at(self, timestamp_sec: float) -> VideoFrame:
        """Return the nearest actual frame (ties choose the earlier frame)."""
        return self._extract([self._nearest_index(float(timestamp_sec))])[0]

    def frames_at(self, timestamps_sec: list[float]) -> list[VideoFrame]:
        """Resolve up to 256 requested times in one batch, preserving input order.

        Returned times are canonical PTS, not the requested times. The same
        original JPEG cache is used by inference and final evidence extraction.
        """
        if not isinstance(timestamps_sec, (list, tuple)) or len(timestamps_sec) > 256:
            raise ValueError("Request at most 256 timestamps per batch")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) for t in timestamps_sec):
            raise ValueError("Timestamps must be numbers, not booleans")
        return self._extract([self._nearest_index(float(t)) for t in timestamps_sec])

    def iter_frames(self, interval_sec: float, batch_size: int = 256):
        """Yield interval samples across the entire video using bounded batches.

        This is a batched iterator, not a second decoder. Each batch requires at
        most one FFmpeg extraction. Unlike sample_frames, 256 is a batch cap,
        not a whole-video cap. Resize inference images after receiving them;
        the shared cache preserves original evidence JPEGs.
        """
        if isinstance(interval_sec, bool) or not isinstance(interval_sec, (int, float)) or not math.isfinite(interval_sec) or interval_sec < 1e-6:
            raise ValueError("interval_sec must be finite and at least 1e-6")
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        metadata = self.probe()
        pending = []
        # Iterate actual frames, avoiding unbounded target loops for tiny intervals.
        next_target = 0.0
        for index, pts in enumerate(metadata.frame_pts):
            if pts + 1e-6 < next_target:
                continue
            # First actual frame at or after a scheduled point. On VFR input,
            # elapsed empty slots collapse to this actual frame, never fake PTS.
            pending.append(index)
            # Advance beyond this frame; never emit repeated copies of a frame.
            next_target = (math.floor((pts + 1e-6) / interval_sec) + 1) * interval_sec
            if len(pending) == batch_size:
                for frame in self._extract(pending):
                    self._remaining()
                    yield frame
                pending = []
        if pending:
            for frame in self._extract(pending):
                self._remaining()
                yield frame

    def sample_frames(self, interval_sec: float, max_frames: int = 256) -> list[VideoFrame]:
        """Sample from zero at the requested interval, capped at max_frames.

        The cap limits sampling; reaching it does not prove the rest of the video
        was analyzed. Callers must select a suitable interval for full coverage.
        """
        if not math.isfinite(interval_sec) or interval_sec <= 0:
            raise ValueError("interval_sec must be finite and positive")
        if not isinstance(max_frames, int) or not 1 <= max_frames <= 256:
            raise ValueError("max_frames must be between 1 and 256")
        metadata = self.probe()
        indices: list[int] = []
        for ordinal in range(max_frames):
            target = ordinal * interval_sec
            if target > metadata.frame_pts[-1] + 1e-6:
                break
            index = self._nearest_index(target)
            if not indices or indices[-1] != index:
                indices.append(index)
        return self._extract(indices)
