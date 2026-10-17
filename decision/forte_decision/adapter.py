"""Official response serializer; backend owns extraction and JPEG verification."""

from dataclasses import dataclass
import base64

from .contracts import number, text
from .engine import DecisionResult


@dataclass(frozen=True)
class VerifiedJPEG:
    """Backend assertion of a verified local frame, not an image decoder.

    The backend must decode JPEG bytes and verify source video, frame identity,
    and extraction timestamp before constructing this value. This class checks
    basic shape and markers only; those checks cannot establish image validity.
    """

    video_id: str
    frame_id: str
    pts_sec: float
    jpeg_bytes: bytes

    def __post_init__(self):
        text(self.video_id, "video_id")
        text(self.frame_id, "frame_id")
        if number(self.pts_sec, "pts_sec") < 0:
            raise ValueError("negative evidence PTS")
        if (
            not isinstance(self.jpeg_bytes, bytes)
            or not self.jpeg_bytes.startswith(b"\xff\xd8")
            or not self.jpeg_bytes.endswith(b"\xff\xd9")
        ):
            raise ValueError("expected backend-verified JPEG bytes")


def to_public_payload(
    result: DecisionResult,
    evidence_map: dict[str, VerifiedJPEG],
    *,
    job_id: str,
):
    """Serialize a complete result using explicit backend-verified evidence."""
    text(job_id, "job_id")
    text(result.video_id, "video_id")
    if result.status != "complete" or result.verdict not in (
        "violation",
        "no_violation",
    ):
        raise ValueError("incomplete analysis cannot serialize as final verdict")
    if (result.verdict == "violation") != bool(result.events):
        raise ValueError("verdict/event inconsistency")
    violations = []
    for event in result.events:
        if event.video_id != result.video_id:
            raise ValueError("event video mismatch")
        if not 1 <= len(event.evidence) <= 3:
            raise ValueError("each event needs 1–3 evidence frames")
        if len({ref.frame_id for ref in event.evidence}) != len(event.evidence):
            raise ValueError("duplicate event evidence frame")
        if (
            not event.start_sec
            <= number(event.timecode_sec, "timecode_sec")
            <= event.end_sec
        ):
            raise ValueError("invalid event anchor")
        evidence_frames = []
        for ref in event.evidence:
            try:
                image = evidence_map[ref.frame_id]
            except KeyError as error:
                raise ValueError("missing evidence frame") from error
            if not isinstance(image, VerifiedJPEG):
                raise ValueError("expected explicit VerifiedJPEG evidence")
            if (image.video_id, image.frame_id, image.pts_sec) != (
                result.video_id,
                ref.frame_id,
                ref.pts_sec,
            ):
                raise ValueError("evidence provenance mismatch")
            evidence_frames.append(
                {
                    "timestamp_sec": ref.pts_sec,
                    "image_base64": base64.b64encode(image.jpeg_bytes).decode("ascii"),
                }
            )
        violations.append(
            {
                "type": event.type,
                "timecode_sec": event.timecode_sec,
                "description": event.description,
                "evidence_frames": evidence_frames,
            }
        )
    return {
        "job_id": job_id,
        "status": "done",
        "verdict": result.verdict,
        "violations": violations,
    }
