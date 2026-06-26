"""
Handle video uploads for visual analysis.

Claude's vision can only read still images, so videos are processed by sampling
a handful of evenly-spaced frames with ffmpeg and sending those frames to Claude
as images. Requires the ``ffmpeg`` (and ``ffprobe``) binaries on PATH.
"""

import asyncio
import base64
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import structlog
from telegram import Video, VideoNote

from src.config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedVideo:
    """Result of sampling frames from a video."""

    prompt: str
    frames: List[Dict[str, str]]  # [{"data": <base64>, "media_type": "image/jpeg"}]
    size: int
    metadata: Dict[str, object] = field(default_factory=dict)


class VideoUnavailableError(RuntimeError):
    """Raised when video processing cannot be performed (ffmpeg missing)."""


class VideoHandler:
    """Sample frames from uploaded videos for Claude vision analysis."""

    def __init__(self, config: Settings):
        self.config = config
        self.temp_dir = Path(tempfile.gettempdir()) / "claude_bot_videos"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = max(1, getattr(config, "video_frame_count", 6))
        self.max_size_bytes = (
            getattr(config, "video_max_file_size_mb", 50) * 1024 * 1024
        )
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ffprobe_path = shutil.which("ffprobe")

    @property
    def available(self) -> bool:
        """Whether the required binaries are present."""
        return self.ffmpeg_path is not None

    async def process_video(
        self,
        video: Union[Video, VideoNote],
        caption: Optional[str] = None,
    ) -> ProcessedVideo:
        """Download a video, sample frames, and build a Claude prompt."""
        if not self.available:
            raise VideoUnavailableError(
                "Video analysis needs ffmpeg, which is not installed on the server."
            )

        file_size = getattr(video, "file_size", None)
        if file_size and file_size > self.max_size_bytes:
            limit_mb = self.max_size_bytes // (1024 * 1024)
            raise VideoUnavailableError(
                f"Video too large ({file_size // (1024 * 1024)}MB, max {limit_mb}MB)."
            )

        tg_file = await video.get_file()
        video_path = self.temp_dir / f"{uuid.uuid4().hex}.mp4"
        await tg_file.download_to_drive(custom_path=str(video_path))

        try:
            duration = await self._probe_duration(video_path)
            frame_paths = await self._extract_frames(video_path, duration)
            frames: List[Dict[str, str]] = []
            for fp in frame_paths:
                try:
                    data = base64.b64encode(fp.read_bytes()).decode("utf-8")
                    frames.append({"data": data, "media_type": "image/jpeg"})
                finally:
                    fp.unlink(missing_ok=True)
        finally:
            video_path.unlink(missing_ok=True)

        if not frames:
            raise VideoUnavailableError("Could not extract any frames from the video.")

        return ProcessedVideo(
            prompt=self._build_prompt(caption, len(frames), duration),
            frames=frames,
            size=file_size or 0,
            metadata={
                "frame_count": len(frames),
                "duration_seconds": round(duration, 1) if duration else None,
                "has_caption": caption is not None,
            },
        )

    async def _probe_duration(self, video_path: Path) -> float:
        """Return video duration in seconds (0.0 if it cannot be determined)."""
        if not self.ffprobe_path:
            return 0.0
        proc = await asyncio.create_subprocess_exec(
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        try:
            return float(stdout.decode().strip())
        except (ValueError, AttributeError):
            return 0.0

    async def _extract_frames(self, video_path: Path, duration: float) -> List[Path]:
        """Extract evenly-spaced frames, scaled down, as JPEG files."""
        outputs: List[Path] = []

        if duration and duration > 0:
            # Sample at the midpoint of each of N equal segments.
            timestamps = [
                duration * (i + 0.5) / self.frame_count for i in range(self.frame_count)
            ]
        else:
            # Unknown duration: grab a single frame from the start.
            timestamps = [0.0]

        for idx, ts in enumerate(timestamps):
            out_path = self.temp_dir / f"{uuid.uuid4().hex}_{idx}.jpg"
            proc = await asyncio.create_subprocess_exec(
                self.ffmpeg_path,
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(768,iw)':-2",
                "-q:v",
                "3",
                "-y",
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if out_path.exists() and out_path.stat().st_size > 0:
                outputs.append(out_path)
            else:
                out_path.unlink(missing_ok=True)

        return outputs

    def _build_prompt(
        self, caption: Optional[str], frame_count: int, duration: float
    ) -> str:
        """Build the prompt that accompanies the sampled frames."""
        dur_text = f" (~{round(duration)}s long)" if duration else ""
        prompt = (
            f"I'm sharing a video{dur_text} with you. Since you can't watch video "
            f"directly, here are {frame_count} still frames sampled evenly across "
            "its duration, in chronological order. Use them to understand what the "
            "video shows and describe what is happening.\n\n"
        )
        if caption:
            prompt += f"Specific request: {caption}"
        return prompt
