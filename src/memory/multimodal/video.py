"""
Video processing for PLM multi-modal support.

Extracts memories from video including:
- Frame analysis (scenes, objects, text)
- Audio track transcription
- Timeline events
- Key moments detection
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audio import AudioMemory, AudioProcessor
from .image import ImageMemory, ImageProcessor


@dataclass
class VideoFrame:
    """A key frame from a video."""

    timestamp: float  # seconds
    image_memory: ImageMemory | None = None
    is_scene_change: bool = False
    motion_score: float = 0.0


@dataclass
class VideoMemory:
    """A memory extracted from a video."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_path: str = ""
    timestamp: datetime | None = None

    # Video metadata
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""

    # Extracted content
    audio_memory: AudioMemory | None = None
    key_frames: list[VideoFrame] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)

    # Analysis
    video_type: str = ""  # vlog, meeting, presentation, movie, etc.
    topics: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)

    # Summary
    summary: str = ""
    chapters: list[dict[str, Any]] = field(default_factory=list)

    def to_text(self) -> str:
        """Convert video memory to searchable text."""
        parts = []

        if self.video_type:
            parts.append(f"Video: {self.video_type}")

        if self.summary:
            parts.append(f"Summary: {self.summary}")

        if self.audio_memory and self.audio_memory.full_transcript:
            parts.append(f"Transcript: {self.audio_memory.full_transcript[:500]}...")

        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")

        if self.people:
            parts.append(f"People: {', '.join(self.people)}")

        if self.chapters:
            chapter_list = [c.get("title", "") for c in self.chapters]
            parts.append(f"Chapters: {', '.join(chapter_list)}")

        duration = str(timedelta(seconds=int(self.duration_seconds)))
        parts.append(f"Duration: {duration}")

        return "\n".join(parts)


class VideoProcessor:
    """
    Processes videos to extract memories.

    Combines image and audio processing with video-specific analysis:
    - Scene detection
    - Key frame extraction
    - Timeline segmentation
    - Content summarization
    """

    def __init__(
        self,
        extract_audio: bool = True,
        extract_frames: bool = True,
        frames_per_minute: int = 2,
        detect_scenes: bool = True,
    ):
        self.extract_audio = extract_audio
        self.extract_frames = extract_frames
        self.frames_per_minute = frames_per_minute
        self.detect_scenes = detect_scenes

        self._image_processor = ImageProcessor()
        self._audio_processor = AudioProcessor()

    async def process_video(
        self,
        video_path: str | Path,
    ) -> VideoMemory:
        """
        Process a video file and extract memory content.

        Args:
            video_path: Path to video file
        """
        path = Path(video_path)
        memory = VideoMemory(source_path=str(path))

        # Get video metadata
        metadata = await self._get_video_metadata(path)
        memory.duration_seconds = metadata.get("duration", 0)
        memory.width = metadata.get("width", 0)
        memory.height = metadata.get("height", 0)
        memory.fps = metadata.get("fps", 0)
        memory.codec = metadata.get("codec", "")
        memory.timestamp = metadata.get("timestamp")

        # Extract and process audio track
        if self.extract_audio:
            audio_path = await self._extract_audio(path)
            if audio_path:
                memory.audio_memory = await self._audio_processor.process_audio(
                    audio_path
                )
                # Clean up temp audio file
                Path(audio_path).unlink(missing_ok=True)

        # Extract key frames
        if self.extract_frames:
            memory.key_frames = await self._extract_key_frames(path)

        # Detect scenes
        if self.detect_scenes:
            memory.scenes = await self._detect_scenes(path)

        # Determine video type
        memory.video_type = await self._detect_video_type(memory)

        # Generate chapters from scenes
        memory.chapters = self._generate_chapters(memory)

        # Extract topics from audio and visual content
        memory.topics = await self._extract_topics(memory)

        # Generate summary
        memory.summary = await self._generate_summary(memory)

        return memory

    async def _get_video_metadata(self, path: Path) -> dict[str, Any]:
        """Get video file metadata."""
        metadata = {}

        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if metadata["fps"] > 0:
                    metadata["duration"] = frame_count / metadata["fps"]
                metadata["codec"] = int(cap.get(cv2.CAP_PROP_FOURCC))
                cap.release()

        except ImportError:
            # Try ffprobe
            try:
                import json
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", str(path)
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    data = json.loads(result.stdout)

                    for stream in data.get("streams", []):
                        if stream.get("codec_type") == "video":
                            metadata["width"] = stream.get("width", 0)
                            metadata["height"] = stream.get("height", 0)
                            fps_str = stream.get("r_frame_rate", "0/1")
                            if "/" in fps_str:
                                num, den = fps_str.split("/")
                                metadata["fps"] = int(num) / int(den) if int(den) > 0 else 0
                            metadata["codec"] = stream.get("codec_name", "")
                            break

                    format_info = data.get("format", {})
                    metadata["duration"] = float(format_info.get("duration", 0))

            except Exception:
                pass

        except Exception:
            pass

        # Get file modification time
        try:
            stat = path.stat()
            metadata["timestamp"] = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            pass

        return metadata

    async def _extract_audio(self, path: Path) -> str | None:
        """Extract audio track from video."""
        try:
            import subprocess

            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            # Use ffmpeg to extract audio
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(path),
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-y", audio_path
                ],
                capture_output=True,
            )

            if result.returncode == 0 and Path(audio_path).exists():
                return audio_path

        except Exception:
            pass

        return None

    async def _extract_key_frames(self, path: Path) -> list[VideoFrame]:
        """Extract key frames from video."""
        frames = []

        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return frames

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames / fps if fps > 0 else 0

            # Calculate frame interval
            if self.frames_per_minute > 0:
                interval_seconds = 60 / self.frames_per_minute
                interval_frames = int(interval_seconds * fps)
            else:
                interval_frames = int(fps * 30)  # Every 30 seconds

            frame_idx = 0
            prev_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval_frames == 0:
                    timestamp = frame_idx / fps

                    # Calculate motion score
                    motion_score = 0.0
                    if prev_frame is not None:
                        diff = cv2.absdiff(frame, prev_frame)
                        motion_score = diff.mean()

                    # Save frame to temp file for processing
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as tmp:
                        cv2.imwrite(tmp.name, frame)

                        # Process frame
                        try:
                            image_memory = await self._image_processor.process_image(
                                tmp.name,
                                extract_text=True,
                                detect_objects=True,
                                generate_description=True,
                            )

                            frames.append(VideoFrame(
                                timestamp=timestamp,
                                image_memory=image_memory,
                                motion_score=motion_score,
                            ))
                        finally:
                            Path(tmp.name).unlink(missing_ok=True)

                    prev_frame = frame.copy()

                frame_idx += 1

            cap.release()

        except ImportError:
            pass
        except Exception:
            pass

        return frames

    async def _detect_scenes(self, path: Path) -> list[dict[str, Any]]:
        """Detect scene changes in video."""
        scenes = []

        try:
            # Try using scenedetect library
            from scenedetect import ContentDetector, detect

            scene_list = detect(str(path), ContentDetector())

            for i, scene in enumerate(scene_list):
                scenes.append({
                    "index": i,
                    "start": scene[0].get_seconds(),
                    "end": scene[1].get_seconds(),
                    "duration": scene[1].get_seconds() - scene[0].get_seconds(),
                })

        except ImportError:
            # Fallback: detect scene changes using frame differences
            try:
                import cv2

                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return scenes

                fps = cap.get(cv2.CAP_PROP_FPS)
                threshold = 30.0  # Scene change threshold

                frame_idx = 0
                prev_frame = None
                scene_start = 0
                scene_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if prev_frame is not None:
                        diff = cv2.absdiff(frame, prev_frame).mean()
                        if diff > threshold:
                            # Scene change detected
                            timestamp = frame_idx / fps
                            scenes.append({
                                "index": scene_idx,
                                "start": scene_start,
                                "end": timestamp,
                                "duration": timestamp - scene_start,
                            })
                            scene_start = timestamp
                            scene_idx += 1

                    prev_frame = frame.copy()
                    frame_idx += 1

                # Add final scene
                final_timestamp = frame_idx / fps
                if scene_start < final_timestamp:
                    scenes.append({
                        "index": scene_idx,
                        "start": scene_start,
                        "end": final_timestamp,
                        "duration": final_timestamp - scene_start,
                    })

                cap.release()

            except Exception:
                pass

        except Exception:
            pass

        return scenes

    async def _detect_video_type(self, memory: VideoMemory) -> str:
        """Detect the type of video content."""
        # Use heuristics based on extracted content
        has_faces = False
        has_slides = False
        has_outdoor = False

        for frame in memory.key_frames:
            if frame.image_memory:
                if frame.image_memory.faces:
                    has_faces = True
                if frame.image_memory.ocr_text:
                    # Lots of text suggests slides/presentation
                    if len(frame.image_memory.ocr_text) > 100:
                        has_slides = True
                if frame.image_memory.scene_type == "outdoor":
                    has_outdoor = True

        # Determine type
        if has_slides:
            return "presentation"
        elif memory.audio_memory and len(memory.audio_memory.speakers or []) > 2:
            return "meeting"
        elif has_faces and memory.duration_seconds < 600:
            return "vlog"
        elif has_outdoor:
            return "travel"
        else:
            return "video"

    def _generate_chapters(self, memory: VideoMemory) -> list[dict[str, Any]]:
        """Generate chapter markers from scenes."""
        chapters = []

        for i, scene in enumerate(memory.scenes):
            # Try to get description from key frames in this scene
            description = ""
            for frame in memory.key_frames:
                if scene["start"] <= frame.timestamp <= scene["end"]:
                    if frame.image_memory and frame.image_memory.description:
                        description = frame.image_memory.description
                        break

            chapters.append({
                "index": i,
                "start": scene["start"],
                "end": scene["end"],
                "title": f"Scene {i + 1}",
                "description": description,
            })

        return chapters

    async def _extract_topics(self, memory: VideoMemory) -> list[str]:
        """Extract topics from video content."""
        topics = set()

        # From audio
        if memory.audio_memory and memory.audio_memory.topics:
            topics.update(memory.audio_memory.topics)

        # From visual content
        for frame in memory.key_frames:
            if frame.image_memory:
                # Add scene type as topic
                if frame.image_memory.scene_type:
                    topics.add(frame.image_memory.scene_type)
                # Add detected objects as topics
                for obj in frame.image_memory.objects[:3]:
                    topics.add(obj)

        return list(topics)

    async def _generate_summary(self, memory: VideoMemory) -> str:
        """Generate a summary of the video."""
        parts = []

        if memory.video_type:
            parts.append(f"A {memory.video_type} video")

        duration = str(timedelta(seconds=int(memory.duration_seconds)))
        parts.append(f"({duration} long)")

        if memory.topics:
            parts.append(f"about {', '.join(memory.topics[:3])}")

        if memory.audio_memory and memory.audio_memory.speakers:
            speaker_count = len(memory.audio_memory.speakers)
            parts.append(f"featuring {speaker_count} speaker(s)")

        if len(memory.scenes) > 1:
            parts.append(f"with {len(memory.scenes)} scenes")

        return " ".join(parts) + "."

    async def process_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> AsyncIterator[VideoMemory]:
        """Process all video files in a directory."""
        path = Path(directory)
        if not path.exists():
            return

        extensions = extensions or [
            ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"
        ]
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in extensions:
                try:
                    memory = await self.process_video(file_path)
                    yield memory
                except Exception:
                    continue
