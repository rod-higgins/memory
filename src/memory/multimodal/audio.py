"""
Audio processing for PLM multi-modal support.

Extracts memories from audio including:
- Speech-to-text transcription
- Speaker diarization
- Emotion detection
- Music/podcast identification
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class AudioSegment:
    """A segment of audio with transcription."""

    start_time: float  # seconds
    end_time: float
    text: str
    speaker: str | None = None
    confidence: float = 1.0
    language: str = "en"


@dataclass
class AudioMemory:
    """A memory extracted from audio."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_path: str = ""
    timestamp: datetime | None = None
    duration_seconds: float = 0.0

    # Transcription
    full_transcript: str = ""
    segments: list[AudioSegment] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)

    # Analysis
    language: str = "en"
    topics: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    audio_type: str = ""  # speech, music, podcast, meeting, voicemail

    # Metadata
    sample_rate: int = 0
    channels: int = 0

    def to_text(self) -> str:
        """Convert audio memory to searchable text."""
        parts = []

        if self.audio_type:
            parts.append(f"Audio: {self.audio_type}")

        if self.full_transcript:
            parts.append(f"Transcript: {self.full_transcript[:500]}...")

        if self.speakers:
            parts.append(f"Speakers: {', '.join(self.speakers)}")

        if self.topics:
            parts.append(f"Topics: {', '.join(self.topics)}")

        if self.duration_seconds:
            duration = str(timedelta(seconds=int(self.duration_seconds)))
            parts.append(f"Duration: {duration}")

        return "\n".join(parts)


class AudioProcessor:
    """
    Processes audio to extract memories.

    Supports multiple backends:
    - Whisper: OpenAI's speech recognition
    - Pyannote: Speaker diarization
    - Local models for offline processing
    """

    def __init__(
        self,
        model_size: str = "base",
        use_diarization: bool = True,
        device: str = "cpu",
    ):
        self.model_size = model_size
        self.use_diarization = use_diarization
        self.device = device

        self._whisper_model = None
        self._diarization_pipeline = None

    async def process_audio(
        self,
        audio_path: str | Path,
        transcribe: bool = True,
        diarize: bool = True,
        detect_type: bool = True,
    ) -> AudioMemory:
        """
        Process an audio file and extract memory content.

        Args:
            audio_path: Path to audio file
            transcribe: Whether to transcribe speech
            diarize: Whether to identify speakers
            detect_type: Whether to detect audio type
        """
        path = Path(audio_path)
        memory = AudioMemory(source_path=str(path))

        # Get audio metadata
        metadata = await self._get_audio_metadata(path)
        memory.duration_seconds = metadata.get("duration", 0)
        memory.sample_rate = metadata.get("sample_rate", 0)
        memory.channels = metadata.get("channels", 0)
        memory.timestamp = metadata.get("timestamp")

        # Detect audio type
        if detect_type:
            memory.audio_type = await self._detect_audio_type(path)

        # Transcribe
        if transcribe and memory.audio_type in ["speech", "podcast", "meeting", "voicemail", ""]:
            result = await self._transcribe(path)
            memory.full_transcript = result.get("text", "")
            memory.language = result.get("language", "en")
            memory.segments = result.get("segments", [])

        # Speaker diarization
        if diarize and self.use_diarization and memory.segments:
            memory.speakers = await self._diarize(path)
            memory.segments = await self._assign_speakers(
                memory.segments, path
            )

        # Extract topics and emotions from transcript
        if memory.full_transcript:
            analysis = await self._analyze_transcript(memory.full_transcript)
            memory.topics = analysis.get("topics", [])
            memory.emotions = analysis.get("emotions", [])

        return memory

    async def _get_audio_metadata(self, path: Path) -> dict[str, Any]:
        """Get audio file metadata."""
        metadata = {}

        try:
            import librosa

            # Load audio file
            y, sr = librosa.load(str(path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            metadata["duration"] = duration
            metadata["sample_rate"] = sr
            metadata["channels"] = 1 if len(y.shape) == 1 else y.shape[0]

        except ImportError:
            # Try with wave for basic WAV files
            try:
                import wave

                with wave.open(str(path), "rb") as wf:
                    metadata["sample_rate"] = wf.getframerate()
                    metadata["channels"] = wf.getnchannels()
                    frames = wf.getnframes()
                    metadata["duration"] = frames / float(wf.getframerate())
            except Exception:
                pass

        except Exception:
            pass

        # Get file modification time as timestamp
        try:
            stat = path.stat()
            metadata["timestamp"] = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            pass

        return metadata

    async def _detect_audio_type(self, path: Path) -> str:
        """Detect the type of audio content."""
        try:
            import librosa
            import numpy as np

            y, sr = librosa.load(str(path), sr=None, duration=30)

            # Calculate features
            # Check for speech vs music
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

            # Simple heuristics
            if tempo > 60 and spectral_centroid > 2000:
                return "music"
            elif zero_crossing_rate > 0.1:
                return "speech"
            else:
                return "speech"  # Default to speech

        except Exception:
            return ""

    async def _transcribe(self, path: Path) -> dict[str, Any]:
        """Transcribe audio using Whisper."""
        try:
            if self._whisper_model is None:
                import whisper
                self._whisper_model = whisper.load_model(
                    self.model_size, device=self.device
                )

            result = self._whisper_model.transcribe(str(path))

            segments = []
            for seg in result.get("segments", []):
                segments.append(AudioSegment(
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"].strip(),
                    confidence=seg.get("confidence", 1.0),
                ))

            return {
                "text": result["text"],
                "language": result.get("language", "en"),
                "segments": segments,
            }

        except ImportError:
            # Try faster-whisper as alternative
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(self.model_size, device=self.device)
                segments_iter, info = model.transcribe(str(path))

                segments = []
                full_text = []
                for seg in segments_iter:
                    segments.append(AudioSegment(
                        start_time=seg.start,
                        end_time=seg.end,
                        text=seg.text.strip(),
                        confidence=seg.avg_logprob,
                    ))
                    full_text.append(seg.text)

                return {
                    "text": " ".join(full_text),
                    "language": info.language,
                    "segments": segments,
                }

            except ImportError:
                return {"text": "", "segments": []}

        except Exception:
            return {"text": "", "segments": []}

    async def _diarize(self, path: Path) -> list[str]:
        """Perform speaker diarization."""
        try:
            if self._diarization_pipeline is None:
                from pyannote.audio import Pipeline

                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True,
                )

            diarization = self._diarization_pipeline(str(path))

            # Extract unique speakers
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)

            return list(speakers)

        except ImportError:
            return []
        except Exception:
            return []

    async def _assign_speakers(
        self,
        segments: list[AudioSegment],
        path: Path,
    ) -> list[AudioSegment]:
        """Assign speakers to transcript segments."""
        try:
            if self._diarization_pipeline is None:
                return segments

            diarization = self._diarization_pipeline(str(path))

            # Create speaker timeline
            speaker_timeline = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })

            # Assign speakers to segments
            for segment in segments:
                segment_mid = (segment.start_time + segment.end_time) / 2
                for speaker_seg in speaker_timeline:
                    if speaker_seg["start"] <= segment_mid <= speaker_seg["end"]:
                        segment.speaker = speaker_seg["speaker"]
                        break

            return segments

        except Exception:
            return segments

    async def _analyze_transcript(self, transcript: str) -> dict[str, Any]:
        """Analyze transcript for topics and emotions."""
        topics = []
        emotions = []

        # Simple keyword-based topic extraction
        topic_keywords = {
            "technology": ["computer", "software", "app", "tech", "code", "program"],
            "business": ["meeting", "project", "deadline", "client", "work"],
            "health": ["doctor", "health", "exercise", "medicine", "symptoms"],
            "travel": ["trip", "flight", "hotel", "vacation", "travel"],
            "food": ["restaurant", "cook", "recipe", "dinner", "lunch"],
            "family": ["family", "kids", "parents", "children", "home"],
        }

        transcript_lower = transcript.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in transcript_lower for kw in keywords):
                topics.append(topic)

        # Simple emotion detection
        emotion_keywords = {
            "happy": ["happy", "excited", "great", "wonderful", "amazing"],
            "sad": ["sad", "unfortunately", "sorry", "disappointed"],
            "angry": ["angry", "frustrated", "annoyed", "upset"],
            "worried": ["worried", "concerned", "anxious", "nervous"],
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in transcript_lower for kw in keywords):
                emotions.append(emotion)

        return {"topics": topics, "emotions": emotions}

    async def process_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> AsyncIterator[AudioMemory]:
        """Process all audio files in a directory."""
        path = Path(directory)
        if not path.exists():
            return

        extensions = extensions or [
            ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"
        ]
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in extensions:
                try:
                    memory = await self.process_audio(file_path)
                    yield memory
                except Exception:
                    continue

    async def search_by_transcript(
        self,
        query: str,
        audio_memories: list[AudioMemory],
        threshold: float = 0.3,
    ) -> list[tuple[AudioMemory, float]]:
        """Search audio memories by transcript content."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []
        for memory in audio_memories:
            if not memory.full_transcript:
                continue

            transcript_lower = memory.full_transcript.lower()
            transcript_words = set(transcript_lower.split())

            # Calculate word overlap
            common = query_words & transcript_words
            if not common:
                continue

            score = len(common) / len(query_words)
            if score >= threshold:
                results.append((memory, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
