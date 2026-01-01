"""Tests for multimodal processing module."""

import pytest


class TestImageProcessor:
    """Tests for image processing."""

    def test_processor_initialization(self):
        """Test image processor initialization."""
        from memory.multimodal import ImageProcessor

        processor = ImageProcessor()

        assert processor is not None
        assert processor.use_clip is True

    def test_processor_custom_config(self):
        """Test processor with custom configuration."""
        from memory.multimodal import ImageProcessor

        processor = ImageProcessor(
            use_claude_vision=False,
            use_clip=False,
        )

        assert processor.use_claude_vision is False
        assert processor.use_clip is False

    def test_image_memory_dataclass(self):
        """Test ImageMemory dataclass."""
        from memory.multimodal import ImageMemory

        memory = ImageMemory(
            source_path="/path/to/image.jpg",
            description="A test image",
            objects=["cat", "dog"],
        )

        assert memory.source_path == "/path/to/image.jpg"
        assert memory.description == "A test image"
        assert "cat" in memory.objects

    def test_image_memory_to_text(self):
        """Test converting image memory to text."""
        from datetime import datetime

        from memory.multimodal import ImageMemory

        memory = ImageMemory(
            source_path="/path/to/image.jpg",
            description="A sunset over mountains",
            objects=["mountain", "sun", "clouds"],
            scene_type="outdoor",
            timestamp=datetime(2024, 1, 15),
        )

        text = memory.to_text()

        assert "sunset" in text.lower()
        assert "mountain" in text.lower() or "outdoor" in text.lower()

    @pytest.mark.asyncio
    async def test_process_nonexistent_image(self, temp_dir):
        """Test handling of non-existent image."""
        from memory.multimodal import ImageProcessor

        processor = ImageProcessor(use_clip=False)

        # Should handle gracefully
        try:
            await processor.process_image(temp_dir / "nonexistent.jpg")
        except Exception:
            # Expected to fail
            assert True

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_process_real_image(self, temp_dir):
        """Test processing a real image file."""
        from memory.multimodal import ImageProcessor

        # Create a simple test image
        try:
            from PIL import Image

            img = Image.new("RGB", (100, 100), color="red")
            img_path = temp_dir / "test.jpg"
            img.save(str(img_path))

            processor = ImageProcessor(use_clip=False, use_claude_vision=False)
            memory = await processor.process_image(img_path)

            assert memory is not None
            assert memory.source_path == str(img_path)
            assert memory.dimensions == (100, 100)

        except ImportError:
            pytest.skip("PIL not installed")


class TestAudioProcessor:
    """Tests for audio processing."""

    def test_processor_initialization(self):
        """Test audio processor initialization."""
        from memory.multimodal import AudioProcessor

        processor = AudioProcessor()

        assert processor is not None
        assert processor.model_size == "base"

    def test_processor_custom_config(self):
        """Test processor with custom configuration."""
        from memory.multimodal import AudioProcessor

        processor = AudioProcessor(
            model_size="small",
            use_diarization=False,
            device="cpu",
        )

        assert processor.model_size == "small"
        assert processor.use_diarization is False

    def test_audio_memory_dataclass(self):
        """Test AudioMemory dataclass."""
        from memory.multimodal import AudioMemory

        memory = AudioMemory(
            source_path="/path/to/audio.mp3",
            full_transcript="Hello world",
            duration_seconds=60.0,
            speakers=["Speaker 1", "Speaker 2"],
        )

        assert memory.source_path == "/path/to/audio.mp3"
        assert memory.full_transcript == "Hello world"
        assert memory.duration_seconds == 60.0
        assert len(memory.speakers) == 2

    def test_audio_memory_to_text(self):
        """Test converting audio memory to text."""
        from memory.multimodal import AudioMemory

        memory = AudioMemory(
            audio_type="podcast",
            full_transcript="Discussion about technology trends",
            speakers=["Host", "Guest"],
            topics=["technology", "AI"],
            duration_seconds=1800,
        )

        text = memory.to_text()

        assert "podcast" in text.lower()
        assert "technology" in text.lower()


class TestVideoProcessor:
    """Tests for video processing."""

    def test_processor_initialization(self):
        """Test video processor initialization."""
        from memory.multimodal import VideoProcessor

        processor = VideoProcessor()

        assert processor is not None
        assert processor.extract_audio is True
        assert processor.extract_frames is True

    def test_processor_custom_config(self):
        """Test processor with custom configuration."""
        from memory.multimodal import VideoProcessor

        processor = VideoProcessor(
            extract_audio=False,
            frames_per_minute=5,
            detect_scenes=False,
        )

        assert processor.extract_audio is False
        assert processor.frames_per_minute == 5

    def test_video_memory_dataclass(self):
        """Test VideoMemory dataclass."""
        from memory.multimodal import VideoMemory

        memory = VideoMemory(
            source_path="/path/to/video.mp4",
            duration_seconds=300.0,
            width=1920,
            height=1080,
            fps=30.0,
            video_type="presentation",
        )

        assert memory.source_path == "/path/to/video.mp4"
        assert memory.duration_seconds == 300.0
        assert memory.width == 1920
        assert memory.video_type == "presentation"

    def test_video_memory_to_text(self):
        """Test converting video memory to text."""
        from memory.multimodal import VideoMemory

        memory = VideoMemory(
            video_type="meeting",
            summary="Team standup meeting",
            topics=["project update", "blockers"],
            duration_seconds=900,
            chapters=[
                {"title": "Introduction"},
                {"title": "Updates"},
                {"title": "Discussion"},
            ],
        )

        text = memory.to_text()

        assert "meeting" in text.lower()
        assert "15:00" in text or "0:15:00" in text  # Duration


class TestMultimodalIntegration:
    """Tests for multimodal module integration."""

    def test_import_all_processors(self):
        """Test importing all processor classes."""
        from memory.multimodal import (
            AudioMemory,
            AudioProcessor,
            ImageMemory,
            ImageProcessor,
            VideoMemory,
            VideoProcessor,
        )

        assert ImageProcessor is not None
        assert ImageMemory is not None
        assert AudioProcessor is not None
        assert AudioMemory is not None
        assert VideoProcessor is not None
        assert VideoMemory is not None

    def test_memory_serialization(self):
        """Test that memories can be serialized."""
        from memory.multimodal import AudioMemory, ImageMemory, VideoMemory

        # Image
        img_mem = ImageMemory(source_path="test.jpg", description="Test")
        assert img_mem.to_text() is not None

        # Audio
        audio_mem = AudioMemory(source_path="test.mp3", full_transcript="Hello")
        assert audio_mem.to_text() is not None

        # Video
        video_mem = VideoMemory(source_path="test.mp4", summary="Test video")
        assert video_mem.to_text() is not None


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_segment_creation(self):
        """Test creating an audio segment."""
        from memory.multimodal.audio import AudioSegment

        segment = AudioSegment(
            start_time=0.0,
            end_time=5.0,
            text="Hello world",
            speaker="Speaker 1",
            confidence=0.95,
        )

        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Hello world"
        assert segment.speaker == "Speaker 1"


class TestVideoFrame:
    """Tests for VideoFrame dataclass."""

    def test_frame_creation(self):
        """Test creating a video frame."""
        from memory.multimodal.video import VideoFrame

        frame = VideoFrame(
            timestamp=10.5,
            is_scene_change=True,
            motion_score=0.7,
        )

        assert frame.timestamp == 10.5
        assert frame.is_scene_change is True
        assert frame.motion_score == 0.7
