# Multimodal Processing

## Overview

The multimodal module processes images, audio, and video to extract memories from non-text content. Each modality has specialized processors that convert media into structured memory entries.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MULTIMODAL PROCESSING                              │
│                                                                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│   │    Image     │  │    Audio     │  │    Video     │                  │
│   │  Processor   │  │  Processor   │  │  Processor   │                  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│          │                 │                 │                          │
│          ▼                 ▼                 ▼                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│   │ ImageMemory  │  │ AudioMemory  │  │ VideoMemory  │                  │
│   │              │  │              │  │              │                  │
│   │ • Objects    │  │ • Transcript │  │ • Summary    │                  │
│   │ • Scene      │  │ • Speakers   │  │ • Chapters   │                  │
│   │ • Text (OCR) │  │ • Topics     │  │ • Keyframes  │                  │
│   │ • Embedding  │  │ • Sentiment  │  │ • Scenes     │                  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│          │                 │                 │                          │
│          └─────────────────┼─────────────────┘                          │
│                            │                                            │
│                            ▼                                            │
│                   ┌────────────────┐                                    │
│                   │  MemoryEntry   │                                    │
│                   │   (unified)    │                                    │
│                   └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Image Processing

### ImageProcessor

Analyzes images to extract visual memories.

```python
from memory.multimodal import ImageProcessor

processor = ImageProcessor(
    use_clip=True,              # CLIP for image understanding
    use_claude_vision=False,    # Optional external vision API
    use_tesseract=True,         # OCR for text extraction
    thumbnail_size=(256, 256),  # Generate thumbnails
)

# Process a single image
memory = await processor.process_image("path/to/photo.jpg")

print(f"Description: {memory.description}")
print(f"Objects: {memory.objects}")
print(f"Scene type: {memory.scene_type}")
print(f"Text found: {memory.extracted_text}")
```

### ImageMemory

Structured representation of image content.

```python
from memory.multimodal import ImageMemory

memory = ImageMemory(
    source_path="/path/to/image.jpg",
    description="A sunset over mountains with clouds",
    objects=["mountain", "sun", "clouds", "trees"],
    scene_type="outdoor",
    extracted_text="Welcome to Yosemite",
    colors=["orange", "blue", "purple"],
    faces_detected=2,
    location={"lat": 37.7749, "lon": -122.4194},
    timestamp=datetime(2024, 1, 15, 18, 30),
    dimensions=(1920, 1080),
    embedding=[...],  # CLIP embedding for similarity search
)

# Convert to text for memory system
text = memory.to_text()
# "Image: A sunset over mountains with clouds.
#  Objects: mountain, sun, clouds, trees.
#  Scene: outdoor. Location: Yosemite.
#  Text found: 'Welcome to Yosemite'"
```

### Capabilities

| Feature | Technology | Description |
|---------|------------|-------------|
| Object Detection | CLIP / YOLO | Identify objects in scene |
| Scene Classification | CLIP | Categorize scene type |
| OCR | Tesseract / Vision API | Extract text from images |
| Face Detection | OpenCV / dlib | Detect and count faces |
| Color Analysis | OpenCV | Extract dominant colors |
| EXIF Extraction | PIL | Get metadata (date, location) |
| Similarity Embeddings | CLIP | Enable image search |

## Audio Processing

### AudioProcessor

Transcribes and analyzes audio content.

```python
from memory.multimodal import AudioProcessor

processor = AudioProcessor(
    model_size="base",          # Whisper model size
    use_diarization=True,       # Speaker identification
    detect_language=True,       # Auto language detection
    device="cpu",               # or "cuda" for GPU
)

# Process audio file
memory = await processor.process_audio("path/to/podcast.mp3")

print(f"Transcript: {memory.full_transcript[:500]}...")
print(f"Speakers: {memory.speakers}")
print(f"Duration: {memory.duration_seconds}s")
print(f"Topics: {memory.topics}")
```

### AudioMemory

Structured representation of audio content.

```python
from memory.multimodal import AudioMemory, AudioSegment

memory = AudioMemory(
    source_path="/path/to/audio.mp3",
    full_transcript="Welcome to the podcast. Today we discuss...",
    segments=[
        AudioSegment(
            start_time=0.0,
            end_time=5.0,
            text="Welcome to the podcast.",
            speaker="Host",
            confidence=0.95,
        ),
        AudioSegment(
            start_time=5.0,
            end_time=12.0,
            text="Today we discuss technology trends.",
            speaker="Host",
            confidence=0.92,
        ),
    ],
    speakers=["Host", "Guest"],
    language="en",
    duration_seconds=1800.0,
    audio_type="podcast",
    topics=["technology", "AI", "future"],
    sentiment="positive",
)

# Convert to text
text = memory.to_text()
```

### Capabilities

| Feature | Technology | Description |
|---------|------------|-------------|
| Transcription | OpenAI Whisper | Speech to text |
| Speaker Diarization | pyannote.audio | Identify who said what |
| Language Detection | Whisper | Detect spoken language |
| Topic Extraction | LLM | Summarize main topics |
| Sentiment Analysis | LLM | Detect overall tone |
| Keyword Extraction | NLP | Find key terms |

## Video Processing

### VideoProcessor

Processes video content including audio and visual tracks.

```python
from memory.multimodal import VideoProcessor

processor = VideoProcessor(
    extract_audio=True,         # Process audio track
    extract_frames=True,        # Extract keyframes
    frames_per_minute=2,        # Keyframe frequency
    detect_scenes=True,         # Scene boundary detection
    generate_chapters=True,     # Auto-generate chapters
)

# Process video file
memory = await processor.process_video("path/to/meeting.mp4")

print(f"Summary: {memory.summary}")
print(f"Duration: {memory.duration_seconds}s")
print(f"Chapters: {len(memory.chapters)}")
print(f"Scenes: {len(memory.scenes)}")
```

### VideoMemory

Structured representation of video content.

```python
from memory.multimodal import VideoMemory, VideoFrame

memory = VideoMemory(
    source_path="/path/to/video.mp4",
    summary="Team standup meeting discussing project progress",
    full_transcript="Good morning everyone...",
    chapters=[
        {"title": "Introduction", "start": 0, "end": 60},
        {"title": "Progress Updates", "start": 60, "end": 300},
        {"title": "Discussion", "start": 300, "end": 600},
        {"title": "Action Items", "start": 600, "end": 720},
    ],
    keyframes=[
        VideoFrame(
            timestamp=30.0,
            description="Title slide showing meeting agenda",
            is_scene_change=False,
        ),
        VideoFrame(
            timestamp=120.0,
            description="Presenter showing project dashboard",
            is_scene_change=True,
        ),
    ],
    duration_seconds=720.0,
    width=1920,
    height=1080,
    fps=30.0,
    video_type="meeting",
    participants=["Alice", "Bob", "Carol"],
    topics=["project", "deadline", "blockers"],
)
```

### Capabilities

| Feature | Technology | Description |
|---------|------------|-------------|
| Audio Extraction | ffmpeg | Separate audio track |
| Transcription | Whisper | Transcribe audio |
| Scene Detection | scenedetect | Find scene boundaries |
| Keyframe Extraction | OpenCV | Extract representative frames |
| Frame Analysis | CLIP | Describe frame content |
| Chapter Generation | LLM | Create logical chapters |
| Motion Detection | OpenCV | Detect significant motion |

## Integration with Memory System

### Converting to MemoryEntry

```python
from memory import MemoryAPI
from memory.multimodal import ImageProcessor, AudioProcessor

api = MemoryAPI()
await api.initialize()

# Process image
image_processor = ImageProcessor()
image_memory = await image_processor.process_image("photo.jpg")

# Store as memory
await api.remember(
    content=image_memory.to_text(),
    memory_type="EVENT",
    domains=["photos", "personal"],
    tags=image_memory.objects,
    metadata={
        "source_type": "image",
        "source_path": image_memory.source_path,
        "dimensions": image_memory.dimensions,
    },
)

# Process audio
audio_processor = AudioProcessor()
audio_memory = await audio_processor.process_audio("podcast.mp3")

await api.remember(
    content=audio_memory.to_text(),
    memory_type="CONTEXT",
    domains=["podcasts", audio_memory.audio_type],
    tags=audio_memory.topics,
    metadata={
        "source_type": "audio",
        "duration": audio_memory.duration_seconds,
        "speakers": audio_memory.speakers,
    },
)
```

### Batch Processing

```python
from memory.multimodal import MultimodalProcessor

processor = MultimodalProcessor(
    image_processor=ImageProcessor(),
    audio_processor=AudioProcessor(),
    video_processor=VideoProcessor(),
)

# Process entire directory
memories = await processor.process_directory(
    path="~/Photos/2024",
    recursive=True,
    file_types=["jpg", "png", "mp4", "mp3"],
)

print(f"Processed {len(memories)} files")
```

## Configuration Reference

### ImageProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_clip` | bool | True | Use CLIP for understanding |
| `use_claude_vision` | bool | False | Use external vision API |
| `use_tesseract` | bool | True | Enable OCR |
| `thumbnail_size` | tuple | (256, 256) | Thumbnail dimensions |
| `min_confidence` | float | 0.7 | Min confidence for objects |

### AudioProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | str | "base" | Whisper model size |
| `use_diarization` | bool | True | Enable speaker ID |
| `detect_language` | bool | True | Auto language detection |
| `device` | str | "cpu" | Processing device |
| `chunk_length` | int | 30 | Audio chunk length (seconds) |

### VideoProcessor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extract_audio` | bool | True | Process audio track |
| `extract_frames` | bool | True | Extract keyframes |
| `frames_per_minute` | int | 2 | Keyframe frequency |
| `detect_scenes` | bool | True | Scene detection |
| `generate_chapters` | bool | True | Auto-generate chapters |
| `max_frames` | int | 100 | Max keyframes to extract |

## Dependencies

Install multimodal dependencies:

```bash
pip install -e ".[multimodal]"
```

This installs:
- Pillow (image processing)
- pytesseract (OCR)
- opencv-python (video/image processing)
- librosa (audio analysis)
- openai-whisper (transcription)
- pyannote.audio (speaker diarization)
- scenedetect (scene detection)

---

*See [ARCHITECTURE.md](./ARCHITECTURE.md) for overall system architecture.*
*See [API.md](./API.md) for complete API reference.*
