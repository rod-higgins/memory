"""
Image processing for PLM multi-modal support.

Extracts memories from images including:
- Text via OCR
- Scene descriptions
- Object detection
- Face recognition
- EXIF metadata
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ImageMemory:
    """A memory extracted from an image."""

    id: str = field(default_factory=lambda: str(uuid4()))
    source_path: str = ""
    timestamp: datetime | None = None

    # Extracted content
    ocr_text: str = ""
    description: str = ""
    objects: list[str] = field(default_factory=list)
    faces: list[dict[str, Any]] = field(default_factory=list)
    scene_type: str = ""
    emotions: list[str] = field(default_factory=list)

    # Metadata
    location: dict[str, float] | None = None  # lat, lon
    camera: str = ""
    dimensions: tuple[int, int] = (0, 0)

    # Embedding for similarity search
    embedding: list[float] | None = None

    def to_text(self) -> str:
        """Convert image memory to searchable text."""
        parts = []

        if self.description:
            parts.append(f"Image: {self.description}")

        if self.ocr_text:
            parts.append(f"Text in image: {self.ocr_text}")

        if self.objects:
            parts.append(f"Objects: {', '.join(self.objects)}")

        if self.scene_type:
            parts.append(f"Scene: {self.scene_type}")

        if self.location:
            parts.append(f"Location: {self.location.get('name', 'Unknown')}")

        if self.timestamp:
            parts.append(f"Date: {self.timestamp.strftime('%Y-%m-%d')}")

        return "\n".join(parts)


class ImageProcessor:
    """
    Processes images to extract memories.

    Supports multiple backends:
    - Local: Uses pytesseract for OCR, PIL for metadata
    - Claude Vision: Uses Claude API for rich descriptions
    - CLIP: Uses CLIP for embeddings and similarity
    """

    def __init__(
        self,
        use_claude_vision: bool = False,
        use_clip: bool = True,
        claude_api_key: str | None = None,
    ):
        self.use_claude_vision = use_claude_vision
        self.use_clip = use_clip
        self.claude_api_key = claude_api_key

        self._clip_model = None
        self._clip_processor = None

    async def process_image(
        self,
        image_path: str | Path,
        extract_text: bool = True,
        detect_objects: bool = True,
        generate_description: bool = True,
    ) -> ImageMemory:
        """
        Process an image and extract memory content.

        Args:
            image_path: Path to image file
            extract_text: Whether to run OCR
            detect_objects: Whether to detect objects
            generate_description: Whether to generate description
        """
        path = Path(image_path)
        memory = ImageMemory(source_path=str(path))

        # Extract EXIF metadata
        metadata = await self._extract_metadata(path)
        memory.timestamp = metadata.get("timestamp")
        memory.location = metadata.get("location")
        memory.camera = metadata.get("camera", "")
        memory.dimensions = metadata.get("dimensions", (0, 0))

        # OCR
        if extract_text:
            memory.ocr_text = await self._extract_text(path)

        # Object detection and description
        if self.use_claude_vision and self.claude_api_key:
            result = await self._claude_vision_analysis(path)
            memory.description = result.get("description", "")
            memory.objects = result.get("objects", [])
            memory.scene_type = result.get("scene_type", "")
            memory.emotions = result.get("emotions", [])
        elif detect_objects or generate_description:
            # Use local models
            if detect_objects:
                memory.objects = await self._detect_objects_local(path)
            if generate_description:
                memory.description = await self._generate_description_local(path)

        # Generate embedding for similarity search
        if self.use_clip:
            memory.embedding = await self._generate_clip_embedding(path)

        return memory

    async def _extract_metadata(self, path: Path) -> dict[str, Any]:
        """Extract EXIF and other metadata from image."""
        metadata = {}

        try:
            from PIL import Image
            from PIL.ExifTags import GPSTAGS, TAGS

            with Image.open(path) as img:
                metadata["dimensions"] = img.size

                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)

                        if tag == "DateTimeOriginal":
                            try:
                                metadata["timestamp"] = datetime.strptime(
                                    value, "%Y:%m:%d %H:%M:%S"
                                )
                            except ValueError:
                                pass

                        elif tag == "Make":
                            metadata["camera"] = value

                        elif tag == "GPSInfo":
                            # Extract GPS coordinates
                            gps_data = {}
                            for gps_tag_id, gps_value in value.items():
                                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                                gps_data[gps_tag] = gps_value

                            if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                                lat = self._convert_gps_coords(
                                    gps_data["GPSLatitude"],
                                    gps_data.get("GPSLatitudeRef", "N"),
                                )
                                lon = self._convert_gps_coords(
                                    gps_data["GPSLongitude"],
                                    gps_data.get("GPSLongitudeRef", "E"),
                                )
                                metadata["location"] = {"lat": lat, "lon": lon}

        except Exception:
            pass

        return metadata

    def _convert_gps_coords(
        self, coords: tuple, ref: str
    ) -> float:
        """Convert GPS coordinates to decimal degrees."""
        try:
            degrees = float(coords[0])
            minutes = float(coords[1])
            seconds = float(coords[2])

            decimal = degrees + minutes / 60 + seconds / 3600

            if ref in ["S", "W"]:
                decimal = -decimal

            return decimal
        except Exception:
            return 0.0

    async def _extract_text(self, path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            import pytesseract
            from PIL import Image

            with Image.open(path) as img:
                text = pytesseract.image_to_string(img)
                return text.strip()

        except ImportError:
            # pytesseract not installed
            return ""
        except Exception:
            return ""

    async def _detect_objects_local(self, path: Path) -> list[str]:
        """Detect objects using local model."""
        try:
            # Try YOLO via ultralytics
            from ultralytics import YOLO

            model = YOLO("yolov8n.pt")
            results = model(str(path), verbose=False)

            objects = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if class_name not in objects:
                        objects.append(class_name)

            return objects

        except ImportError:
            return []
        except Exception:
            return []

    async def _generate_description_local(self, path: Path) -> str:
        """Generate image description using local model."""
        try:
            # Try BLIP for image captioning
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

            with Image.open(path) as img:
                inputs = processor(img, return_tensors="pt")
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

            return caption

        except ImportError:
            return ""
        except Exception:
            return ""

    async def _claude_vision_analysis(self, path: Path) -> dict[str, Any]:
        """Use Claude Vision API for rich image analysis."""
        try:
            import httpx

            # Read and encode image
            with open(path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            # Determine media type
            suffix = path.suffix.lower()
            media_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            media_type = media_types.get(suffix, "image/jpeg")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.claude_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1024,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": """Analyze this image and provide:
1. A brief description (1-2 sentences)
2. List of objects visible
3. Scene type (indoor, outdoor, nature, urban, etc.)
4. Any emotions or mood conveyed

Respond in JSON format:
{"description": "...", "objects": [...], "scene_type": "...", "emotions": [...]}""",
                                    },
                                ],
                            }
                        ],
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"]
                    # Parse JSON from response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"description": content}

        except Exception as e:
            return {"error": str(e)}

        return {}

    async def _generate_clip_embedding(self, path: Path) -> list[float] | None:
        """Generate CLIP embedding for image similarity search."""
        try:
            if self._clip_model is None:
                from transformers import CLIPModel, CLIPProcessor

                self._clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )

            from PIL import Image

            with Image.open(path) as img:
                inputs = self._clip_processor(
                    images=img, return_tensors="pt"
                )
                outputs = self._clip_model.get_image_features(**inputs)
                embedding = outputs[0].detach().numpy().tolist()

            return embedding

        except ImportError:
            return None
        except Exception:
            return None

    async def process_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> AsyncIterator[ImageMemory]:
        """Process all images in a directory."""
        path = Path(directory)
        if not path.exists():
            return

        extensions = extensions or [".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"]
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in extensions:
                try:
                    memory = await self.process_image(file_path)
                    yield memory
                except Exception:
                    continue

    async def find_similar_images(
        self,
        query_image: str | Path,
        image_memories: list[ImageMemory],
        top_k: int = 5,
    ) -> list[tuple[ImageMemory, float]]:
        """Find images similar to a query image."""
        query_embedding = await self._generate_clip_embedding(Path(query_image))
        if not query_embedding:
            return []

        # Calculate similarities
        results = []
        for memory in image_memories:
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                results.append((memory, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
