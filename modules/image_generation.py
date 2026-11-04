import os
import uuid
import urllib.parse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import httpx
from PIL import Image

logger = logging.getLogger("haven.image_generation")

class ImageGenerationService:
    """
    Dedicated AI Image Generation Engine.
    Supports Hugging Face Inference API and Pollinations High-Definition Neural Diffusion.
    Generates actual image files (.png/.jpg) and persists them for conversation rendering and download.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        try:
            from app.config import settings
            default_dir = getattr(
                settings,
                "GENERATED_IMAGES_DIR",
                getattr(settings, "MEDIA_DIR", Path("media")) / "generated_images"
            )
        except Exception as e:
            logger.warning("Could not load settings in ImageGenerationService: %s", e)
            default_dir = Path("media") / "generated_images"

        self.output_dir = Path(output_dir) if output_dir else default_dir
        self._ensure_output_dir()
        self.hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or ""

    def _ensure_output_dir(self):
        """Ensures destination directory exists and is writable."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create image output directory %s: %s", self.output_dir, e)

    def generate_image(self, prompt: str, seed: Optional[int] = None) -> Tuple[bool, Optional[Path], str]:
        """
        Generates an actual image from prompt.
        Returns: (success, saved_image_path, status_message)
        """
        if not prompt or not prompt.strip():
            return False, None, "Please provide an image prompt description."

        clean_prompt = prompt.strip()
        self._ensure_output_dir()

        filename = f"gen_{uuid.uuid4().hex[:10]}.png"
        dest_path = self.output_dir / filename

        # 1. Try Hugging Face Stable Diffusion / FLUX if API key provided
        if self.hf_token:
            hf_success, hf_path = self._generate_with_huggingface(clean_prompt, dest_path)
            if hf_success and hf_path and hf_path.exists():
                return True, hf_path, "Image successfully generated via Hugging Face Diffusion."

        # 2. High-Performance Direct Neural Diffusion Engine (Pollinations / FLUX serverless)
        direct_success, direct_path = self._generate_with_direct_diffusion(clean_prompt, dest_path, seed)
        if direct_success and direct_path and direct_path.exists():
            return True, direct_path, "Image successfully generated via Neural Diffusion Engine."

        return False, None, "Could not generate image. Please check network connectivity or API key."

    def _generate_with_huggingface(self, prompt: str, dest_path: Path) -> Tuple[bool, Optional[Path]]:
        """Invokes Hugging Face SDXL / FLUX text-to-image pipeline."""
        try:
            hf_model = os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {"inputs": prompt}

            with httpx.Client(timeout=60) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code == 200 and "image" in res.headers.get("content-type", ""):
                    with open(dest_path, "wb") as f:
                        f.write(res.content)
                    return True, dest_path
        except Exception as e:
            logger.warning("Hugging Face image generation notice: %s", e)
        return False, None

    def _generate_with_direct_diffusion(self, prompt: str, dest_path: Path, seed: Optional[int] = None) -> Tuple[bool, Optional[Path]]:
        """Generates real high-resolution images via Pollinations / FLUX pipeline."""
        try:
            encoded_prompt = urllib.parse.quote(prompt)
            seed_val = seed or uuid.uuid4().int % 100000
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed_val}"

            with httpx.Client(timeout=45, follow_redirects=True) as client:
                res = client.get(url)
                if res.status_code == 200 and len(res.content) > 1000:
                    with open(dest_path, "wb") as f:
                        f.write(res.content)
                    # Verify image validity
                    with Image.open(dest_path) as img:
                        img.verify()
                    return True, dest_path
        except Exception as e:
            logger.error("Direct diffusion image error: %s", e)
        return False, None
