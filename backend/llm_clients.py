"""
LLM Client Manager for AI Brand Studio
======================================
Manages connections to various LLM providers (Hugging Face, local models).
"""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.language_models.base import BaseLanguageModel

# Hugging Face imports
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Image generation imports
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LLMClientManager:
    """
    Manages LLM clients for text and image generation.
    
    Supports:
    - Hugging Face Inference API
    - Local Hugging Face models
    - Stable Diffusion for image generation
    """
    
    def __init__(
        self,
        text_model: str = None,
        image_model: str = None,
        hf_token: str = None,
        use_local_gpu: bool = False,
        use_hf_inference_api: bool = True
    ):
        """
        Initialize the LLM client manager.
        
        Args:
            text_model: Hugging Face model ID for text generation
            image_model: Hugging Face model ID for image generation
            hf_token: Hugging Face API token
            use_local_gpu: Whether to use local GPU for inference
            use_hf_inference_api: Whether to use HF Inference API
        """
        self.text_model = text_model or os.getenv(
            "HF_TEXT_MODEL", 
            "mistralai/Mistral-7B-Instruct-v0.1"
        )
        self.image_model = image_model or os.getenv(
            "HF_IMAGE_MODEL",
            "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.use_local_gpu = use_local_gpu or os.getenv("USE_LOCAL_GPU", "false").lower() == "true"
        self.use_hf_inference_api = use_hf_inference_api or os.getenv("USE_HF_INFERENCE_API", "true").lower() == "true"
        
        # Initialize clients
        self._text_llm = None
        self._image_pipeline = None
        self._inference_client = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration."""
        if not self.hf_token and self.use_hf_inference_api:
            print("Warning: No HF_TOKEN provided. Some features may not work.")
        
        if self.use_local_gpu and not DIFFUSERS_AVAILABLE:
            print("Warning: diffusers not installed. Local GPU inference disabled.")
            self.use_local_gpu = False
    
    def get_text_llm(self) -> BaseLanguageModel:
        """
        Get or create the text generation LLM.
        
        Returns:
            LangChain LLM instance
        """
        if self._text_llm is not None:
            return self._text_llm
        
        try:
            if self.use_hf_inference_api and self.hf_token:
                # Use Hugging Face Inference API
                self._text_llm = HuggingFaceEndpoint(
                    repo_id=self.text_model,
                    huggingfacehub_api_token=self.hf_token,
                    task="text-generation",
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )
            else:
                # Use HuggingFaceHub (older interface)
                self._text_llm = HuggingFaceHub(
                    repo_id=self.text_model,
                    huggingfacehub_api_token=self.hf_token,
                    model_kwargs={
                        "temperature": 0.7,
                        "max_new_tokens": 1024,
                    }
                )
            
            return self._text_llm
            
        except Exception as e:
            print(f"Error initializing text LLM: {e}")
            # Return a mock LLM for development/testing
            return MockTextLLM()
    
    def get_inference_client(self) -> Optional['InferenceClient']:
        """
        Get or create the Hugging Face Inference Client.
        
        Returns:
            InferenceClient instance or None
        """
        if not HF_AVAILABLE:
            return None
        
        if self._inference_client is not None:
            return self._inference_client
        
        if self.hf_token:
            self._inference_client = InferenceClient(token=self.hf_token)
        
        return self._inference_client
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30
    ) -> Optional[str]:
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt: Text prompt for image generation
            output_path: Path to save the generated image
            negative_prompt: Negative prompt to avoid certain features
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            
        Returns:
            Path to the generated image or None if failed
        """
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default negative prompt for logos
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "text, watermark, signature, letters, words, "
                "realistic photo, photograph, human face, person"
            )
        
        # Try HF Inference API first
        if self.use_hf_inference_api and self.hf_token:
            try:
                return self._generate_image_api(
                    prompt, output_path, negative_prompt, width, height
                )
            except Exception as e:
                print(f"HF API image generation failed: {e}")
        
        # Try local GPU
        if self.use_local_gpu and DIFFUSERS_AVAILABLE:
            try:
                return self._generate_image_local(
                    prompt, output_path, negative_prompt, 
                    width, height, num_inference_steps
                )
            except Exception as e:
                print(f"Local image generation failed: {e}")
        
        # Fallback: Generate a placeholder image
        return self._generate_placeholder_image(output_path, prompt)
    
    def _generate_image_api(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str,
        width: int,
        height: int
    ) -> str:
        """Generate image using Hugging Face Inference API."""
        API_URL = f"https://api-inference.huggingface.co/models/{self.image_model}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": 30
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            # Save image
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        elif response.status_code == 503:
            # Model is loading
            print("Model is loading, waiting...")
            time.sleep(20)
            raise Exception("Model loading, retry")
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")
    
    def _generate_image_local(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int
    ) -> str:
        """Generate image using local Stable Diffusion."""
        if self._image_pipeline is None:
            # Determine which pipeline to use
            if "xl" in self.image_model.lower():
                self._image_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.image_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True
                )
            else:
                self._image_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.image_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._image_pipeline = self._image_pipeline.to("cuda")
                self._image_pipeline.enable_attention_slicing()
        
        # Generate image
        image = self._image_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        # Save image
        image.save(output_path)
        return output_path
    
    def _generate_placeholder_image(self, output_path: str, prompt: str) -> str:
        """Generate a placeholder image when other methods fail."""
        if not PIL_AVAILABLE:
            print("PIL not available, cannot generate placeholder")
            return None
        
        # Create a simple gradient placeholder with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create gradient background
        width, height = 512, 512
        img = Image.new('RGB', (width, height))
        
        # Generate colors based on prompt hash
        import hashlib
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        
        # Color 1 (top-left)
        r1 = (prompt_hash >> 16) & 0xFF
        g1 = (prompt_hash >> 8) & 0xFF
        b1 = prompt_hash & 0xFF
        
        # Color 2 (bottom-right) - complementary-ish
        r2 = (r1 + 128) % 256
        g2 = (g1 + 64) % 256
        b2 = (b1 + 96) % 256
        
        # Create gradient
        for y in range(height):
            for x in range(width):
                # Calculate interpolation factor
                factor = (x + y) / (width + height)
                r = int(r1 + (r2 - r1) * factor)
                g = int(g1 + (g2 - g1) * factor)
                b = int(b1 + (b2 - b1) * factor)
                img.putpixel((x, y), (r, g, b))
        
        # Add a simple icon in the center
        draw = ImageDraw.Draw(img)
        
        # Draw a circle
        center_x, center_y = width // 2, height // 2
        radius = 100
        draw.ellipse(
            [center_x - radius, center_y - radius, 
             center_x + radius, center_y + radius],
            fill=(255, 255, 255, 200),
            outline=(255, 255, 255)
        )
        
        # Add placeholder text
        try:
            # Try to use a system font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = "LOGO"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text,
            fill=(r1, g1, b1),
            font=font
        )
        
        # Save
        img.save(output_path)
        return output_path
    
    def estimate_cost(
        self,
        text_tokens: int = 0,
        image_calls: int = 0
    ) -> Dict[str, float]:
        """
        Estimate the cost of API calls.
        
        Args:
            text_tokens: Estimated number of text tokens
            image_calls: Number of image generation calls
            
        Returns:
            Dictionary with cost breakdown
        """
        # Approximate costs (varies by model and provider)
        text_cost_per_1k = 0.0004  # Approximate for HF inference
        image_cost_per_call = 0.02  # Approximate for SD
        
        text_cost = (text_tokens / 1000) * text_cost_per_1k
        image_cost = image_calls * image_cost_per_call
        
        return {
            "text_tokens": text_tokens,
            "text_cost": text_cost,
            "image_calls": image_calls,
            "image_cost": image_cost,
            "total_cost": text_cost + image_cost
        }


class MockTextLLM:
    """
    Mock LLM for development and testing when real API is unavailable.
    """
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def invoke(self, prompt: Union[str, Dict]) -> str:
        """Generate a mock response."""
        if isinstance(prompt, dict):
            prompt_text = str(prompt)
        else:
            prompt_text = prompt
        
        # Detect what kind of output is expected
        if "domain" in prompt_text.lower():
            return self._mock_domain_response()
        elif "hero" in prompt_text.lower() or "headline" in prompt_text.lower():
            return self._mock_hero_response()
        elif "social" in prompt_text.lower():
            return self._mock_social_response()
        elif "logo" in prompt_text.lower():
            return self._mock_logo_prompt()
        else:
            return "Mock response for: " + prompt_text[:100]
    
    def _mock_domain_response(self) -> str:
        return '''
        {
            "domains": [
                {"name": "brightidea", "tld": ".com", "rationale": "Captures innovation", "score": 95, "available": true},
                {"name": "novahub", "tld": ".com", "rationale": "Modern and memorable", "score": 90, "available": true},
                {"name": "sparkventure", "tld": ".com", "rationale": "Energetic and entrepreneurial", "score": 85, "available": true},
                {"name": "blueleaf", "tld": ".com", "rationale": "Natural and trustworthy", "score": 80, "available": false},
                {"name": "swiftstart", "tld": ".com", "rationale": "Implies quick action", "score": 75, "available": true}
            ]
        }
        '''
    
    def _mock_hero_response(self) -> str:
        return '''
        {
            "brand_name": "BrightIdea Co",
            "headline": "Transform Your Vision Into Reality Today",
            "tagline": "We help entrepreneurs and businesses bring their innovative ideas to life with cutting-edge solutions.",
            "bullets": [
                "Expert guidance from concept to launch",
                "Cutting-edge technology and modern approach",
                "Dedicated support every step of the way"
            ]
        }
        '''
    
    def _mock_social_response(self) -> str:
        return '''
        {
            "posts": [
                {
                    "platform": "twitter",
                    "content": "🚀 Big news! We are launching something amazing. Stay tuned for innovation like you have never seen before!",
                    "hashtags": ["#Innovation", "#Startup", "#LaunchDay"],
                    "call_to_action": "Follow us for updates!"
                },
                {
                    "platform": "instagram",
                    "content": "Behind every great business is a bold idea. ✨ We are here to help you turn yours into reality. Our journey begins now, and we want you to be part of it!",
                    "hashtags": ["#Entrepreneurship", "#BusinessLaunch", "#Dreams", "#Success"],
                    "call_to_action": "Link in bio!"
                },
                {
                    "platform": "linkedin",
                    "content": "Excited to announce the launch of our new venture! We are committed to delivering innovative solutions that make a real difference. Looking forward to connecting with industry leaders and potential partners.",
                    "hashtags": ["#NewBusiness", "#Innovation", "#B2B"],
                    "call_to_action": "Let us connect and explore collaboration opportunities."
                }
            ]
        }
        '''
    
    def _mock_logo_prompt(self) -> str:
        return (
            "Modern minimalist logo design, abstract geometric shape, "
            "gradient from deep blue to teal, clean lines, professional, "
            "suitable for tech startup, no text, scalable vector style, "
            "white background, corporate identity"
        )
    
    def __call__(self, prompt: Union[str, Dict]) -> str:
        return self.invoke(prompt)