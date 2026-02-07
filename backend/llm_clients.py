"""
Manages connections to various LLM providers (Hugging Face, local models).
"""

import os
import io
import time
import math
from typing import Optional, Dict, Any, Union
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# LangChain imports
from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint
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
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LLMClientManager:
    """
    Manages LLM clients for text and image generation.
    """
    
    def __init__(
        self,
        text_model: str = None,
        image_model: str = None,
        hf_token: str = None,
        use_local_gpu: bool = False,
        use_hf_inference_api: bool = True
    ):
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
        
        self._text_llm = None
        self._image_pipeline = None
        self._inference_client = None
        self._last_image_error = None
        
        self._validate_config()
    
    def _validate_config(self):
        if not self.hf_token and self.use_hf_inference_api:
            print("⚠️ Warning: No HF_TOKEN provided. Image generation may not work.")
    
    def get_text_llm(self) -> BaseLanguageModel:
        if self._text_llm is not None:
            return self._text_llm
        
        try:
            if self.use_hf_inference_api and self.hf_token:
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
            return MockTextLLM()
    
    def get_last_image_error(self) -> Optional[str]:
        """Get the last image generation error message."""
        return self._last_image_error
    
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Generate an image using Stable Diffusion.
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, deformed, ugly, "
                "text, watermark, signature, letters, words, "
                "realistic photo, photograph, human face, person, "
                "bad anatomy, bad proportions, noise, grainy"
            )
        
        self._last_image_error = None
        
        # Try HF Inference API
        if self.use_hf_inference_api and self.hf_token:
            for attempt in range(max_retries):
                try:
                    result = self._generate_image_api(
                        prompt, output_path, negative_prompt, width, height
                    )
                    if result:
                        return result
                except Exception as e:
                    self._last_image_error = str(e)
                    print(f"⚠️ HF API attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 10
                        print(f"   Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
        
        # Try local GPU
        if self.use_local_gpu and DIFFUSERS_AVAILABLE:
            try:
                return self._generate_image_local(
                    prompt, output_path, negative_prompt, 
                    width, height, num_inference_steps
                )
            except Exception as e:
                self._last_image_error = str(e)
                print(f"⚠️ Local image generation failed: {e}")
        
        # Fallback: Generate a nice placeholder
        print("📝 Using placeholder image generation")
        return self._generate_professional_placeholder(output_path, prompt)
    
    def _generate_image_api(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str,
        width: int,
        height: int
    ) -> Optional[str]:
        """Generate image using Hugging Face Inference API."""
        
        # Try multiple models in order of preference
        models_to_try = [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "runwayml/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-2-1",
            "CompVis/stable-diffusion-v1-4",
        ]
        
        # Put the configured model first
        if self.image_model not in models_to_try:
            models_to_try.insert(0, self.image_model)
        else:
            models_to_try.remove(self.image_model)
            models_to_try.insert(0, self.image_model)
        
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        for model in models_to_try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "negative_prompt": negative_prompt,
                    "width": min(width, 1024),
                    "height": min(height, 1024),
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            try:
                print(f"🎨 Trying model: {model}")
                response = requests.post(
                    API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=180
                )
                
                if response.status_code == 200:
                    # Check if response is an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        print(f"✅ Logo generated successfully with {model}")
                        return output_path
                    else:
                        print(f"   Unexpected response type: {content_type}")
                        
                elif response.status_code == 503:
                    # Model loading
                    try:
                        data = response.json()
                        estimated_time = data.get('estimated_time', 30)
                        print(f"   Model loading, estimated time: {estimated_time}s")
                    except:
                        pass
                    continue
                    
                elif response.status_code == 500:
                    print(f"   Server error with {model}")
                    continue
                    
                else:
                    print(f"   API error {response.status_code}: {response.text[:200]}")
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"   Timeout with {model}")
                continue
            except Exception as e:
                print(f"   Error with {model}: {e}")
                continue
        
        raise Exception("All image generation models failed")
    
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
            
            if torch.cuda.is_available():
                self._image_pipeline = self._image_pipeline.to("cuda")
                self._image_pipeline.enable_attention_slicing()
        
        image = self._image_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        image.save(output_path)
        return output_path
    
    def _generate_professional_placeholder(self, output_path: str, prompt: str) -> str:
        """Generate a professional-looking placeholder logo."""
        if not PIL_AVAILABLE:
            print("PIL not available")
            return None
        
        # Parse prompt for context
        prompt_lower = prompt.lower()
        
        # Determine color scheme based on business type
        if any(word in prompt_lower for word in ['pizza', 'food', 'restaurant', 'bakery', 'cafe']):
            primary_color = (220, 53, 69)    # Red
            secondary_color = (255, 193, 7)   # Yellow/Gold
            accent_color = (40, 167, 69)      # Green
            icon_type = "food"
        elif any(word in prompt_lower for word in ['tech', 'software', 'digital', 'ai', 'app']):
            primary_color = (0, 123, 255)     # Blue
            secondary_color = (102, 126, 234) # Purple-blue
            accent_color = (23, 162, 184)     # Cyan
            icon_type = "tech"
        elif any(word in prompt_lower for word in ['eco', 'green', 'sustainable', 'organic', 'nature']):
            primary_color = (40, 167, 69)     # Green
            secondary_color = (32, 201, 151)  # Teal
            accent_color = (255, 193, 7)      # Yellow
            icon_type = "eco"
        elif any(word in prompt_lower for word in ['health', 'fitness', 'yoga', 'wellness']):
            primary_color = (111, 66, 193)    # Purple
            secondary_color = (232, 62, 140)  # Pink
            accent_color = (23, 162, 184)     # Cyan
            icon_type = "health"
        elif any(word in prompt_lower for word in ['bike', 'cycle', 'delivery', 'transport']):
            primary_color = (255, 128, 0)     # Orange
            secondary_color = (40, 167, 69)   # Green
            accent_color = (52, 58, 64)       # Dark gray
            icon_type = "delivery"
        else:
            primary_color = (102, 126, 234)   # Default purple-blue
            secondary_color = (118, 75, 162)  # Purple
            accent_color = (240, 147, 251)    # Light pink
            icon_type = "default"
        
        # Create the logo
        size = 512
        img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Create gradient background circle
        center = size // 2
        radius = 200
        
        # Draw outer glow
        for i in range(20, 0, -1):
            alpha = int(255 * (1 - i/20) * 0.3)
            glow_color = (*primary_color, alpha)
            draw.ellipse(
                [center - radius - i*3, center - radius - i*3,
                 center + radius + i*3, center + radius + i*3],
                fill=glow_color
            )
        
        # Draw main circle with gradient effect
        for i in range(radius, 0, -1):
            # Interpolate between primary and secondary colors
            t = i / radius
            r = int(primary_color[0] * t + secondary_color[0] * (1-t))
            g = int(primary_color[1] * t + secondary_color[1] * (1-t))
            b = int(primary_color[2] * t + secondary_color[2] * (1-t))
            
            draw.ellipse(
                [center - i, center - i, center + i, center + i],
                fill=(r, g, b, 255)
            )
        
        # Draw icon based on business type
        icon_color = (255, 255, 255)
        
        if icon_type == "food" or icon_type == "delivery":
            # Draw a pizza slice or delivery icon
            self._draw_pizza_icon(draw, center, icon_color, accent_color)
        elif icon_type == "tech":
            self._draw_tech_icon(draw, center, icon_color)
        elif icon_type == "eco":
            self._draw_leaf_icon(draw, center, icon_color)
        elif icon_type == "health":
            self._draw_wellness_icon(draw, center, icon_color)
        else:
            self._draw_abstract_icon(draw, center, icon_color)
        
        # Add subtle inner shadow
        for i in range(10):
            alpha = int(30 * (1 - i/10))
            draw.ellipse(
                [center - radius + i, center - radius + i,
                 center + radius - i, center + radius - i],
                outline=(0, 0, 0, alpha),
                width=1
            )
        
        # Save with white background for compatibility
        background = Image.new('RGB', (size, size), (255, 255, 255))
        background.paste(img, (0, 0), img)
        background.save(output_path, 'PNG', quality=95)
        
        return output_path
    
    def _draw_pizza_icon(self, draw, center, color, accent):
        """Draw a stylized pizza/food delivery icon."""
        # Pizza slice shape
        points = [
            (center, center - 60),
            (center - 50, center + 40),
            (center + 50, center + 40),
        ]
        draw.polygon(points, fill=color)
        
        # Pepperoni dots
        dot_positions = [
            (center - 15, center - 10),
            (center + 15, center),
            (center, center + 15),
        ]
        for pos in dot_positions:
            draw.ellipse(
                [pos[0] - 8, pos[1] - 8, pos[0] + 8, pos[1] + 8],
                fill=accent
            )
        
        # Speed lines (for delivery)
        for i in range(3):
            y = center - 30 + i * 25
            draw.line(
                [(center - 80 - i*10, y), (center - 60, y)],
                fill=(*color, 150),
                width=3
            )
    
    def _draw_tech_icon(self, draw, center, color):
        """Draw a tech/digital icon."""
        # Circuit-like pattern
        draw.rectangle(
            [center - 40, center - 40, center + 40, center + 40],
            outline=color,
            width=4
        )
        
        # Inner square
        draw.rectangle(
            [center - 20, center - 20, center + 20, center + 20],
            fill=color
        )
        
        # Connection lines
        lines = [
            [(center, center - 40), (center, center - 60)],
            [(center, center + 40), (center, center + 60)],
            [(center - 40, center), (center - 60, center)],
            [(center + 40, center), (center + 60, center)],
        ]
        for line in lines:
            draw.line(line, fill=color, width=4)
        
        # Corner dots
        for dx, dy in [(-60, -60), (60, -60), (-60, 60), (60, 60)]:
            draw.ellipse(
                [center + dx - 6, center + dy - 6,
                 center + dx + 6, center + dy + 6],
                fill=color
            )
    
    def _draw_leaf_icon(self, draw, center, color):
        """Draw an eco/leaf icon."""
        # Leaf shape using bezier approximation
        leaf_points = []
        for i in range(50):
            t = i / 49
            # Simple leaf curve
            x = center + 60 * math.sin(t * math.pi) * (1 - t * 0.5)
            y = center - 80 * t + 40
            leaf_points.append((x, y))
        
        for i in range(49, -1, -1):
            t = i / 49
            x = center - 60 * math.sin(t * math.pi) * (1 - t * 0.5)
            y = center - 80 * t + 40
            leaf_points.append((x, y))
        
        draw.polygon(leaf_points, fill=color)
        
        # Stem
        draw.line(
            [(center, center + 40), (center, center + 70)],
            fill=color,
            width=6
        )
    
    def _draw_wellness_icon(self, draw, center, color):
        """Draw a wellness/health icon."""
        # Person in yoga pose
        head_y = center - 50
        draw.ellipse(
            [center - 15, head_y - 15, center + 15, head_y + 15],
            fill=color
        )
        
        # Body
        draw.line(
            [(center, head_y + 15), (center, center + 20)],
            fill=color,
            width=6
        )
        
        # Arms up (yoga pose)
        draw.line(
            [(center, center - 20), (center - 40, center - 50)],
            fill=color,
            width=6
        )
        draw.line(
            [(center, center - 20), (center + 40, center - 50)],
            fill=color,
            width=6
        )
        
        # Legs
        draw.line(
            [(center, center + 20), (center - 30, center + 60)],
            fill=color,
            width=6
        )
        draw.line(
            [(center, center + 20), (center + 30, center + 60)],
            fill=color,
            width=6
        )
    
    def _draw_abstract_icon(self, draw, center, color):
        """Draw an abstract professional icon."""
        # Three overlapping circles
        positions = [
            (center - 25, center - 15),
            (center + 25, center - 15),
            (center, center + 20),
        ]
        
        for pos in positions:
            draw.ellipse(
                [pos[0] - 30, pos[1] - 30, pos[0] + 30, pos[1] + 30],
                outline=color,
                width=4
            )
        
        # Center dot
        draw.ellipse(
            [center - 10, center - 5, center + 10, center + 15],
            fill=color
        )
    
    def estimate_cost(
        self,
        text_tokens: int = 0,
        image_calls: int = 0
    ) -> Dict[str, float]:
        text_cost_per_1k = 0.0004
        image_cost_per_call = 0.02
        
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
    """Mock LLM for development/testing."""
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def invoke(self, prompt: Union[str, Dict]) -> str:
        if isinstance(prompt, dict):
            prompt_text = str(prompt)
        else:
            prompt_text = prompt
        
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
        return '''{"domains": [{"name": "brightidea", "tld": ".com", "rationale": "Captures innovation", "score": 95, "available": true}]}'''
    
    def _mock_hero_response(self) -> str:
        return '''{"brand_name": "BrightIdea", "headline": "Transform Your Vision Into Reality", "tagline": "Innovative solutions for modern challenges.", "bullets": ["Expert guidance", "Modern approach", "Dedicated support"]}'''
    
    def _mock_social_response(self) -> str:
        return '''{"posts": [{"platform": "twitter", "content": "Exciting news!", "hashtags": ["#Innovation"], "call_to_action": "Learn more!"}]}'''
    
    def _mock_logo_prompt(self) -> str:
        return "Modern minimalist logo, abstract geometric shape, gradient colors, clean lines, professional"
    
    def __call__(self, prompt: Union[str, Dict]) -> str:
        return self.invoke(prompt)