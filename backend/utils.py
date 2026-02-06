"""
Utility Functions for AI Brand Studio
=====================================
Common helper functions used throughout the application.
"""

import os
import re
import uuid
import hashlib
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Session & ID Generation
# ============================================

def generate_session_id() -> str:
    """
    Generate a unique session identifier.
    
    Returns:
        8-character unique session ID
    """
    return str(uuid.uuid4())[:8]


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique identifier with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    unique_part = str(uuid.uuid4()).replace("-", "")[:12]
    if prefix:
        return f"{prefix}_{unique_part}"
    return unique_part


def generate_deterministic_id(seed: str) -> str:
    """
    Generate a deterministic ID based on a seed string.
    Useful for caching and reproducibility.
    
    Args:
        seed: Seed string for ID generation
        
    Returns:
        Deterministic 8-character ID
    """
    return hashlib.md5(seed.encode()).hexdigest()[:8]


# ============================================
# Input Sanitization & Validation
# ============================================

def sanitize_input(text: str, max_length: int = 500) -> str:
    """
    Sanitize user input by removing potentially harmful content.
    
    Args:
        text: Raw user input
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    # Remove potentially harmful characters/patterns
    # Remove script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove style tags
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove control characters (except newlines and tabs)
    text = ''.join(char for char in text if char == '\n' or char == '\t' or not unicodedata_iscntrl(char))
    
    return text.strip()


def unicodedata_iscntrl(char: str) -> bool:
    """Check if a character is a control character."""
    import unicodedata
    try:
        return unicodedata.category(char) == 'Cc'
    except:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def validate_business_description(description: str) -> Dict[str, Any]:
    """
    Validate a business description for quality and safety.
    
    Args:
        description: Business description to validate
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "score": 100,
        "cleaned_text": ""
    }
    
    if not description:
        result["valid"] = False
        result["errors"].append("Description is required")
        result["score"] = 0
        return result
    
    # Clean the description
    cleaned = sanitize_input(description)
    result["cleaned_text"] = cleaned
    
    # Check minimum length
    if len(cleaned) < 20:
        result["valid"] = False
        result["errors"].append("Description too short (minimum 20 characters)")
        result["score"] -= 50
    
    # Check maximum length
    if len(description) > 500:
        result["warnings"].append("Description truncated to 500 characters")
        result["score"] -= 10
    
    # Check for useful content
    word_count = len(cleaned.split())
    if word_count < 5:
        result["warnings"].append("Description could be more detailed (less than 5 words)")
        result["score"] -= 20
    elif word_count < 10:
        result["warnings"].append("Consider adding more details for better results")
        result["score"] -= 10
    
    # Check for common spam patterns
    spam_patterns = [
        r'(?:buy|click|free|winner|congratulations)\s+now',
        r'\b(?:viagra|casino|lottery|porn)\b',
        r'(?:make\s+money\s+fast)',
        r'(?:work\s+from\s+home.*\$\d+)',
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            result["valid"] = False
            result["errors"].append("Description contains prohibited content")
            result["score"] = 0
            break
    
    # Check for excessive caps
    if cleaned.isupper() and len(cleaned) > 20:
        result["warnings"].append("Avoid using all capital letters")
        result["score"] -= 10
    
    # Check for excessive repetition
    words = cleaned.lower().split()
    if words:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        max_freq = max(word_freq.values())
        if max_freq > len(words) * 0.3 and max_freq > 3:
            result["warnings"].append("Description contains excessive word repetition")
            result["score"] -= 15
    
    # Ensure score doesn't go below 0
    result["score"] = max(0, result["score"])
    
    return result


def check_content_safety(text: str) -> Dict[str, Any]:
    """
    Check content for safety issues.
    
    Args:
        text: Text to check
        
    Returns:
        Safety check results
    """
    result = {
        "safe": True,
        "issues": [],
        "categories": []
    }
    
    # Patterns for potentially unsafe content
    unsafe_patterns = {
        "violence": [
            r'\b(?:kill|murder|attack|weapon|bomb)\b',
        ],
        "hate_speech": [
            r'\b(?:hate|racist|sexist)\b.*\b(?:all|every)\b',
        ],
        "adult_content": [
            r'\b(?:adult|xxx|nsfw|explicit)\b',
        ],
        "illegal": [
            r'\b(?:illegal|drugs|hack|pirate)\b',
        ],
        "spam": [
            r'(?:click\s+here|free\s+money|act\s+now)',
            r'(?:\$\d+.*(?:day|hour|week))',
        ]
    }
    
    text_lower = text.lower()
    
    for category, patterns in unsafe_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                result["safe"] = False
                result["issues"].append(f"Potential {category} content detected")
                if category not in result["categories"]:
                    result["categories"].append(category)
    
    return result


# ============================================
# Cost Estimation
# ============================================

def calculate_cost_estimate(results: Dict[str, Any]) -> float:
    """
    Calculate estimated cost of a generation run.
    
    Args:
        results: Generation results dictionary
        
    Returns:
        Estimated cost in USD
    """
    # Approximate token counts
    token_estimates = {
        'domains': 500,      # Input + output tokens for domain generation
        'hero': 400,         # Hero copy generation
        'social_posts': 600, # Social posts generation
        'logo_prompt': 200,  # Logo prompt generation
    }
    
    # Cost per 1K tokens (approximate for HF inference)
    cost_per_1k_tokens = 0.0004
    
    # Image generation cost (approximate)
    image_cost = 0.02
    
    # Calculate text costs
    total_tokens = 0
    
    if results.get('domains'):
        total_tokens += token_estimates['domains']
    
    if results.get('hero'):
        total_tokens += token_estimates['hero']
    
    if results.get('social_posts'):
        total_tokens += token_estimates['social_posts']
    
    if results.get('logo_path'):
        total_tokens += token_estimates['logo_prompt']
    
    text_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    # Add image cost if logo was generated
    total_cost = text_cost
    if results.get('logo_path'):
        total_cost += image_cost
    
    return round(total_cost, 4)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Rough approximation: ~4 characters per token for English.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimation: average of character-based and word-based
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) * 1.3
    
    return int((char_estimate + word_estimate) / 2)


def calculate_detailed_cost(
    input_tokens: int,
    output_tokens: int,
    image_generations: int = 0,
    model_type: str = "default"
) -> Dict[str, float]:
    """
    Calculate detailed cost breakdown.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        image_generations: Number of images generated
        model_type: Type of model used
        
    Returns:
        Detailed cost breakdown
    """
    # Cost rates (per 1K tokens)
    rates = {
        "default": {"input": 0.0004, "output": 0.0004},
        "premium": {"input": 0.001, "output": 0.002},
        "economy": {"input": 0.0001, "output": 0.0001},
    }
    
    rate = rates.get(model_type, rates["default"])
    
    input_cost = (input_tokens / 1000) * rate["input"]
    output_cost = (output_tokens / 1000) * rate["output"]
    image_cost = image_generations * 0.02
    
    total = input_cost + output_cost + image_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "image_generations": image_generations,
        "image_cost": round(image_cost, 4),
        "total_cost": round(total, 4),
        "model_type": model_type
    }


# ============================================
# Time & Duration Formatting
# ============================================

def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 0:
        return "0ms"
    
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_timestamp(dt: datetime = None, format_type: str = "iso") -> str:
    """
    Format a datetime object to string.
    
    Args:
        dt: Datetime object (defaults to now)
        format_type: Format type ('iso', 'readable', 'filename')
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    
    formats = {
        "iso": "%Y-%m-%dT%H:%M:%S",
        "readable": "%B %d, %Y at %I:%M %p",
        "filename": "%Y%m%d_%H%M%S",
        "date_only": "%Y-%m-%d",
        "time_only": "%H:%M:%S",
    }
    
    fmt = formats.get(format_type, formats["iso"])
    return dt.strftime(fmt)


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse a timestamp string to datetime object.
    
    Args:
        timestamp_str: Timestamp string to parse
        
    Returns:
        Datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    return None


# ============================================
# Text Processing
# ============================================

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for categorization.
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    # Common stop words to filter out
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'that', 'this', 'these', 'those', 'it', 'its', "it's", 'i', 'you',
        'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
        'your', 'his', 'our', 'their', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'now', 'here', 'there', 'then', 'once', 'if', 'because', 'until',
        'while', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'any', 'am', "i'm", "you're",
        "we're", "they're", "i've", "you've", "we've", "they've", "i'd",
        "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
        "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't",
        "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't",
        "cannot", "couldn't", "mustn't", "let's", "that's", "who's",
        "what's", "here's", "there's", "when's", "where's", "why's",
        "how's", 'like', 'get', 'got', 'getting', 'make', 'made', 'making'
    }
    
    # Tokenize and clean
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, freq in sorted_words[:max_keywords]]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding suffix if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Try to truncate at a word boundary
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:  # Only use word boundary if not too far back
        truncated = truncated[:last_space]
    
    return truncated + suffix


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a URL-friendly slug.
    
    Args:
        text: Text to slugify
        max_length: Maximum length of slug
        
    Returns:
        URL-friendly slug
    """
    # Convert to lowercase
    slug = text.lower()
    
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    
    # Remove non-alphanumeric characters (except hyphens)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    
    # Remove multiple consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    
    # Truncate
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug


def capitalize_words(text: str) -> str:
    """
    Capitalize the first letter of each word.
    
    Args:
        text: Text to capitalize
        
    Returns:
        Capitalized text
    """
    # Words that shouldn't be capitalized (unless first)
    small_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 
                   'on', 'at', 'to', 'by', 'of', 'in'}
    
    words = text.lower().split()
    result = []
    
    for i, word in enumerate(words):
        if i == 0 or word not in small_words:
            result.append(word.capitalize())
        else:
            result.append(word)
    
    return ' '.join(result)


# ============================================
# JSON Utilities
# ============================================

def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON from text, with fallback to default.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default
    """
    if not text:
        return default
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object/array in text
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        json_match = re.search(pattern, text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                continue
    
    return default


def pretty_json(data: Any, indent: int = 2) -> str:
    """
    Convert data to pretty-printed JSON string.
    
    Args:
        data: Data to convert
        indent: Indentation level
        
    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


# ============================================
# Decorators
# ============================================

def timer(func):
    """
    Decorator to time function execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {format_duration(end - start)}")
        return result
    return wrapper


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def log_call(func):
    """
    Decorator to log function calls.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    return wrapper


# ============================================
# Validation Helpers
# ============================================

def is_valid_email(email: str) -> bool:
    """
    Check if a string is a valid email address.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Check if a string is a valid URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[\w=&-]*)?$'
    return bool(re.match(pattern, url))


def is_valid_domain(domain: str) -> bool:
    """
    Check if a string is a valid domain name.
    
    Args:
        domain: Domain string to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Remove protocol if present
    domain = re.sub(r'^https?://', '', domain)
    
    # Remove path if present
    domain = domain.split('/')[0]
    
    # Check domain pattern
    pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    return bool(re.match(pattern, domain))


# ============================================
# Color Utilities
# ============================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string (with or without #)
        
    Returns:
        RGB tuple (r, g, b)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color.
    
    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)
        
    Returns:
        Hex color string with #
    """
    return f'#{r:02x}{g:02x}{b:02x}'


def generate_color_palette(seed: str, num_colors: int = 5) -> List[str]:
    """
    Generate a deterministic color palette from a seed string.
    
    Args:
        seed: Seed string for palette generation
        num_colors: Number of colors to generate
        
    Returns:
        List of hex color strings
    """
    # Use hash for deterministic generation
    hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    
    colors = []
    for i in range(num_colors):
        # Generate hue based on hash and index
        hue = ((hash_val + i * 137) % 360) / 360
        
        # Use golden ratio for pleasing distribution
        saturation = 0.5 + (((hash_val >> (i * 4)) & 0xF) / 32)
        lightness = 0.4 + (((hash_val >> (i * 4 + 2)) & 0xF) / 40)
        
        # HSL to RGB conversion
        r, g, b = hsl_to_rgb(hue, saturation, lightness)
        colors.append(rgb_to_hex(r, g, b))
    
    return colors


def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """
    Convert HSL to RGB.
    
    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        l: Lightness (0-1)
        
    Returns:
        RGB tuple (0-255 for each)
    """
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    return (int(r * 255), int(g * 255), int(b * 255))


# ============================================
# Analytics & Metrics
# ============================================

class MetricsCollector:
    """
    Simple metrics collector for tracking application metrics.
    """
    
    def __init__(self):
        self.metrics = {}
        self.events = []
    
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
        self.metrics[metric_name] += value
    
    def set_gauge(self, metric_name: str, value: float):
        """Set a gauge metric."""
        self.metrics[metric_name] = value
    
    def record_event(self, event_name: str, data: Dict[str, Any] = None):
        """Record an event."""
        self.events.append({
            "event": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "metrics": self.metrics.copy(),
            "events_count": len(self.events),
            "collected_at": datetime.now().isoformat()
        }
    
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events."""
        return self.events[-limit:]
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.events = []


# Global metrics collector instance
metrics_collector = MetricsCollector()


# ============================================
# Environment & Configuration
# ============================================

def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional default and required check.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default
        
    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    
    if value is None and required:
        raise ValueError(f"Required environment variable '{key}' is not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        
    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        
    Returns:
        Integer value
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_list(key: str, default: List[str] = None, separator: str = ",") -> List[str]:
    """
    Get list environment variable.
    
    Args:
        key: Environment variable name
        default: Default value
        separator: List separator
        
    Returns:
        List of strings
    """
    value = os.getenv(key)
    if value is None:
        return default or []
    
    return [item.strip() for item in value.split(separator) if item.strip()]