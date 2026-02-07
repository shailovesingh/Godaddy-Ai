"""

This package contains all tests for the AI Brand Studio application.

Test Categories:
- Unit tests: Test individual functions and classes
- Integration tests: Test component interactions
- E2E tests: Test complete workflows

Usage:
    pytest tests/ -v                    # Run all tests
    pytest tests/test_chains.py -v      # Run specific test file
    pytest tests/ -v -k "domain"        # Run tests matching pattern
    pytest tests/ -v --cov=backend      # Run with coverage
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("HF_TOKEN", "test_token")
os.environ.setdefault("STORAGE_PATH", "/tmp/ai_brand_studio_tests")
os.environ.setdefault("USE_HF_INFERENCE_API", "false")  # Use mocks in tests


# Test configuration
class TestConfig:
    """Configuration for test suite."""
    
    # Test data directory
    TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
    
    # Sample prompts for testing
    SAMPLE_PROMPTS = [
        "A cozy bakery in Portland specializing in sourdough bread",
        "Tech startup building AI tools for small businesses",
        "Sustainable fashion brand using recycled materials",
    ]
    
    # Expected output keys
    EXPECTED_DOMAIN_KEYS = {"name", "tld", "rationale", "score", "available"}
    EXPECTED_HERO_KEYS = {"brand_name", "headline", "tagline", "bullets"}
    EXPECTED_SOCIAL_KEYS = {"platform", "content", "hashtags", "call_to_action"}
    
    # Timeouts
    GENERATION_TIMEOUT = 60  # seconds
    API_TIMEOUT = 30  # seconds
    
    # Quality thresholds
    MIN_DOMAIN_SUGGESTIONS = 3
    MIN_SOCIAL_POSTS = 2
    MIN_HEADLINE_LENGTH = 10
    MAX_HEADLINE_LENGTH = 200


# Fixtures available to all tests
import pytest


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration to all tests."""
    return TestConfig()


@pytest.fixture(scope="session")
def sample_description():
    """Provide a sample business description."""
    return "A cozy artisan bakery in downtown Portland specializing in sourdough bread and French pastries with organic ingredients"


@pytest.fixture(scope="function")
def temp_storage_dir(tmp_path):
    """Provide a temporary storage directory for each test."""
    storage_dir = tmp_path / "ai_brand_studio"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture(scope="session")
def mock_llm_response():
    """Provide mock LLM responses for testing."""
    return {
        "domains": {
            "domains": [
                {"name": "portlandbakes", "tld": ".com", "rationale": "Location + service", "score": 95, "available": True},
                {"name": "sourdoughpdx", "tld": ".com", "rationale": "Product + location", "score": 90, "available": True},
                {"name": "artisanbread", "tld": ".com", "rationale": "Describes offering", "score": 85, "available": False},
            ]
        },
        "hero": {
            "brand_name": "Portland Bakes",
            "headline": "Artisan Sourdough, Crafted with Love",
            "tagline": "Experience the authentic taste of handcrafted bread made with organic ingredients and traditional techniques.",
            "bullets": [
                "100% organic, locally-sourced ingredients",
                "Traditional 72-hour fermentation process",
                "Fresh-baked daily in our Portland kitchen"
            ]
        },
        "social_posts": {
            "posts": [
                {
                    "platform": "twitter",
                    "content": "Fresh sourdough coming out of the oven! 🍞 Stop by Portland Bakes for the best artisan bread in town.",
                    "hashtags": ["#PortlandBakes", "#Sourdough", "#ArtisanBread"],
                    "call_to_action": "Visit us today!"
                },
                {
                    "platform": "instagram",
                    "content": "The smell of fresh bread in the morning is pure magic. ✨ Our bakers start at 4am to bring you the perfect loaf.",
                    "hashtags": ["#BakeryLife", "#FreshBread", "#Portland", "#Organic"],
                    "call_to_action": "Link in bio for our menu!"
                }
            ]
        }
    }


# Utility functions for tests
def assert_valid_json(data: dict, required_keys: set, context: str = ""):
    """Assert that data is valid JSON with required keys."""
    assert isinstance(data, dict), f"{context}: Expected dict, got {type(data)}"
    missing_keys = required_keys - set(data.keys())
    assert not missing_keys, f"{context}: Missing required keys: {missing_keys}"


def assert_non_empty_string(value: str, min_length: int = 1, context: str = ""):
    """Assert that value is a non-empty string."""
    assert isinstance(value, str), f"{context}: Expected string, got {type(value)}"
    assert len(value) >= min_length, f"{context}: String too short (min {min_length})"


def assert_valid_list(data: list, min_items: int = 1, context: str = ""):
    """Assert that data is a non-empty list."""
    assert isinstance(data, list), f"{context}: Expected list, got {type(data)}"
    assert len(data) >= min_items, f"{context}: List too short (min {min_items} items)"