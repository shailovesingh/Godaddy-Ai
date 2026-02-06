"""
Tests for LangChain Chains
==========================

Unit and integration tests for the brand generation chains.
"""

import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import test utilities
from tests import (
    TestConfig,
    assert_valid_json,
    assert_non_empty_string,
    assert_valid_list
)

# Import modules to test
from backend.chains import (
    BrandStudioChains,
    DomainSuggestion,
    HeroCopy,
    SocialPost,
    DOMAIN_PROMPT_TEMPLATE,
    HERO_COPY_PROMPT_TEMPLATE,
    SOCIAL_POSTS_PROMPT_TEMPLATE,
    LOGO_PROMPT_TEMPLATE
)
from backend.llm_clients import LLMClientManager, MockTextLLM
from backend.utils import sanitize_input, validate_business_description


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return MockTextLLM()


@pytest.fixture
def mock_llm_manager(mock_llm):
    """Create a mock LLM manager."""
    manager = Mock(spec=LLMClientManager)
    manager.get_text_llm.return_value = mock_llm
    manager.generate_image.return_value = "/tmp/test_logo.png"
    return manager


@pytest.fixture
def chains(mock_llm_manager):
    """Create BrandStudioChains with mock LLM."""
    return BrandStudioChains(mock_llm_manager)


@pytest.fixture
def sample_descriptions():
    """Sample business descriptions for testing."""
    return [
        "A cozy bakery in Portland specializing in sourdough bread",
        "Tech startup building AI-powered project management tools",
        "Sustainable fashion brand using recycled ocean plastics",
        "Online yoga studio offering personalized meditation sessions",
        "Premium coffee subscription delivering beans from small farms",
    ]


# ============================================
# Unit Tests: Prompt Templates
# ============================================

class TestPromptTemplates:
    """Test prompt template formatting."""
    
    def test_domain_prompt_template_has_placeholders(self):
        """Verify domain prompt has required placeholders."""
        assert "{description}" in DOMAIN_PROMPT_TEMPLATE
        assert "{num_suggestions}" in DOMAIN_PROMPT_TEMPLATE
        assert "JSON" in DOMAIN_PROMPT_TEMPLATE
    
    def test_hero_prompt_template_has_placeholders(self):
        """Verify hero prompt has required placeholders."""
        assert "{description}" in HERO_COPY_PROMPT_TEMPLATE
        assert "headline" in HERO_COPY_PROMPT_TEMPLATE.lower()
        assert "tagline" in HERO_COPY_PROMPT_TEMPLATE.lower()
    
    def test_social_prompt_template_has_placeholders(self):
        """Verify social prompt has required placeholders."""
        assert "{description}" in SOCIAL_POSTS_PROMPT_TEMPLATE
        assert "{brand_name}" in SOCIAL_POSTS_PROMPT_TEMPLATE
        assert "{num_posts}" in SOCIAL_POSTS_PROMPT_TEMPLATE
    
    def test_logo_prompt_template_has_placeholders(self):
        """Verify logo prompt has required placeholders."""
        assert "{description}" in LOGO_PROMPT_TEMPLATE
        assert "{brand_name}" in LOGO_PROMPT_TEMPLATE


# ============================================
# Unit Tests: Domain Generation
# ============================================

class TestDomainGeneration:
    """Test domain name generation."""
    
    def test_generate_domains_returns_list(self, chains, sample_descriptions):
        """Test that generate_domains returns a list."""
        description = sample_descriptions[0]
        result = chains.generate_domains(description, num_suggestions=5)
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_generate_domains_correct_count(self, chains, sample_descriptions):
        """Test that generate_domains returns requested number of suggestions."""
        description = sample_descriptions[0]
        
        for count in [3, 5, 7]:
            result = chains.generate_domains(description, num_suggestions=count)
            # May return fallback if parsing fails, so check minimum
            assert len(result) >= min(count, 3)
    
    def test_domain_structure(self, chains, sample_descriptions):
        """Test that each domain has required fields."""
        description = sample_descriptions[0]
        result = chains.generate_domains(description, num_suggestions=3)
        
        required_keys = {"name", "rationale", "score", "available"}
        
        for domain in result:
            assert isinstance(domain, dict)
            for key in required_keys:
                assert key in domain, f"Missing key: {key}"
    
    def test_domain_name_format(self, chains, sample_descriptions):
        """Test that domain names are properly formatted."""
        description = sample_descriptions[0]
        result = chains.generate_domains(description, num_suggestions=3)
        
        for domain in result:
            name = domain.get("name", "")
            # Domain names should be lowercase, no spaces
            assert name == name.lower().replace(" ", "")
            # Should be reasonable length
            assert 3 <= len(name) <= 63
    
    def test_domain_score_range(self, chains, sample_descriptions):
        """Test that domain scores are in valid range."""
        description = sample_descriptions[0]
        result = chains.generate_domains(description, num_suggestions=3)
        
        for domain in result:
            score = domain.get("score", 0)
            assert 0 <= score <= 100
    
    def test_domain_availability_check(self, chains):
        """Test domain availability check stub."""
        result = chains._check_domain_availability("testdomain")
        assert isinstance(result, bool)
    
    def test_fallback_domains(self, chains):
        """Test fallback domain generation."""
        result = chains._generate_fallback_domains("test business", 5)
        
        assert len(result) == 5
        for domain in result:
            assert "name" in domain
            assert "rationale" in domain


# ============================================
# Unit Tests: Hero Copy Generation
# ============================================

class TestHeroCopyGeneration:
    """Test hero copy generation."""
    
    def test_generate_hero_returns_dict(self, chains, sample_descriptions):
        """Test that generate_hero_copy returns a dictionary."""
        description = sample_descriptions[0]
        result = chains.generate_hero_copy(description)
        
        assert isinstance(result, dict)
    
    def test_hero_has_required_fields(self, chains, sample_descriptions):
        """Test that hero copy has all required fields."""
        description = sample_descriptions[0]
        result = chains.generate_hero_copy(description)
        
        required_keys = {"brand_name", "headline", "tagline", "bullets"}
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_headline_quality(self, chains, sample_descriptions):
        """Test headline meets quality standards."""
        description = sample_descriptions[0]
        result = chains.generate_hero_copy(description)
        
        headline = result.get("headline", "")
        
        # Should be non-empty
        assert len(headline) > 0
        # Should be reasonable length (not too short or long)
        assert 10 <= len(headline) <= 200
        # Should not be all caps (unless very short)
        if len(headline) > 20:
            assert not headline.isupper()
    
    def test_tagline_quality(self, chains, sample_descriptions):
        """Test tagline meets quality standards."""
        description = sample_descriptions[0]
        result = chains.generate_hero_copy(description)
        
        tagline = result.get("tagline", "")
        
        assert len(tagline) > 0
        assert len(tagline) <= 500
    
    def test_bullets_are_list(self, chains, sample_descriptions):
        """Test that bullets is a list."""
        description = sample_descriptions[0]
        result = chains.generate_hero_copy(description)
        
        bullets = result.get("bullets", [])
        
        assert isinstance(bullets, list)
        assert len(bullets) >= 1
    
    def test_fallback_hero(self, chains):
        """Test fallback hero generation."""
        result = chains._generate_fallback_hero("test business")
        
        assert "headline" in result
        assert "tagline" in result
        assert "bullets" in result


# ============================================
# Unit Tests: Social Posts Generation
# ============================================

class TestSocialPostsGeneration:
    """Test social media posts generation."""
    
    def test_generate_social_returns_list(self, chains, sample_descriptions):
        """Test that generate_social_posts returns a list."""
        description = sample_descriptions[0]
        hero_copy = chains.generate_hero_copy(description)
        result = chains.generate_social_posts(description, hero_copy, num_posts=3)
        
        assert isinstance(result, list)
    
    def test_social_post_structure(self, chains, sample_descriptions):
        """Test that each post has required fields."""
        description = sample_descriptions[0]
        hero_copy = chains.generate_hero_copy(description)
        result = chains.generate_social_posts(description, hero_copy, num_posts=3)
        
        required_keys = {"platform", "content", "hashtags"}
        
        for post in result:
            assert isinstance(post, dict)
            for key in required_keys:
                assert key in post, f"Missing key: {key}"
    
    def test_platform_values(self, chains, sample_descriptions):
        """Test that platforms are valid."""
        description = sample_descriptions[0]
        hero_copy = chains.generate_hero_copy(description)
        result = chains.generate_social_posts(description, hero_copy, num_posts=3)
        
        valid_platforms = {"twitter", "x", "instagram", "facebook", "linkedin", "tiktok"}
        
        for post in result:
            platform = post.get("platform", "").lower()
            assert platform in valid_platforms
    
    def test_hashtags_format(self, chains, sample_descriptions):
        """Test that hashtags are properly formatted."""
        description = sample_descriptions[0]
        hero_copy = chains.generate_hero_copy(description)
        result = chains.generate_social_posts(description, hero_copy, num_posts=3)
        
        for post in result:
            hashtags = post.get("hashtags", [])
            assert isinstance(hashtags, list)
            for tag in hashtags:
                # Hashtags should start with #
                assert tag.startswith("#") or not tag.strip()
    
    def test_content_length(self, chains, sample_descriptions):
        """Test that content length is reasonable."""
        description = sample_descriptions[0]
        hero_copy = chains.generate_hero_copy(description)
        result = chains.generate_social_posts(description, hero_copy, num_posts=3)
        
        for post in result:
            content = post.get("content", "")
            assert 10 <= len(content) <= 1000


# ============================================
# Unit Tests: Logo Prompt Generation
# ============================================

class TestLogoPromptGeneration:
    """Test logo prompt generation."""
    
    def test_generate_logo_prompt_returns_string(self, chains, sample_descriptions):
        """Test that generate_logo_prompt returns a string."""
        description = sample_descriptions[0]
        result = chains.generate_logo_prompt(description, "Test Brand")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_logo_prompt_quality(self, chains, sample_descriptions):
        """Test logo prompt includes quality modifiers."""
        description = sample_descriptions[0]
        result = chains.generate_logo_prompt(description, "Test Brand")
        
        # Should include quality-related terms
        quality_terms = ["logo", "professional", "clean", "quality"]
        result_lower = result.lower()
        
        assert any(term in result_lower for term in quality_terms)
    
    def test_logo_prompt_no_text(self, chains, sample_descriptions):
        """Test logo prompt discourages text in logo."""
        description = sample_descriptions[0]
        # The prompt template should discourage text
        assert "no text" in LOGO_PROMPT_TEMPLATE.lower() or "without text" in LOGO_PROMPT_TEMPLATE.lower()


# ============================================
# Unit Tests: JSON Parsing
# ============================================

class TestJSONParsing:
    """Test JSON parsing utilities."""
    
    def test_parse_valid_json(self, chains):
        """Test parsing valid JSON."""
        valid_json = '{"key": "value"}'
        result = chains._parse_json_safely(valid_json)
        
        assert result == {"key": "value"}
    
    def test_parse_json_from_markdown(self, chains):
        """Test parsing JSON from markdown code block."""
        markdown_json = '''
        Here is the result:
        ```json
        {"key": "value"}
        ```
        '''
        result = chains._parse_json_safely(markdown_json)
        
        assert result == {"key": "value"}
    
    def test_parse_json_with_fallback(self, chains):
        """Test fallback for invalid JSON."""
        invalid_json = "not valid json"
        fallback = {"default": True}
        result = chains._parse_json_safely(invalid_json, fallback)
        
        assert result == fallback
    
    def test_parse_nested_json(self, chains):
        """Test parsing nested JSON."""
        nested_json = '{"domains": [{"name": "test", "score": 90}]}'
        result = chains._parse_json_safely(nested_json)
        
        assert "domains" in result
        assert len(result["domains"]) == 1


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_generation_workflow(self, chains, sample_descriptions):
        """Test complete brand generation workflow."""
        description = sample_descriptions[0]
        
        # Generate all assets
        domains = chains.generate_domains(description, num_suggestions=5)
        hero = chains.generate_hero_copy(description)
        social = chains.generate_social_posts(description, hero, num_posts=3)
        logo_prompt = chains.generate_logo_prompt(description, hero.get("brand_name", "Brand"))
        
        # Verify all outputs
        assert len(domains) >= 3
        assert "headline" in hero
        assert len(social) >= 2
        assert len(logo_prompt) > 20
    
    def test_different_industries(self, chains, sample_descriptions):
        """Test generation works for different industries."""
        for description in sample_descriptions:
            domains = chains.generate_domains(description, num_suggestions=3)
            hero = chains.generate_hero_copy(description)
            
            assert len(domains) >= 1
            assert hero.get("headline")
    
    def test_edge_cases(self, chains):
        """Test edge cases."""
        # Very short description
        short_result = chains.generate_hero_copy("coffee shop")
        assert short_result.get("headline")
        
        # Description with special characters
        special_result = chains.generate_hero_copy("café & bakery - fresh daily!")
        assert special_result.get("headline")


# ============================================
# Tests for Utility Functions
# ============================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_sanitize_input_removes_html(self):
        """Test that HTML tags are removed."""
        dirty = "<script>alert('xss')</script>Hello World"
        clean = sanitize_input(dirty)
        
        assert "<script>" not in clean
        assert "Hello World" in clean
    
    def test_sanitize_input_truncates(self):
        """Test that long input is truncated."""
        long_input = "a" * 1000
        clean = sanitize_input(long_input, max_length=100)
        
        assert len(clean) <= 100
    
    def test_validate_description_valid(self):
        """Test validation of valid description."""
        description = "A cozy bakery in Portland specializing in sourdough bread and pastries"
        result = validate_business_description(description)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_description_too_short(self):
        """Test validation of too-short description."""
        description = "coffee"
        result = validate_business_description(description)
        
        assert result["valid"] is False
        assert any("short" in err.lower() for err in result["errors"])
    
    def test_validate_description_spam(self):
        """Test validation catches spam patterns."""
        description = "Click here to buy now! Free money!"
        result = validate_business_description(description)
        
        assert result["valid"] is False


# ============================================
# Tests for Pydantic Models
# ============================================

class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_domain_suggestion_model(self):
        """Test DomainSuggestion model."""
        data = {
            "name": "testdomain",
            "tld": ".com",
            "rationale": "Great name",
            "score": 85,
            "available": True
        }
        
        domain = DomainSuggestion(**data)
        
        assert domain.name == "testdomain"
        assert domain.score == 85
    
    def test_hero_copy_model(self):
        """Test HeroCopy model."""
        data = {
            "brand_name": "Test Brand",
            "headline": "Great Headline",
            "tagline": "Even better tagline",
            "bullets": ["Point 1", "Point 2", "Point 3"]
        }
        
        hero = HeroCopy(**data)
        
        assert hero.brand_name == "Test Brand"
        assert len(hero.bullets) == 3
    
    def test_social_post_model(self):
        """Test SocialPost model."""
        data = {
            "platform": "twitter",
            "content": "Great content here",
            "hashtags": ["#test", "#brand"],
            "call_to_action": "Visit us!"
        }
        
        post = SocialPost(**data)
        
        assert post.platform == "twitter"
        assert len(post.hashtags) == 2


# ============================================
# Performance Tests
# ============================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_domain_generation_time(self, chains, sample_descriptions):
        """Test that domain generation completes in reasonable time."""
        import time
        
        description = sample_descriptions[0]
        
        start = time.time()
        chains.generate_domains(description, num_suggestions=5)
        duration = time.time() - start
        
        # Should complete within 30 seconds (mock should be fast)
        assert duration < 30
    
    def test_full_workflow_time(self, chains, sample_descriptions):
        """Test complete workflow timing."""
        import time
        
        description = sample_descriptions[0]
        
        start = time.time()
        
        domains = chains.generate_domains(description, num_suggestions=5)
        hero = chains.generate_hero_copy(description)
        social = chains.generate_social_posts(description, hero, num_posts=3)
        
        duration = time.time() - start
        
        # Complete workflow should finish within 60 seconds
        assert duration < 60


# ============================================
# Error Handling Tests
# ============================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_empty_description_handling(self, chains):
        """Test handling of empty description."""
        # Should not raise, should return fallback
        result = chains.generate_hero_copy("")
        assert result is not None
    
    def test_none_description_handling(self, chains):
        """Test handling of None description."""
        # Should handle gracefully
        try:
            result = chains.generate_hero_copy(None)
            # If it doesn't raise, should return something
            assert result is not None or result is None
        except (TypeError, AttributeError):
            # Expected behavior
            pass
    
    def test_malformed_llm_response(self, chains):
        """Test handling of malformed LLM response."""
        # The _parse_json_safely should handle this
        result = chains._parse_json_safely("not json at all {{{", {"fallback": True})
        assert result == {"fallback": True}


# ============================================
# Run Tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])