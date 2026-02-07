"""
Defines all prompt templates and chains for brand asset generation.
"""

import json
import re
from typing import Dict, List, Any, Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field


# Output Schemas (Pydantic models)

class DomainSuggestion(BaseModel):
    """Schema for a single domain suggestion."""
    name: str = Field(description="The domain name (without TLD)")
    tld: str = Field(default=".com", description="Top-level domain")
    rationale: str = Field(description="Why this name works for the business")
    score: int = Field(description="Memorability score 0-100")
    available: bool = Field(default=True, description="Simulated availability")


class DomainSuggestions(BaseModel):
    """Schema for multiple domain suggestions."""
    domains: List[DomainSuggestion] = Field(description="List of domain suggestions")


class HeroCopy(BaseModel):
    """Schema for hero section copy."""
    brand_name: str = Field(description="Suggested brand name")
    headline: str = Field(description="Main headline (8-12 words)")
    tagline: str = Field(description="Tagline/subheadline (15-20 words)")
    bullets: List[str] = Field(description="3 key value propositions")


class SocialPost(BaseModel):
    """Schema for a social media post."""
    platform: str = Field(description="Target platform: twitter, instagram, linkedin")
    content: str = Field(description="Post content")
    hashtags: List[str] = Field(description="Relevant hashtags")
    call_to_action: str = Field(description="CTA for the post")


class SocialPosts(BaseModel):
    """Schema for multiple social posts."""
    posts: List[SocialPost] = Field(description="List of social media posts")


# Prompt Templates

DOMAIN_PROMPT_TEMPLATE = """You are a creative domain name generator specializing in memorable, brandable names.

Given the following business description, generate {num_suggestions} unique domain name suggestions.

Business Description:
{description}

Requirements:
- Names should be short (ideally 6-12 characters)
- Easy to spell and pronounce
- Memorable and brandable
- Relevant to the business concept
- Available as .com (assume all are available for this exercise)

Return your response as valid JSON matching this exact format:
{{
    "domains": [
        {{
            "name": "domainname",
            "tld": ".com",
            "rationale": "Brief explanation of why this name works",
            "score": 85,
            "available": true
        }}
    ]
}}

Generate exactly {num_suggestions} domain suggestions, ranked by memorability score (highest first).
Only return the JSON, no additional text."""


HERO_COPY_PROMPT_TEMPLATE = """You are an expert copywriter specializing in brand messaging and website hero sections.

Create compelling hero section copy for a business with this description:

Business Description:
{description}

Create:
1. A suggested brand name (if not obvious from the description)
2. A powerful headline (8-12 words) that captures the essence
3. A supporting tagline (15-20 words) that elaborates the value proposition
4. 3 bullet points highlighting key benefits

Requirements:
- Use active, engaging language
- Focus on customer benefits, not features
- Be specific to the business type
- Avoid clichés and generic phrases

Return your response as valid JSON matching this exact format:
{{
    "brand_name": "BrandName",
    "headline": "Your Compelling Headline Goes Here",
    "tagline": "A slightly longer tagline that expands on the headline and provides context.",
    "bullets": [
        "First key benefit or value proposition",
        "Second key benefit or value proposition",
        "Third key benefit or value proposition"
    ]
}}

Only return the JSON, no additional text."""


SOCIAL_POSTS_PROMPT_TEMPLATE = """You are a social media marketing expert who creates engaging, platform-optimized content.

Create {num_posts} social media posts for this business:

Business Description:
{description}

Brand Context:
- Brand Name: {brand_name}
- Headline: {headline}
- Tagline: {tagline}

Create posts for these platforms (distribute evenly):
- Twitter/X (max 280 chars)
- Instagram (engaging, visual focus)
- LinkedIn (professional tone)

Requirements:
- Match the tone to each platform
- Include relevant hashtags (3-5 per post)
- Include a clear call-to-action
- Be authentic and engaging, not salesy

Return your response as valid JSON matching this exact format:
{{
    "posts": [
        {{
            "platform": "twitter",
            "content": "Tweet content here",
            "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"],
            "call_to_action": "Learn more at our website!"
        }}
    ]
}}

Generate exactly {num_posts} posts.
Only return the JSON, no additional text."""


LOGO_PROMPT_TEMPLATE = """You are an expert at creating prompts for AI image generation, specifically for logo design.

Create a detailed prompt for generating a professional logo for this business:

Business Description:
{description}

Brand Name:
{brand_name}

Create a prompt that will generate a clean, modern logo suitable for:
- Website header
- Favicon
- Social media profiles
- Business cards

The prompt should specify:
- Style (modern, minimalist, vintage, etc.)
- Shape (circular, square, abstract, etc.)
- Color palette (describe colors that match the brand)
- Any symbolic elements relevant to the business
- Technical requirements (vector-like, scalable, clean edges)

Return ONLY the image generation prompt, nothing else. The prompt should be 50-100 words.

Important: Do NOT include any text or letters in the logo description - the logo should be purely symbolic/iconic."""


# Chain Implementations

class BrandStudioChains:
    """
    Collection of LangChain chains for brand asset generation.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize chains with an LLM manager.
        
        Args:
            llm_manager: LLMClientManager instance with configured models
        """
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_text_llm()
        
        # Initialize output parsers
        self.domain_parser = JsonOutputParser(pydantic_object=DomainSuggestions)
        self.hero_parser = JsonOutputParser(pydantic_object=HeroCopy)
        self.social_parser = JsonOutputParser(pydantic_object=SocialPosts)
        
        # Build chains
        self._build_chains()
    
    def _build_chains(self):
        """Build all LangChain chains."""
        # Domain Generation Chain
        self.domain_prompt = PromptTemplate(
            template=DOMAIN_PROMPT_TEMPLATE,
            input_variables=["description", "num_suggestions"]
        )
        self.domain_chain = self.domain_prompt | self.llm | StrOutputParser()
        
        # Hero Copy Chain
        self.hero_prompt = PromptTemplate(
            template=HERO_COPY_PROMPT_TEMPLATE,
            input_variables=["description"]
        )
        self.hero_chain = self.hero_prompt | self.llm | StrOutputParser()
        
        # Social Posts Chain
        self.social_prompt = PromptTemplate(
            template=SOCIAL_POSTS_PROMPT_TEMPLATE,
            input_variables=["description", "brand_name", "headline", "tagline", "num_posts"]
        )
        self.social_chain = self.social_prompt | self.llm | StrOutputParser()
        
        # Logo Prompt Chain
        self.logo_prompt_template = PromptTemplate(
            template=LOGO_PROMPT_TEMPLATE,
            input_variables=["description", "brand_name"]
        )
        self.logo_prompt_chain = self.logo_prompt_template | self.llm | StrOutputParser()
    
    def _parse_json_safely(self, text: str, fallback: Any = None) -> Any:
        """
        Safely parse JSON from LLM output.
        
        Args:
            text: Raw text output from LLM
            fallback: Value to return if parsing fails
            
        Returns:
            Parsed JSON or fallback value
        """
        # Try to find JSON in the text
        try:
            # First, try direct parsing
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
        
        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return fallback
    
    def generate_domains(
        self, 
        description: str, 
        num_suggestions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate domain name suggestions.
        
        Args:
            description: Business description
            num_suggestions: Number of domains to generate
            
        Returns:
            List of domain suggestion dictionaries
        """
        try:
            result = self.domain_chain.invoke({
                "description": description,
                "num_suggestions": num_suggestions
            })
            
            parsed = self._parse_json_safely(result)
            
            if parsed and "domains" in parsed:
                domains = parsed["domains"]
                # Add simulated availability check
                for domain in domains:
                    domain["available"] = self._check_domain_availability(domain.get("name", ""))
                return domains
            
            # Fallback: generate basic suggestions
            return self._generate_fallback_domains(description, num_suggestions)
            
        except Exception as e:
            print(f"Domain generation error: {e}")
            return self._generate_fallback_domains(description, num_suggestions)
    
    def _check_domain_availability(self, domain_name: str) -> bool:
        """
        Stub for domain availability check.
        In production, this would call GoDaddy's API.
        
        Args:
            domain_name: Domain name to check
            
        Returns:
            Simulated availability (True/False)
        """
        # Simulate availability - in production, call actual API
        import hashlib
        # Use hash to create deterministic but seemingly random availability
        hash_val = int(hashlib.md5(domain_name.encode()).hexdigest(), 16)
        return hash_val % 10 > 2  # ~70% available
    
    def _generate_fallback_domains(
        self, 
        description: str, 
        num_suggestions: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback domain suggestions if main generation fails."""
        # Extract keywords from description
        words = description.lower().split()
        keywords = [w for w in words if len(w) > 3][:3]
        
        suggestions = []
        prefixes = ["get", "try", "my", "the", "go"]
        suffixes = ["hq", "app", "hub", "io", "co"]
        
        for i in range(num_suggestions):
            if keywords:
                base = keywords[i % len(keywords)]
                if i < len(prefixes):
                    name = f"{prefixes[i]}{base}"
                else:
                    name = f"{base}{suffixes[i % len(suffixes)]}"
            else:
                name = f"mybrand{i+1}"
            
            suggestions.append({
                "name": name,
                "tld": ".com",
                "rationale": "Auto-generated suggestion",
                "score": 70 - (i * 5),
                "available": True
            })
        
        return suggestions
    
    def generate_hero_copy(self, description: str) -> Dict[str, Any]:
        """
        Generate hero section copy.
        
        Args:
            description: Business description
            
        Returns:
            Dictionary with headline, tagline, and bullets
        """
        try:
            result = self.hero_chain.invoke({"description": description})
            parsed = self._parse_json_safely(result)
            
            if parsed:
                return {
                    "brand_name": parsed.get("brand_name", "Your Brand"),
                    "headline": parsed.get("headline", "Transform Your Business Today"),
                    "tagline": parsed.get("tagline", "Innovative solutions for modern challenges."),
                    "bullets": parsed.get("bullets", [
                        "High-quality products and services",
                        "Dedicated customer support",
                        "Competitive pricing"
                    ])
                }
            
            return self._generate_fallback_hero(description)
            
        except Exception as e:
            print(f"Hero copy generation error: {e}")
            return self._generate_fallback_hero(description)
    
    def _generate_fallback_hero(self, description: str) -> Dict[str, Any]:
        """Generate fallback hero copy if main generation fails."""
        # Extract key terms for basic personalization
        words = description.split()[:5]
        context = " ".join(words) if words else "your business"
        
        return {
            "brand_name": "Your Brand",
            "headline": f"Elevate Your Experience with {context.title()}",
            "tagline": "Discover the difference that passion and expertise can make for you.",
            "bullets": [
                "Premium quality you can trust",
                "Personalized service tailored to your needs",
                "Innovation meets tradition"
            ]
        }
    
    def generate_social_posts(
        self, 
        description: str, 
        hero_copy: Dict[str, Any],
        num_posts: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate social media posts.
        
        Args:
            description: Business description
            hero_copy: Generated hero copy for context
            num_posts: Number of posts to generate
            
        Returns:
            List of social post dictionaries
        """
        try:
            result = self.social_chain.invoke({
                "description": description,
                "brand_name": hero_copy.get("brand_name", "Brand"),
                "headline": hero_copy.get("headline", ""),
                "tagline": hero_copy.get("tagline", ""),
                "num_posts": num_posts
            })
            
            parsed = self._parse_json_safely(result)
            
            if parsed and "posts" in parsed:
                return parsed["posts"]
            
            return self._generate_fallback_social(hero_copy, num_posts)
            
        except Exception as e:
            print(f"Social posts generation error: {e}")
            return self._generate_fallback_social(hero_copy, num_posts)
    
    def _generate_fallback_social(
        self, 
        hero_copy: Dict[str, Any], 
        num_posts: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback social posts if main generation fails."""
        brand_name = hero_copy.get("brand_name", "Brand")
        headline = hero_copy.get("headline", "Check us out!")
        
        platforms = ["twitter", "instagram", "linkedin"]
        posts = []
        
        templates = [
            {
                "platform": "twitter",
                "content": f"🚀 Exciting news! {brand_name} is here to revolutionize your experience. {headline}",
                "hashtags": ["#NewBusiness", "#Innovation", "#LaunchDay"],
                "call_to_action": "Visit our website to learn more!"
            },
            {
                "platform": "instagram",
                "content": f"✨ Say hello to {brand_name}! We're passionate about delivering exceptional quality and service. Follow our journey as we build something amazing. {headline}",
                "hashtags": ["#NewBrand", "#Entrepreneurship", "#SmallBusiness", "#Launch"],
                "call_to_action": "Link in bio! 👆"
            },
            {
                "platform": "linkedin",
                "content": f"I'm thrilled to announce the launch of {brand_name}. Our mission is to bring innovation and excellence to our industry. {headline} Looking forward to connecting with fellow professionals and potential partners.",
                "hashtags": ["#Startup", "#Entrepreneurship", "#BusinessLaunch"],
                "call_to_action": "Connect with us to learn more about collaboration opportunities."
            }
        ]
        
        return templates[:num_posts]
    
    def generate_logo_prompt(
        self, 
        description: str, 
        brand_name: str
    ) -> str:
        """
        Generate a prompt for logo image generation.
        
        Args:
            description: Business description
            brand_name: Brand name for the logo
            
        Returns:
            Optimized prompt for Stable Diffusion
        """
        try:
            result = self.logo_prompt_chain.invoke({
                "description": description,
                "brand_name": brand_name
            })
            
            # Clean up the prompt
            prompt = result.strip()
            
            # Ensure it's suitable for image generation
            if len(prompt) < 20:
                prompt = self._generate_fallback_logo_prompt(description, brand_name)
            
            # Add quality modifiers
            quality_suffix = ", professional logo design, vector art style, clean lines, high quality, 4k"
            
            return prompt + quality_suffix
            
        except Exception as e:
            print(f"Logo prompt generation error: {e}")
            return self._generate_fallback_logo_prompt(description, brand_name)
    

    def _generate_fallback_logo_prompt(
        self, 
        description: str, 
        brand_name: str
    ) -> str:
        """Generate fallback logo prompt if main generation fails."""
        # Extract business type from description
        words = description.lower().split()
        business_type = " ".join(words[:3]) if len(words) >= 3 else "business"
        
        return (
            f"Modern minimalist logo design for {brand_name}, "
            f"related to {business_type}, clean geometric shapes, "
            f"professional color palette, simple and memorable, "
            f"scalable vector style, no text or letters, "
            f"professional logo design, vector art style, clean lines, high quality, 4k"
        )