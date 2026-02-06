"""
UI Helper Functions for AI Brand Studio
========================================
Reusable UI components for the Streamlit interface.
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import base64


def display_domain_cards(domains: List[Dict[str, Any]]):
    """Display domain suggestions as cards."""
    if not domains:
        st.warning("No domains generated.")
        return
    
    for i, domain in enumerate(domains, 1):
        availability_icon = "✅" if domain.get('available', True) else "❌"
        score = domain.get('score', 0)
        score_stars = "⭐" * min(int(score / 20), 5)
        
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(
                    f"""
                    <div class="domain-card">
                        <h4 style="margin:0; color:#333;">
                            {i}. {domain['name']}
                        </h4>
                        <p style="margin:0.25rem 0 0 0; color:#666; font-size:0.9rem;">
                            {domain.get('rationale', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(f"**Score:** {score_stars}")
                st.caption(f"{score}/100")
            
            with col3:
                st.markdown(f"**Status:** {availability_icon}")
                if domain.get('available', True):
                    st.button(
                        "Check", 
                        key=f"check_domain_{i}",
                        help=f"Check availability for {domain['name']}"
                    )


def display_hero_section(hero: Dict[str, Any]):
    """Display the hero copy section."""
    if not hero:
        st.warning("No hero copy generated.")
        return
    
    # Brand Name
    if hero.get('brand_name'):
        st.markdown(f"### 🏷️ Brand Name: **{hero['brand_name']}**")
    
    # Main Headline
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            color: white;
        ">
            <h1 style="font-size: 2.5rem; margin: 0;">{hero.get('headline', '')}</h1>
            <p style="font-size: 1.3rem; margin-top: 1rem; opacity: 0.9;">
                {hero.get('tagline', '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Value Propositions / Bullets
    if hero.get('bullets'):
        st.markdown("### 💎 Key Value Propositions")
        for bullet in hero['bullets']:
            st.markdown(f"- {bullet}")
    
    # Copy to clipboard buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        headline_text = hero.get('headline', '')
        st.text_input(
            "📋 Copy Headline",
            value=headline_text,
            key="copy_headline",
            help="Click to select, then copy"
        )
    
    with col2:
        tagline_text = hero.get('tagline', '')
        st.text_input(
            "📋 Copy Tagline",
            value=tagline_text,
            key="copy_tagline",
            help="Click to select, then copy"
        )


def display_social_posts(posts: List[Dict[str, Any]]):
    """Display social media posts."""
    if not posts:
        st.warning("No social posts generated.")
        return
    
    platform_icons = {
        'twitter': '🐦',
        'x': '𝕏',
        'instagram': '📸',
        'facebook': '👥',
        'linkedin': '💼',
        'tiktok': '🎵'
    }
    
    for i, post in enumerate(posts, 1):
        platform = post.get('platform', 'twitter').lower()
        icon = platform_icons.get(platform, '📱')
        
        st.markdown(
            f"""
            <div class="social-post">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <strong>{platform.title()}</strong>
                    <span style="margin-left: auto; color: #666; font-size: 0.8rem;">
                        {len(post.get('content', ''))} chars
                    </span>
                </div>
                <p style="margin: 0; line-height: 1.5;">{post.get('content', '')}</p>
                {f'<p style="color: #1da1f2; margin-top: 0.5rem;">{" ".join(post.get("hashtags", []))}</p>' if post.get('hashtags') else ''}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Copy button
        st.text_area(
            f"Copy Post {i}",
            value=post.get('content', '') + (' ' + ' '.join(post.get('hashtags', [])) if post.get('hashtags') else ''),
            height=80,
            key=f"copy_post_{i}",
            label_visibility="collapsed"
        )


def display_logo(logo_path: str):
    """Display the generated logo."""
    if not logo_path or not Path(logo_path).exists():
        st.warning("Logo file not found. Displaying placeholder.")
        # Display a placeholder
        st.markdown(
            """
            <div style="
                width: 200px;
                height: 200px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 2rem auto;
            ">
                <span style="font-size: 4rem; color: white;">🎨</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Display the actual logo
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image(
            logo_path,
            caption="Your AI-Generated Logo",
            width=300
        )
    
    # Logo variations info
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 1rem;">
            <p>💡 <strong>Tip:</strong> This is your primary logo. 
            For best results, consider creating variations for:</p>
            <ul style="list-style: none; padding: 0;">
                <li>🔲 Favicon (32x32px)</li>
                <li>📱 Mobile App Icon (512x512px)</li>
                <li>🌙 Light/Dark mode versions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_langgraph_visual():
    """Display the LangGraph workflow visualization."""
    # ASCII representation of the workflow
    workflow_visual = """
    ```
    ┌─────────────────┐
    │  User Input     │
    │  (Description)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Orchestrator   │
    │   (LangChain)   │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Domain │ │ Hero  │ │Social │
│ Chain │ │ Chain │ │ Chain │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    ▼         ▼         ▼
┌───────────────────────────┐
│    Logo Prompt Chain      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Stable Diffusion (Image) │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   Asset Packager (ZIP)    │
└───────────────────────────┘
    ```
    """
    
    st.markdown(workflow_visual)
    
    with st.expander("📊 View Detailed Flow"):
        st.markdown("""
        **Chain Details:**
        
        1. **Domain Chain** → Generates 5 domain suggestions
           - Input: Business description
           - Output: JSON with names + rationales
           
        2. **Hero Chain** → Creates headline + tagline
           - Input: Business description
           - Output: Headline, tagline, 3 bullets
           
        3. **Social Chain** → Writes platform-specific posts
           - Input: Description + hero copy
           - Output: 3 posts with hashtags
           
        4. **Logo Prompt Chain** → Creates SD prompt
           - Input: Brand name + description
           - Output: Optimized image prompt
           
        5. **Image Generation** → Stable Diffusion XL
           - Input: Logo prompt
           - Output: PNG image file
        """)


def display_metrics_sidebar(metrics: Dict[str, Any]):
    """Display generation metrics in the sidebar."""
    st.markdown("### 📈 Generation Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "⏱️ Time",
            f"{metrics.get('generation_time', 0):.1f}s"
        )
    
    with col2:
        st.metric(
            "🌐 Domains",
            metrics.get('domains_generated', 0)
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric(
            "📱 Posts",
            metrics.get('posts_generated', 0)
        )
    
    with col4:
        st.metric(
            "💰 Est. Cost",
            f"${metrics.get('cost_estimate', 0):.4f}"
        )
    
    # Logo status
    logo_status = "✅ Generated" if metrics.get('logo_generated') else "⏭️ Skipped"
    st.markdown(f"**🎨 Logo:** {logo_status}")


def create_download_section(results: Dict[str, Any]):
    """Create the download section with ZIP and individual files."""
    st.markdown("""
    Your complete brand kit includes:
    - 📄 `domains.json` - Domain suggestions with rationales
    - 📄 `hero_copy.json` - Headline, tagline, and value props
    - 📄 `social_posts.json` - Ready-to-post social content
    - 🎨 `logo.png` - AI-generated logo
    - 📋 `brand_guidelines.md` - Brand style guide
    - 📊 `metadata.json` - Generation details
    """)
    
    # Main ZIP download
    zip_path = results.get('zip_path')
    
    if zip_path and Path(zip_path).exists():
        with open(zip_path, 'rb') as f:
            zip_data = f.read()
        
        st.download_button(
            label="📦 Download Complete Brand Kit (ZIP)",
            data=zip_data,
            file_name=f"brand_kit_{results['metadata']['session_id'][:8]}.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.warning("ZIP file not available for download.")
    
    # Individual file downloads
    st.markdown("---")
    st.markdown("#### 📁 Individual Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Domains JSON
        st.download_button(
            label="🌐 Download Domains (JSON)",
            data=json.dumps(results.get('domains', []), indent=2),
            file_name="domains.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Social Posts JSON
        st.download_button(
            label="📱 Download Social Posts (JSON)",
            data=json.dumps(results.get('social_posts', []), indent=2),
            file_name="social_posts.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Hero Copy JSON
        st.download_button(
            label="✨ Download Hero Copy (JSON)",
            data=json.dumps(results.get('hero', {}), indent=2),
            file_name="hero_copy.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Logo PNG
        logo_path = results.get('logo_path')
        if logo_path and Path(logo_path).exists():
            with open(logo_path, 'rb') as f:
                logo_data = f.read()
            
            st.download_button(
                label="🎨 Download Logo (PNG)",
                data=logo_data,
                file_name="logo.png",
                mime="image/png",
                use_container_width=True
            )


def show_loading_animation(message: str = "Generating..."):
    """Display a loading animation with a message."""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="loader"></div>
            <p style="margin-top: 1rem; color: #666;">{message}</p>
        </div>
        <style>
            .loader {{
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def display_error_message(error: str, suggestion: str = None):
    """Display a formatted error message."""
    st.markdown(
        f"""
        <div style="
            background: #fee;
            border: 1px solid #fcc;
            border-left: 4px solid #f44;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        ">
            <strong>❌ Error:</strong> {error}
            {f'<br><br><em>💡 Suggestion: {suggestion}</em>' if suggestion else ''}
        </div>
        """,
        unsafe_allow_html=True
    )