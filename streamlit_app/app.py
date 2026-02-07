"""
GoDaddy AI - Main Streamlit Application

Generate complete brand starter kits from a single business description.
"""

import streamlit as st
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.chains import BrandStudioChains
from backend.llm_clients import LLMClientManager
from backend.storage import AssetStorage, ZipPackager
from backend.utils import (
    generate_session_id, 
    sanitize_input, 
    calculate_cost_estimate,
    format_duration
)
from streamlit_app.ui_helpers import (
    display_domain_cards,
    display_hero_section,
    display_social_posts,
    display_logo,
    display_langgraph_visual,
    display_metrics_sidebar,
    create_download_section,
    show_loading_animation
)

# Page Configuration
st.set_page_config(
    page_title="GoDaddy AI Brand Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .domain-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .social-post {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        width: 100%;
    }
    .success-banner {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'history' not in st.session_state:
        st.session_state.history = []


def render_sidebar():
    """Render the sidebar with settings and metrics."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model Selection
        st.markdown("### 🤖 Model Configuration")
        text_model = st.selectbox(
            "Text Model",
            [
                "mistralai/Mistral-7B-Instruct-v0.1",
                "google/flan-t5-large",
                "meta-llama/Llama-2-7b-chat-hf"
            ],
            index=0
        )
        
        image_model = st.selectbox(
            "Image Model",
            [
                "stabilityai/stable-diffusion-xl-base-1.0",
                "runwayml/stable-diffusion-v1-5",
                "stabilityai/stable-diffusion-2-1"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Generation Options
        st.markdown("### 🎛️ Generation Options")
        num_domains = st.slider("Number of Domains", 3, 10, 5)
        num_social_posts = st.slider("Number of Social Posts", 1, 5, 3)
        include_logo = st.checkbox("Generate Logo", value=True)
        
        st.markdown("---")
        
        # Metrics Display
        if st.session_state.metrics:
            display_metrics_sidebar(st.session_state.metrics)
        
        st.markdown("---")
        
        # LangGraph Visualization
        st.markdown("### 🔄 AI Workflow")
        display_langgraph_visual()
        
        st.markdown("---")
        
        # Session Info
        st.markdown("### 📊 Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"Generations: {len(st.session_state.history)}")
        
        return {
            "text_model": text_model,
            "image_model": image_model,
            "num_domains": num_domains,
            "num_social_posts": num_social_posts,
            "include_logo": include_logo
        }


def render_main_content(settings: dict):
    """Render the main content area."""
    # Header
    st.markdown('<h1 class="main-header">🚀 GoDaddy AI Brand Studio</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Generate your complete brand starter kit in seconds</p>', 
        unsafe_allow_html=True
    )
    
    # Input Section
    st.markdown("### 💡 Describe Your Business")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        business_description = st.text_area(
            "Enter a brief description of your business idea",
            placeholder="Example: A cozy artisan bakery in downtown Portland specializing in sourdough bread and French pastries, with a focus on organic ingredients and sustainable practices.",
            height=100,
            max_chars=500,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.button("✨ Generate Brand Kit", use_container_width=True)
    
    # Sample Prompts
    with st.expander("📝 Need inspiration? Try these sample prompts"):
        sample_prompts = [
            "🍕 A premium pizza delivery service using electric bikes in San Francisco",
            "🧘 An online yoga studio offering personalized meditation sessions",
            "🌱 A subscription box service for rare houseplants and succulents",
            "🎮 A mobile game development studio focused on educational games for kids",
            "☕ A specialty coffee roaster sourcing beans directly from small farms"
        ]
        for prompt in sample_prompts:
            if st.button(prompt, key=f"sample_{prompt[:20]}"):
                st.session_state.sample_prompt = prompt[2:].strip()
                st.rerun()
    
    # Handle sample prompt selection
    if hasattr(st.session_state, 'sample_prompt') and st.session_state.sample_prompt:
        business_description = st.session_state.sample_prompt
        st.session_state.sample_prompt = None
    
    # Generation Logic
    if generate_clicked and business_description:
        generate_brand_kit(business_description, settings)
    elif generate_clicked and not business_description:
        st.warning("⚠️ Please enter a business description to continue.")
    
    # Display Results
    if st.session_state.generation_complete and st.session_state.results:
        display_results(st.session_state.results)


def generate_brand_kit(description: str, settings: dict):
    """Generate the complete brand kit."""
    start_time = time.time()
    
    # Sanitize input
    clean_description = sanitize_input(description)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize clients
        status_text.text("🔄 Initializing AI models...")
        progress_bar.progress(10)
        
        llm_manager = LLMClientManager(
            text_model=settings["text_model"],
            image_model=settings["image_model"]
        )
        
        chains = BrandStudioChains(llm_manager)
        storage = AssetStorage(st.session_state.session_id)
        
        results = {}
        
        # Generate Domains
        status_text.text("🌐 Generating domain suggestions...")
        progress_bar.progress(25)
        results['domains'] = chains.generate_domains(
            clean_description, 
            num_suggestions=settings["num_domains"]
        )
        
        # Generate Hero Copy
        status_text.text("✍️ Creating hero copy...")
        progress_bar.progress(45)
        results['hero'] = chains.generate_hero_copy(clean_description)
        
        # Generate Social Posts
        status_text.text("📱 Writing social media posts...")
        progress_bar.progress(60)
        results['social_posts'] = chains.generate_social_posts(
            clean_description,
            results['hero'],
            num_posts=settings["num_social_posts"]
        )
        
        # Generate Logo
        if settings["include_logo"]:
            status_text.text("🎨 Generating logo...")
            progress_bar.progress(75)
            
            # First, generate a logo prompt
            logo_prompt = chains.generate_logo_prompt(
                clean_description,
                results['hero'].get('brand_name', 'Brand')
            )
            
            # Then generate the image
            logo_path = llm_manager.generate_image(
                logo_prompt,
                storage.get_logo_path()
            )
            results['logo_path'] = logo_path
        
        # Package Assets
        status_text.text("📦 Packaging your brand kit...")
        progress_bar.progress(90)
        
        # Save all assets
        storage.save_domains(results['domains'])
        storage.save_hero_copy(results['hero'])
        storage.save_social_posts(results['social_posts'])
        storage.save_brand_guidelines(clean_description, results)
        
        # Create ZIP
        zip_path = ZipPackager.create_zip(
            storage.session_dir,
            st.session_state.session_id
        )
        results['zip_path'] = zip_path
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        
        results['metadata'] = {
            'session_id': st.session_state.session_id,
            'description': clean_description,
            'generation_time': duration,
            'timestamp': datetime.now().isoformat(),
            'settings': settings
        }
        
        # Save to session state
        st.session_state.results = results
        st.session_state.generation_complete = True
        st.session_state.metrics = {
            'generation_time': duration,
            'domains_generated': len(results['domains']),
            'posts_generated': len(results['social_posts']),
            'logo_generated': settings["include_logo"],
            'cost_estimate': calculate_cost_estimate(results)
        }
        
        # Add to history
        st.session_state.history.append({
            'timestamp': datetime.now().isoformat(),
            'description': clean_description[:50] + "...",
            'session_id': st.session_state.session_id
        })
        
        progress_bar.progress(100)
        status_text.text("✅ Brand kit generated successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error generating brand kit: {str(e)}")
        st.exception(e)


def display_results(results: dict):
    """Display the generated brand kit results."""
    st.markdown("---")
    
    # Success Banner
    generation_time = results.get("metadata", {}).get("generation_time", 0)
    st.markdown(
        f'''<div class="success-banner">
            🎉 Your brand kit is ready! Generated in {generation_time:.1f} seconds
        </div>''',
        unsafe_allow_html=True
    )
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Domains", 
        "✨ Hero Copy", 
        "📱 Social Posts", 
        "🎨 Logo",
        "📦 Download"
    ])
    
    with tab1:
        st.markdown("### 🌐 Domain Suggestions")
        display_domain_cards(results.get('domains', []))
    
    with tab2:
        st.markdown("### ✨ Hero Copy")
        display_hero_section(results.get('hero', {}))
    
    with tab3:
        st.markdown("### 📱 Social Media Posts")
        display_social_posts(results.get('social_posts', []))
    
    with tab4:
        st.markdown("### 🎨 Your Logo")
        # Pass any generation error for user feedback
        logo_error = results.get('metadata', {}).get('logo_error')
        if results.get('logo_path'):
            display_logo(results['logo_path'], logo_error)
        else:
            st.info("Logo generation was skipped.")
    
    with tab5:
        st.markdown("### 📦 Download Your Brand Kit")
        create_download_section(results)
    
    # Call to Action
    st.markdown("---")
    st.markdown("### 🚀 Ready to Launch?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🛒 Buy Domain", use_container_width=True, key="buy_domain")
    
    with col2:
        st.button("🌐 Publish Website", use_container_width=True, key="publish_site")
    
    with col3:
        st.button("🔄 Generate New Kit", use_container_width=True, key="new_kit",
                 on_click=reset_session)


def reset_session():
    """Reset the session for a new generation."""
    st.session_state.session_id = generate_session_id()
    st.session_state.generation_complete = False
    st.session_state.results = None
    st.session_state.metrics = {}


def main():
    """Main application entry point."""
    initialize_session_state()
    settings = render_sidebar()
    render_main_content(settings)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Built with ❤️ using LangChain, Hugging Face, and Streamlit<br>
            <strong>GoDaddy AI Brand Studio</strong> - Powered by Generative AI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()