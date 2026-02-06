# 🚀 AI Brand Studio

**An AI-powered brand asset generator inspired by GoDaddy Airo™**

Generate a complete brand starter kit from a single business description:
- 🌐 5 Domain name suggestions
- ✨ Hero copy (headline + tagline)
- 📱 3 Social media posts
- 🎨 AI-generated logo
- 📦 Downloadable ZIP of all assets

![AI Brand Studio Demo](demo_assets/screenshot.png)

## 🎯 Features

- **AI Domain Search**: Suggests catchy, memorable domain names
- **AI Logo Generation**: Creates professional logos using Stable Diffusion
- **AI Marketing Content**: Auto-writes social posts, headlines, and taglines
- **One-Click Download**: Package all assets in a ZIP file
- **LangGraph Visualization**: See the AI workflow in action

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain + LangGraph
- **Text Generation**: Hugging Face (Flan-T5 / Mistral)
- **Image Generation**: Stable Diffusion via Diffusers
- **Storage**: Local filesystem with ZIP packaging

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Hugging Face account and API token
- (Optional) NVIDIA GPU for local image generation

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/ai-brand-studio.git
cd ai-brand-studio