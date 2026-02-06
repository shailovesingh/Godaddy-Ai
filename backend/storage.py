"""
Asset Storage and Packaging for AI Brand Studio
================================================
Handles saving, organizing, and packaging generated brand assets.
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid


class AssetStorage:
    """
    Manages storage of generated brand assets.
    
    Creates a session-based directory structure for organizing
    all generated content before packaging.
    """
    
    def __init__(self, session_id: str, base_path: str = None):
        """
        Initialize asset storage for a session.
        
        Args:
            session_id: Unique session identifier
            base_path: Base directory for storage (defaults to env var or /tmp)
        """
        self.session_id = session_id
        self.base_path = base_path or os.getenv(
            "STORAGE_PATH", 
            "/tmp/ai_brand_studio"
        )
        
        # Create session directory
        self.session_dir = Path(self.base_path) / "sessions" / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.assets_dir = self.session_dir / "assets"
        self.assets_dir.mkdir(exist_ok=True)
        
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
    
    def get_logo_path(self) -> str:
        """Get the path where the logo should be saved."""
        return str(self.images_dir / "logo.png")
    
    def save_domains(self, domains: List[Dict[str, Any]]) -> str:
        """
        Save domain suggestions to JSON file.
        
        Args:
            domains: List of domain suggestion dictionaries
            
        Returns:
            Path to saved file
        """
        file_path = self.assets_dir / "domains.json"
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "count": len(domains),
            "domains": domains
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def save_hero_copy(self, hero: Dict[str, Any]) -> str:
        """
        Save hero copy to JSON file.
        
        Args:
            hero: Hero copy dictionary
            
        Returns:
            Path to saved file
        """
        file_path = self.assets_dir / "hero_copy.json"
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            **hero
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def save_social_posts(self, posts: List[Dict[str, Any]]) -> str:
        """
        Save social media posts to JSON file.
        
        Args:
            posts: List of social post dictionaries
            
        Returns:
            Path to saved file
        """
        file_path = self.assets_dir / "social_posts.json"
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "count": len(posts),
            "posts": posts
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def save_brand_guidelines(
        self, 
        description: str, 
        results: Dict[str, Any]
    ) -> str:
        """
        Generate and save brand guidelines markdown file.
        
        Args:
            description: Original business description
            results: All generated results
            
        Returns:
            Path to saved file
        """
        file_path = self.session_dir / "brand_guidelines.md"
        
        hero = results.get('hero', {})
        domains = results.get('domains', [])
        posts = results.get('social_posts', [])
        
        # Generate markdown content
        content = f"""# Brand Guidelines
## {hero.get('brand_name', 'Your Brand')}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {self.session_id}

---

## 📋 Business Overview

{description}

---

## 🎯 Brand Messaging

### Headline
> {hero.get('headline', 'Your Headline Here')}

### Tagline
> {hero.get('tagline', 'Your Tagline Here')}

### Key Value Propositions
"""
        
        for bullet in hero.get('bullets', []):
            content += f"- {bullet}\n"
        
        content += """
---

## 🌐 Domain Recommendations

| Domain | Score | Status | Rationale |
|--------|-------|--------|-----------|
"""
        
        for domain in domains[:5]:
            status = "✅ Available" if domain.get('available', True) else "❌ Taken"
            content += f"| {domain.get('name', '')}.com | {domain.get('score', 'N/A')} | {status} | {domain.get('rationale', '')} |\n"
        
        content += """
---

## 📱 Social Media Content

"""
        
        for i, post in enumerate(posts, 1):
            platform = post.get('platform', 'social').title()
            content += f"""### {platform} Post {i}

**Content:**
{post.get('content', '')}

**Hashtags:** {post.get('hashtags', '')}


**CTA:** {post.get('call_to_action', '')}

"""
        
        content += """---

## 🎨 Visual Identity

### Logo
Your logo has been generated and is included in this brand kit as `logo.png`.

### Color Palette Suggestions
Based on your brand, consider these color directions:
- Primary: Professional, trustworthy tones
- Secondary: Accent colors that convey energy
- Neutral: Clean whites and grays for backgrounds

### Typography Recommendations
- Headlines: Bold, modern sans-serif (e.g., Montserrat, Inter)
- Body: Clean, readable sans-serif (e.g., Open Sans, Roboto)
- Accent: Consider a distinctive font for special elements

---

## 📦 Included Assets

- `logo.png` - Primary logo (512x512)
- `domains.json` - Domain suggestions with availability
- `hero_copy.json` - Headlines and taglines
- `social_posts.json` - Ready-to-post social content
- `brand_guidelines.md` - This document
- `metadata.json` - Generation details

---

## 🚀 Next Steps

1. **Secure Your Domain**: Register your preferred domain name
2. **Refine Your Logo**: Use the generated logo as a starting point
3. **Launch Your Website**: Build your online presence
4. **Start Marketing**: Post your social content and engage your audience

---

*Generated by AI Brand Studio - Powered by Generative AI*
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Save session metadata.
        
        Args:
            metadata: Session metadata dictionary
            
        Returns:
            Path to saved file
        """
        file_path = self.session_dir / "metadata.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return str(file_path)
    
    def copy_logo(self, source_path: str) -> Optional[str]:
        """
        Copy logo from source to session images directory.
        
        Args:
            source_path: Path to source logo file
            
        Returns:
            Path to copied logo or None if failed
        """
        if not source_path or not Path(source_path).exists():
            return None
        
        dest_path = self.images_dir / "logo.png"
        
        try:
            shutil.copy2(source_path, dest_path)
            return str(dest_path)
        except Exception as e:
            print(f"Error copying logo: {e}")
            return None
    
    def list_assets(self) -> Dict[str, List[str]]:
        """
        List all assets in the session directory.
        
        Returns:
            Dictionary with categorized asset paths
        """
        assets = {
            "json_files": [],
            "images": [],
            "documents": [],
            "other": []
        }
        
        for file_path in self.session_dir.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                rel_path = str(file_path.relative_to(self.session_dir))
                
                if suffix == ".json":
                    assets["json_files"].append(rel_path)
                elif suffix in [".png", ".jpg", ".jpeg", ".svg"]:
                    assets["images"].append(rel_path)
                elif suffix in [".md", ".txt", ".pdf"]:
                    assets["documents"].append(rel_path)
                else:
                    assets["other"].append(rel_path)
        
        return assets
    
    def cleanup(self):
        """Remove the session directory and all its contents."""
        try:
            shutil.rmtree(self.session_dir)
        except Exception as e:
            print(f"Error cleaning up session {self.session_id}: {e}")


class ZipPackager:
    """
    Handles creation of ZIP packages from session assets.
    """
    
    @staticmethod
    def create_zip(
        session_dir: Path, 
        session_id: str,
        output_dir: Path = None
    ) -> str:
        """
        Create a ZIP file containing all session assets.
        
        Args:
            session_dir: Path to the session directory
            session_id: Session identifier for naming
            output_dir: Output directory for ZIP (defaults to session dir parent)
            
        Returns:
            Path to the created ZIP file
        """
        session_dir = Path(session_dir)
        
        if output_dir is None:
            output_dir = session_dir.parent
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        zip_filename = f"brand_kit_{session_id}.zip"
        zip_path = output_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in session_dir.rglob("*"):
                if file_path.is_file():
                    # Calculate the archive name (relative path)
                    arcname = file_path.relative_to(session_dir)
                    zipf.write(file_path, arcname)
        
        return str(zip_path)
    
    @staticmethod
    def extract_zip(
        zip_path: str, 
        extract_dir: str
    ) -> str:
        """
        Extract a ZIP file to a directory.
        
        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract to
            
        Returns:
            Path to the extraction directory
        """
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        return str(extract_dir)
    
    @staticmethod
    def list_zip_contents(zip_path: str) -> List[Dict[str, Any]]:
        """
        List contents of a ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            List of file info dictionaries
        """
        contents = []
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            for info in zipf.infolist():
                contents.append({
                    "filename": info.filename,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                    "is_dir": info.is_dir()
                })
        
        return contents


class SessionManager:
    """
    Manages multiple sessions and handles cleanup.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the session manager.
        
        Args:
            base_path: Base directory for all sessions
        """
        self.base_path = Path(base_path or os.getenv(
            "STORAGE_PATH",
            "/tmp/ai_brand_studio"
        ))
        self.sessions_dir = self.base_path / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> AssetStorage:
        """
        Create a new session.
        
        Returns:
            AssetStorage instance for the new session
        """
        session_id = str(uuid.uuid4())[:8]
        return AssetStorage(session_id, str(self.base_path))
    
    def get_session(self, session_id: str) -> Optional[AssetStorage]:
        """
        Get an existing session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            AssetStorage instance or None if not found
        """
        session_dir = self.sessions_dir / session_id
        
        if session_dir.exists():
            return AssetStorage(session_id, str(self.base_path))
        
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            List of session info dictionaries
        """
        sessions = []
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                
                session_info = {
                    "session_id": session_dir.name,
                    "created_at": datetime.fromtimestamp(
                        session_dir.stat().st_ctime
                    ).isoformat(),
                    "size_bytes": sum(
                        f.stat().st_size 
                        for f in session_dir.rglob("*") 
                        if f.is_file()
                    )
                }
                
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        session_info["description"] = metadata.get(
                            "description", ""
                        )[:100]
                
                sessions.append(session_info)
        
        return sessions
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than the specified age.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        max_age_hours = max_age_hours or int(os.getenv("MAX_SESSION_AGE_HOURS", "24"))
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                if session_dir.stat().st_ctime < cutoff_time:
                    try:
                        shutil.rmtree(session_dir)
                        print(f"Cleaned up old session: {session_dir.name}")
                    except Exception as e:
                        print(f"Error cleaning up {session_dir.name}: {e}")
    
    def get_total_storage_usage(self) -> Dict[str, Any]:
        """
        Get total storage usage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        total_files = 0
        total_sessions = 0
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                total_sessions += 1
                for file_path in session_dir.rglob("*"):
                    if file_path.is_file():
                        total_files += 1
                        total_size += file_path.stat().st_size
        
        return {
            "total_sessions": total_sessions,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "base_path": str(self.base_path)
        }