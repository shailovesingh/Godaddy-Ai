"""
LangChain Agents for AI Brand Studio
====================================
Agent-based orchestration for complex brand generation workflows.
"""

import os
from typing import Dict, List, Any, Optional, Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json


# ============================================
# Tool Definitions
# ============================================

class DomainCheckInput(BaseModel):
    """Input schema for domain availability check."""
    domain_name: str = Field(description="Domain name to check (without TLD)")
    tld: str = Field(default=".com", description="Top-level domain")


class DomainCheckOutput(BaseModel):
    """Output schema for domain availability check."""
    domain: str
    available: bool
    price: Optional[float] = None
    alternatives: List[str] = []


def check_domain_availability(domain_name: str, tld: str = ".com") -> Dict[str, Any]:
    """
    Stub tool for checking domain availability.
    In production, this would call GoDaddy's Domain Availability API.
    
    Args:
        domain_name: Domain name without TLD
        tld: Top-level domain (default: .com)
        
    Returns:
        Dictionary with availability info
    """
    import hashlib
    
    full_domain = f"{domain_name}{tld}"
    
    # Simulate API response with deterministic randomness
    hash_val = int(hashlib.md5(full_domain.encode()).hexdigest(), 16)
    is_available = hash_val % 10 > 3  # ~60% available
    
    # Simulate pricing
    base_prices = {
        ".com": 12.99,
        ".io": 39.99,
        ".co": 24.99,
        ".net": 14.99,
        ".org": 12.99,
        ".ai": 79.99
    }
    
    price = base_prices.get(tld, 15.99)
    if not is_available:
        price = price * 10  # Premium pricing for taken domains
    
    # Generate alternatives if not available
    alternatives = []
    if not is_available:
        suffixes = ["app", "hq", "hub", "now", "go"]
        alt_tlds = [".io", ".co", ".net"]
        alternatives = [
            f"{domain_name}{suffix}.com" for suffix in suffixes[:2]
        ] + [
            f"{domain_name}{alt_tld}" for alt_tld in alt_tlds[:2]
        ]
    
    return {
        "domain": full_domain,
        "available": is_available,
        "price": price if is_available else None,
        "alternatives": alternatives,
        "registrar": "GoDaddy",
        "check_timestamp": "2024-01-01T00:00:00Z"  # Stub timestamp
    }


def update_dns_records(domain: str, records: List[Dict]) -> Dict[str, Any]:
    """
    Stub tool for updating DNS records.
    In production, this would call GoDaddy's DNS API.
    
    Args:
        domain: Domain name
        records: List of DNS records to update
        
    Returns:
        Status of the update operation
    """
    # Simulate DNS update
    return {
        "domain": domain,
        "status": "success",
        "message": "DNS records updated successfully (simulated)",
        "records_updated": len(records),
        "propagation_time": "24-48 hours"
    }


def publish_website(domain: str, template_id: str, content: Dict) -> Dict[str, Any]:
    """
    Stub tool for publishing a website.
    In production, this would call GoDaddy's Website Builder API.
    
    Args:
        domain: Domain to publish to
        template_id: Template identifier
        content: Website content
        
    Returns:
        Publication status
    """
    return {
        "domain": domain,
        "status": "published",
        "url": f"https://{domain}",
        "template_id": template_id,
        "ssl_enabled": True,
        "message": "Website published successfully (simulated)"
    }


# ============================================
# LangChain Tools
# ============================================

domain_check_tool = StructuredTool.from_function(
    func=check_domain_availability,
    name="check_domain",
    description="Check if a domain name is available for registration. Input: domain_name (without .com), tld (optional, default .com)",
    args_schema=DomainCheckInput
)

dns_update_tool = Tool(
    name="update_dns",
    func=lambda x: update_dns_records(x.get("domain"), x.get("records", [])),
    description="Update DNS records for a domain. Input: JSON with 'domain' and 'records' array"
)

publish_tool = Tool(
    name="publish_website",
    func=lambda x: publish_website(x.get("domain"), x.get("template_id"), x.get("content", {})),
    description="Publish a website to a domain. Input: JSON with 'domain', 'template_id', and 'content'"
)


# ============================================
# LangGraph State & Workflow
# ============================================

class BrandGenerationState(TypedDict):
    """State schema for the brand generation workflow."""
    # Input
    business_description: str
    user_preferences: Optional[Dict[str, Any]]
    
    # Generated assets
    domains: Optional[List[Dict[str, Any]]]
    hero_copy: Optional[Dict[str, Any]]
    social_posts: Optional[List[Dict[str, Any]]]
    logo_prompt: Optional[str]
    logo_path: Optional[str]
    
    # Metadata
    current_step: str
    errors: List[str]
    messages: Sequence[BaseMessage]
    
    # Output
    zip_path: Optional[str]
    completed: bool


class BrandStudioAgent:
    """
    LangGraph-based agent for orchestrating brand generation workflow.
    
    This agent manages the entire brand generation pipeline using
    a state machine approach with LangGraph.
    """
    
    def __init__(self, chains, llm_manager):
        """
        Initialize the brand studio agent.
        
        Args:
            chains: BrandStudioChains instance
            llm_manager: LLMClientManager instance
        """
        self.chains = chains
        self.llm_manager = llm_manager
        self.tools = [domain_check_tool, dns_update_tool, publish_tool]
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for brand generation.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(BrandGenerationState)
        
        # Add nodes
        workflow.add_node("generate_domains", self._generate_domains_node)
        workflow.add_node("generate_hero", self._generate_hero_node)
        workflow.add_node("generate_social", self._generate_social_node)
        workflow.add_node("generate_logo", self._generate_logo_node)
        workflow.add_node("package_assets", self._package_assets_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define edges
        workflow.set_entry_point("generate_domains")
        
        workflow.add_edge("generate_domains", "generate_hero")
        workflow.add_edge("generate_hero", "generate_social")
        workflow.add_edge("generate_social", "generate_logo")
        workflow.add_edge("generate_logo", "package_assets")
        workflow.add_edge("package_assets", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "generate_domains",
            self._should_continue,
            {
                "continue": "generate_hero",
                "error": "handle_error"
            }
        )
        
        return workflow.compile()
    
    def _should_continue(self, state: BrandGenerationState) -> str:
        """Determine if workflow should continue or handle error."""
        if state.get("errors") and len(state["errors"]) > 3:
            return "error"
        return "continue"
    
    def _generate_domains_node(self, state: BrandGenerationState) -> Dict:
        """Node: Generate domain suggestions."""
        try:
            domains = self.chains.generate_domains(
                state["business_description"],
                num_suggestions=5
            )
            
            # Check availability for each domain
            for domain in domains:
                check_result = check_domain_availability(domain["name"])
                domain["available"] = check_result["available"]
                domain["price"] = check_result.get("price")
            
            return {
                "domains": domains,
                "current_step": "domains_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Generated {len(domains)} domain suggestions")
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Domain generation failed: {str(e)}"],
                "current_step": "domains_failed"
            }
    
    def _generate_hero_node(self, state: BrandGenerationState) -> Dict:
        """Node: Generate hero copy."""
        try:
            hero_copy = self.chains.generate_hero_copy(state["business_description"])
            
            return {
                "hero_copy": hero_copy,
                "current_step": "hero_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Generated hero copy: {hero_copy.get('headline', '')[:50]}...")
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Hero copy generation failed: {str(e)}"],
                "current_step": "hero_failed"
            }
    
    def _generate_social_node(self, state: BrandGenerationState) -> Dict:
        """Node: Generate social media posts."""
        try:
            social_posts = self.chains.generate_social_posts(
                state["business_description"],
                state.get("hero_copy", {}),
                num_posts=3
            )
            
            return {
                "social_posts": social_posts,
                "current_step": "social_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Generated {len(social_posts)} social media posts")
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Social posts generation failed: {str(e)}"],
                "current_step": "social_failed"
            }
    
    def _generate_logo_node(self, state: BrandGenerationState) -> Dict:
        """Node: Generate logo."""
        try:
            # Generate logo prompt
            brand_name = state.get("hero_copy", {}).get("brand_name", "Brand")
            logo_prompt = self.chains.generate_logo_prompt(
                state["business_description"],
                brand_name
            )
            
            # Generate logo image (if image generation is available)
            logo_path = None
            try:
                logo_path = self.llm_manager.generate_image(
                    logo_prompt,
                    f"/tmp/ai_brand_studio/logo_{brand_name.lower().replace(' ', '_')}.png"
                )
            except Exception as img_error:
                print(f"Image generation failed: {img_error}")
            
            return {
                "logo_prompt": logo_prompt,
                "logo_path": logo_path,
                "current_step": "logo_complete",
                "messages": state.get("messages", []) + [
                    AIMessage(content="Generated logo" if logo_path else "Logo prompt generated (image pending)")
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Logo generation failed: {str(e)}"],
                "current_step": "logo_failed"
            }
    
    def _package_assets_node(self, state: BrandGenerationState) -> Dict:
        """Node: Package all assets into a ZIP file."""
        try:
            from backend.storage import AssetStorage, ZipPackager
            import uuid
            
            session_id = str(uuid.uuid4())[:8]
            storage = AssetStorage(session_id)
            
            # Save all assets
            if state.get("domains"):
                storage.save_domains(state["domains"])
            if state.get("hero_copy"):
                storage.save_hero_copy(state["hero_copy"])
            if state.get("social_posts"):
                storage.save_social_posts(state["social_posts"])
            
            # Create brand guidelines
            storage.save_brand_guidelines(
                state["business_description"],
                {
                    "domains": state.get("domains"),
                    "hero": state.get("hero_copy"),
                    "social_posts": state.get("social_posts")
                }
            )
            
            # Create ZIP
            zip_path = ZipPackager.create_zip(storage.session_dir, session_id)
            
            return {
                "zip_path": zip_path,
                "current_step": "complete",
                "completed": True,
                "messages": state.get("messages", []) + [
                    AIMessage(content=f"Brand kit packaged: {zip_path}")
                ]
            }
        except Exception as e:
            return {
                "errors": state.get("errors", []) + [f"Packaging failed: {str(e)}"],
                "current_step": "packaging_failed",
                "completed": False
            }
    
    def _handle_error_node(self, state: BrandGenerationState) -> Dict:
        """Node: Handle errors in the workflow."""
        error_summary = "\n".join(state.get("errors", ["Unknown error"]))
        
        return {
            "current_step": "error_handled",
            "completed": False,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Workflow encountered errors:\n{error_summary}")
            ]
        }
    
    def run(
        self, 
        business_description: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> BrandGenerationState:
        """
        Run the complete brand generation workflow.
        
        Args:
            business_description: Description of the business
            user_preferences: Optional user preferences
            
        Returns:
            Final workflow state with all generated assets
        """
        initial_state: BrandGenerationState = {
            "business_description": business_description,
            "user_preferences": user_preferences or {},
            "domains": None,
            "hero_copy": None,
            "social_posts": None,
            "logo_prompt": None,
            "logo_path": None,
            "current_step": "initialized",
            "errors": [],
            "messages": [HumanMessage(content=business_description)],
            "zip_path": None,
            "completed": False
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def get_workflow_visualization(self) -> str:
        """
        Get a text-based visualization of the workflow.
        
        Returns:
            ASCII art representation of the workflow
        """
        return """
        ╔═══════════════════════════════════════════════════════════╗
        ║              AI Brand Studio Workflow                      ║
        ╠═══════════════════════════════════════════════════════════╣
        ║                                                           ║
        ║    ┌─────────────────┐                                    ║
        ║    │   User Input    │                                    ║
        ║    │  (Description)  │                                    ║
        ║    └────────┬────────┘                                    ║
        ║             │                                             ║
        ║             ▼                                             ║
        ║    ┌─────────────────┐                                    ║
        ║    │ Generate Domains│──────┐                             ║
        ║    │   (5 names)     │      │ error                       ║
        ║    └────────┬────────┘      │                             ║
        ║             │               │                             ║
        ║             ▼               │                             ║
        ║    ┌─────────────────┐      │                             ║
        ║    │  Generate Hero  │      │                             ║
        ║    │ (headline/tag)  │      │                             ║
        ║    └────────┬────────┘      │                             ║
        ║             │               │                             ║
        ║             ▼               │                             ║
        ║    ┌─────────────────┐      │                             ║
        ║    │ Generate Social │      ▼                             ║
        ║    │   (3 posts)     │  ┌───────────┐                     ║
        ║    └────────┬────────┘  │  Handle   │                     ║
        ║             │           │   Error   │                     ║
        ║             ▼           └─────┬─────┘                     ║
        ║    ┌─────────────────┐        │                           ║
        ║    │  Generate Logo  │        │                           ║
        ║    │  (SD prompt)    │        │                           ║
        ║    └────────┬────────┘        │                           ║
        ║             │                 │                           ║
        ║             ▼                 │                           ║
        ║    ┌─────────────────┐        │                           ║
        ║    │ Package Assets  │        │                           ║
        ║    │   (ZIP file)    │        │                           ║
        ║    └────────┬────────┘        │                           ║
        ║             │                 │                           ║
        ║             ▼                 ▼                           ║
        ║    ┌─────────────────────────────┐                        ║
        ║    │           END               │                        ║
        ║    │   (Return final state)      │                        ║
        ║    └─────────────────────────────┘                        ║
        ║                                                           ║
        ╚═══════════════════════════════════════════════════════════╝
        """


# ============================================
# MCP-Style Tool Interface (Stub)
# ============================================

class MCPToolInterface:
    """
    Model Context Protocol (MCP) style tool interface.
    
    This provides a standardized way to expose tools that could be
    called by external systems or AI models.
    """
    
    def __init__(self):
        self.tools = {
            "check_domain": {
                "name": "check_domain",
                "description": "Check domain availability",
                "input_schema": DomainCheckInput.schema(),
                "handler": check_domain_availability
            },
            "update_dns": {
                "name": "update_dns",
                "description": "Update DNS records",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "records": {"type": "array"}
                    }
                },
                "handler": update_dns_records
            },
            "publish_website": {
                "name": "publish_website",
                "description": "Publish website to domain",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "template_id": {"type": "string"},
                        "content": {"type": "object"}
                    }
                },
                "handler": publish_website
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
            for tool in self.tools.values()
        ]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool by name with arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        
        try:
            handler = self.tools[name]["handler"]
            result = handler(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}