#!/usr/bin/env python3
"""
Fix missing imports by adding proper import paths.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


class MissingImportFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.fixes_applied = []
        
        # Map of common import fixes
        self.import_fixes = {
            # Core modules
            "principle_engine": "from principle_engine import",
            "communication_style": "from communication_style import",
            "relationship_tracker": "from relationship_tracker import",
            "conflict_resolver": "from conflict_resolver import",
            "a2a_task_handler": "from a2a_task_handler import",
            "agent_card": "from agent_card import",
            "session_manager": "from session_manager import",
            "collaborative_task_handler": "from collaborative_task_handler import",
            "communication_adapter": "from communication_adapter import",
            "content_handler": "from content_handler import",
            "file_exchange_handler": "from file_exchange_handler import",
            "orchestrator_engine": "from orchestrator_engine import",
            "agent_registry": "from agent_registry import",
            "learning_system": "from learning_system import",
            "emotional_intelligence": "from emotional_intelligence import",
            "communication_style_analyzer": "from communication_style_analyzer import",
            "api_gateway_system": "from api_gateway_system import",
            "communication_channel_manager": "from communication_channel_manager import",
            "email_channel_adapter": "from email_channel_adapter import",
            "api_channel_adapter": "from api_channel_adapter import",
            "chat_channel_adapter": "from chat_channel_adapter import",
            "human_interaction_styler": "from human_interaction_styler import",
            "cross_modal_context_manager": "from cross_modal_context_manager import",
            "project_orchestrator": "from project_orchestrator import",
            "result_synthesizer": "from result_synthesizer import",
            "continuous_evolution_system": "from continuous_evolution_system import",
            "feedback_integration_system": "from feedback_integration_system import",
            "crisis_response_coordinator": "from crisis_response_coordinator import",
            "learning_journey_orchestrator": "from learning_journey_orchestrator import",
            "security_privacy_manager": "from security_privacy_manager import",
            "universal_agent_connector": "from universal_agent_connector import",
            "orchestration_analytics": "from orchestration_analytics import",
            "multilingual_engine": "from multilingual_engine import",
            "media_content_processor": "from media_content_processor import",
            "test_framework": "from test_framework import",
            "test_scenarios": "from test_scenarios import",
            
            # Emoji modules
            "emoji_knowledge_base": "from emoji_knowledge_base import",
            "emoji_translation_engine": "from emoji_translation_engine import",
            "emoji_grammar_system": "from emoji_grammar_system import",
            "emoji_dialogue_manager": "from emoji_dialogue_manager import",
            "emoji_sequence_optimizer": "from emoji_sequence_optimizer import",
            "emoji_communication_endpoint": "from emoji_communication_endpoint import",
            "emoji_emotional_analyzer": "from emoji_emotional_analyzer import",
            "emoji_tutorial_system": "from emoji_tutorial_system import",
            "domain_specific_emoji_sets": "from domain_specific_emoji_sets import",
            
            # Fairness and ethics modules
            "fairness_evaluator": "from fairness_evaluator import",
            "fairness_evaluation": "from fairness_evaluation import",
            "fairness_evaluation_integrator": "from fairness_evaluation_integrator import",
            "fairness_evaluator_implementations": "from fairness_evaluator_implementations import",
            "authenticity_verifier": "from authenticity_verifier import",
            "trust_evaluator": "from trust_evaluator import",
            "engagement_strategist": "from engagement_strategist import",
            "self_reflection": "from self_reflection import",
            "conflict_engagement": "from conflict_engagement import",
            "conflict_engagement_functions": "from conflict_engagement_functions import",
            "collaborative_growth": "from collaborative_growth import",
            "strategic_adaptation": "from strategic_adaptation import",
            "ethical_dilemma_resolver": "from ethical_dilemma_resolver import",
            
            # Principle engine extensions
            "principle_engine_example": "from principle_engine_example import",
            "principle_engine_positive_reinforcement": "from principle_engine_positive_reinforcement import",
            "principle_engine_integration": "from principle_engine_integration import",
            "principle_engine_fairness": "from principle_engine_fairness import",
            "principle_engine_fairness_extension": "from principle_engine_fairness_extension import",
            "principle_engine_llm": "from principle_engine_llm import",
            "principle_engine_llm_enhanced": "from principle_engine_llm_enhanced import",
            "principle_engine_llm_example": "from principle_engine_llm_example import",
            "principle_engine_db": "from principle_engine_db import",
            "principle_engine_db_example": "from principle_engine_db_example import",
            "principle_engine_action_evaluator": "from principle_engine_action_evaluator import",
            "principle_decision_points": "from principle_decision_points import",
            "principle_repository": "from principle_repository import",
            "principles_converter": "from principles_converter import",
            "principles_integration": "from principles_integration import",
            "enhanced_principle_evaluator": "from enhanced_principle_evaluator import",
            
            # API gateway extensions
            "api_gateway_system_email": "from api_gateway_system_email import",
            "api_gateway_system_calendar": "from api_gateway_system_calendar import",
            "enhanced_security_privacy_manager": "from enhanced_security_privacy_manager import",
            
            # LLM adapters
            "llm_adapter_interface": "from llm_adapter_interface import",
            "llm_key_manager": "from llm_key_manager import",
            "llm_selector": "from llm_selector import",
            "anthropic_llm_adapter": "from anthropic_llm_adapter import",
            "openai_llm_adapter": "from openai_llm_adapter import",
            "google_llm_adapter": "from google_llm_adapter import",
            "mistral_llm_adapter": "from mistral_llm_adapter import",
            "agent_registry_llm_integration": "from agent_registry_llm_integration import",
            "universal_agent_connector_llm": "from universal_agent_connector_llm import",
            "langchain_integration": "from langchain_integration import",
            
            # Other modules
            "adaptive_bridge_builder": "from adaptive_bridge_builder import",
            "anthropic_agent": "from anthropic_agent import",
            "protocol_adapter": "from protocol_adapter import",
            "ai_framework_detector": "from ai_framework_detector import",
            "universal_agent_connector_quick": "from universal_agent_connector_quick import",
            "empire_agent_card_extensions": "from empire_agent_card_extensions import",
        }
        
        # External dependencies
        self.external_deps = {
            "semver": "import semver",
            "mimetypes": "import mimetypes",
            "dicttoxml": "from dicttoxml import dicttoxml",
            "xmltodict": "import xmltodict",
            "xml.etree.ElementTree": "import xml.etree.ElementTree as ET",
            "marshal": "import marshal",
            "imaplib": "import imaplib",
            "markdown": "import markdown",
            "html": "import html",
            "html.unescape": "from html import unescape",
            "PIL": "from PIL import Image",
            "PIL.Image": "from PIL import Image",
            "PIL.ImageDraw": "from PIL import Image, ImageDraw",
            "PIL.ImageFont": "from PIL import Image, ImageFont",
            "reportlab": "import reportlab",
            "tqdm": "from tqdm import tqdm",
            "emoji": "import emoji",
            "jsonschema": "import jsonschema",
            "cmd": "import cmd",
            "tiktoken": "import tiktoken",
            "pprint": "import pprint",
            "google.generativeai": "import google.generativeai as genai",
            "llama_cpp": "from llama_cpp import Llama",
            "ctransformers": "from ctransformers import AutoModelForCausalLM",
            "langchain_core": "from langchain_core.language_models.llms import BaseLLM",
        }
    
    def fix_missing_imports_in_file(self, file_path: Path):
        """Fix missing imports in a single file."""
        if not file_path.exists():
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Find existing imports to avoid duplicates
            existing_imports = set()
            import_section_end = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    existing_imports.add(line.strip())
                    import_section_end = i + 1
                elif line.strip() and not line.strip().startswith('#'):
                    # Non-import, non-comment line found
                    break
            
            # Collect imports to add
            imports_to_add = []
            
            # Check for missing internal module imports
            for module, import_stmt in self.import_fixes.items():
                # Look for usage patterns like module.ClassName or module.function_name
                pattern = rf'\b{module}\.[\w_]+'
                if re.search(pattern, content):
                    full_import = import_stmt.strip()
                    if full_import not in existing_imports and not any(full_import in imp for imp in existing_imports):
                        imports_to_add.append(full_import)
            
            # Check for missing external dependencies
            for module, import_stmt in self.external_deps.items():
                pattern = rf'\b{re.escape(module)}\b'
                if re.search(pattern, content):
                    if import_stmt not in existing_imports and not any(import_stmt in imp for imp in existing_imports):
                        imports_to_add.append(import_stmt)
            
            if imports_to_add:
                # Sort imports
                imports_to_add = sorted(set(imports_to_add))
                
                # Insert imports after existing imports
                for imp in reversed(imports_to_add):
                    lines.insert(import_section_end, imp)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                self.fixes_applied.append(f"Added {len(imports_to_add)} imports to {file_path.name}")
                
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    def fix_all_missing_imports(self):
        """Fix missing imports in all Python files."""
        print("Scanning for missing imports...")
        
        # Get all Python files
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip __pycache__ and test files for now
            if "__pycache__" in str(file_path):
                continue
                
            self.fix_missing_imports_in_file(file_path)
        
        print(f"\nCompleted {len(self.fixes_applied)} files with import fixes:")
        for fix in self.fixes_applied[:20]:  # Show first 20
            print(f"  - {fix}")
        if len(self.fixes_applied) > 20:
            print(f"  ... and {len(self.fixes_applied) - 20} more")
        
        print("\nMissing import fixes completed!")


def main():
    """Run the missing import fixer."""
    fixer = MissingImportFixer()
    fixer.fix_all_missing_imports()


if __name__ == "__main__":
    main()
