"""
AI Framework Detector

This module provides functionality to detect AI agent frameworks from code snippets
or endpoint URLs. It identifies frameworks like LangChain, AutoGPT, OpenAI, Anthropic,
and others using regex patterns and heuristics.
"""

import re
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import urllib.parse


@dataclass
class FrameworkDetectionResult:
    """Result of framework detection."""
    framework: str
    confidence: float
    suggested_adapter: str
    details: Dict[str, any] = None


class AIFrameworkDetector:
    """
    Detects AI agent frameworks from code snippets or URLs.
    
    Supports detection of:
    - LangChain
    - AutoGPT
    - OpenAI (including GPT-3/4, ChatGPT)
    - Anthropic (Claude)
    - Google (Gemini, PaLM)
    - Mistral
    - Cohere
    - Hugging Face
    - LlamaIndex
    - CrewAI
    - AutoGen
    - SuperAGI
    - BabyAGI
    - MetaGPT
    - AgentGPT
    """
    
    def __init__(self) -> None:
        # Define patterns for each framework
        self.patterns = {
            "langchain": {
                "imports": [
                    r"from\s+langchain",
                    r"import\s+langchain",
                    r"from\s+langchain\.(agents|chains|llms|memory|prompts|tools)",
                    r"LLMChain|ConversationChain|AgentExecutor",
                    r"OpenAI\s*\(\s*\)|ChatOpenAI\s*\(\s*\)"
                ],
                "urls": [
                    r"langchain.*\.com",
                    r"api\.langchain\.",
                    r"smith\.langchain\."
                ],
                "keywords": [
                    r"langchain",
                    r"LangSmith",
                    r"LangServe"
                ],
                "adapter": "langchain_adapter"
            },
            
            "autogpt": {
                "imports": [
                    r"from\s+autogpt",
                    r"import\s+autogpt",
                    r"from\s+auto_gpt",
                    r"AutoGPT|auto_gpt"
                ],
                "urls": [
                    r"autogpt.*\.com",
                    r"significant-gravitas\.com"
                ],
                "keywords": [
                    r"autogpt",
                    r"auto-gpt",
                    r"autonomous\s+gpt"
                ],
                "adapter": "autogpt_adapter"
            },
            
            "openai": {
                "imports": [
                    r"import\s+openai",
                    r"from\s+openai",
                    r"openai\.(ChatCompletion|Completion|Embedding)",
                    r"client\s*=\s*OpenAI\s*\(\s*\)",
                    r"openai\.OpenAI\s*\(\s*\)",
                    r"openai\.api_key",
                    r"\.chat\.completions\.create",
                    r"\.completions\.create"
                ],
                "urls": [
                    r"api\.openai\.com",
                    r"openai\.com/v1",
                    r"platform\.openai\.com"
                ],
                "keywords": [
                    r"gpt-[34]",
                    r"text-davinci",
                    r"text-embedding-ada",
                    r"dall-e",
                    r"whisper",
                    r"openai"
                ],
                "adapter": "openai_llm_adapter"
            },
            
            "anthropic": {
                "imports": [
                    r"import\s+anthropic",
                    r"from\s+anthropic",
                    r"Anthropic\s*\(\s*\)",
                    r"claude\s*=\s*.*Anthropic",
                    r"anthropic\.HUMAN_PROMPT|anthropic\.AI_PROMPT"
                ],
                "urls": [
                    r"api\.anthropic\.com",
                    r"anthropic\.com",
                    r"claude\.ai"
                ],
                "keywords": [
                    r"claude",
                    r"claude-[123]",
                    r"claude-instant",
                    r"claude-v\d",
                    r"anthropic"
                ],
                "adapter": "anthropic_llm_adapter"
            },
            
            "google": {
                "imports": [
                    r"import\s+google\.generativeai",
                    r"from\s+google\.generativeai",
                    r"import\s+vertexai",
                    r"from\s+vertexai",
                    r"genai\.GenerativeModel",
                    r"palm|gemini"
                ],
                "urls": [
                    r"generativelanguage\.googleapis\.com",
                    r"aiplatform\.googleapis\.com",
                    r"makersuite\.google\.com"
                ],
                "keywords": [
                    r"gemini",
                    r"palm",
                    r"bard",
                    r"google\s+ai",
                    r"vertex\s*ai"
                ],
                "adapter": "google_llm_adapter"
            },
            
            "mistral": {
                "imports": [
                    r"from\s+mistral",
                    r"import\s+mistral",
                    r"MistralClient|Mistral",
                    r"mistral_models"
                ],
                "urls": [
                    r"api\.mistral\.ai",
                    r"mistral\.ai"
                ],
                "keywords": [
                    r"mistral",
                    r"mistral-\d+b",
                    r"mixtral"
                ],
                "adapter": "mistral_llm_adapter"
            },
            
            "cohere": {
                "imports": [
                    r"import\s+cohere",
                    r"from\s+cohere",
                    r"cohere\.Client",
                    r"co\s*=\s*cohere"
                ],
                "urls": [
                    r"api\.cohere\.ai",
                    r"cohere\.ai"
                ],
                "keywords": [
                    r"cohere",
                    r"command",
                    r"command-nightly"
                ],
                "adapter": "cohere_adapter"
            },
            
            "huggingface": {
                "imports": [
                    r"from\s+transformers",
                    r"import\s+transformers",
                    r"from\s+huggingface_hub",
                    r"AutoModel|AutoTokenizer|pipeline",
                    r"HuggingFaceHub|HuggingFacePipeline"
                ],
                "urls": [
                    r"api\.huggingface\.co",
                    r"huggingface\.co",
                    r"hf\.co"
                ],
                "keywords": [
                    r"huggingface",
                    r"transformers",
                    r"model_id\s*=",
                    r"hf_.*token"
                ],
                "adapter": "huggingface_adapter"
            },
            
            "llamaindex": {
                "imports": [
                    r"from\s+llama_index",
                    r"import\s+llama_index",
                    r"from\s+gpt_index",
                    r"GPTSimpleVectorIndex|GPTListIndex",
                    r"ServiceContext|StorageContext"
                ],
                "urls": [
                    r"llamaindex\.ai",
                    r"gpt-index"
                ],
                "keywords": [
                    r"llama.?index",
                    r"gpt.?index",
                    r"vector.?store.?index"
                ],
                "adapter": "llamaindex_adapter"
            },
            
            "crewai": {
                "imports": [
                    r"from\s+crewai",
                    r"import\s+crewai",
                    r"Agent\s*\(|Task\s*\(|Crew\s*\(",
                    r"crewai\.(Agent|Task|Crew)"
                ],
                "urls": [
                    r"crewai\.com",
                    r"api\.crewai"
                ],
                "keywords": [
                    r"crewai",
                    r"crew\s+ai",
                    r"agent.*role.*goal.*backstory"
                ],
                "adapter": "crewai_adapter"
            },
            
            "autogen": {
                "imports": [
                    r"import\s+autogen",
                    r"from\s+autogen",
                    r"AssistantAgent|UserProxyAgent|GroupChat",
                    r"autogen\.(AssistantAgent|UserProxyAgent)"
                ],
                "urls": [
                    r"autogen.*\.com",
                    r"microsoft.*autogen"
                ],
                "keywords": [
                    r"autogen",
                    r"auto-gen",
                    r"microsoft.*autogen"
                ],
                "adapter": "autogen_adapter"
            },
            
            "superagi": {
                "imports": [
                    r"from\s+superagi",
                    r"import\s+superagi",
                    r"SuperAGI|super_agi"
                ],
                "urls": [
                    r"superagi\.com",
                    r"api\.superagi"
                ],
                "keywords": [
                    r"superagi",
                    r"super-agi",
                    r"super\s+agi"
                ],
                "adapter": "superagi_adapter"
            },
            
            "babyagi": {
                "imports": [
                    r"baby.?agi",
                    r"BabyAGI|baby_agi"
                ],
                "urls": [
                    r"babyagi"
                ],
                "keywords": [
                    r"babyagi",
                    r"baby-agi",
                    r"baby\s+agi",
                    r"task_creation_agent|execution_agent|prioritization_agent"
                ],
                "adapter": "babyagi_adapter"
            },
            
            "metagpt": {
                "imports": [
                    r"from\s+metagpt",
                    r"import\s+metagpt",
                    r"MetaGPT|meta_gpt"
                ],
                "urls": [
                    r"metagpt"
                ],
                "keywords": [
                    r"metagpt",
                    r"meta-gpt",
                    r"meta\s+gpt"
                ],
                "adapter": "metagpt_adapter"
            },
            
            "agentgpt": {
                "imports": [
                    r"agent.?gpt",
                    r"AgentGPT|agent_gpt"
                ],
                "urls": [
                    r"agentgpt\.reworkd\.ai",
                    r"api\.agentgpt"
                ],
                "keywords": [
                    r"agentgpt",
                    r"agent-gpt",
                    r"agent\s+gpt",
                    r"reworkd"
                ],
                "adapter": "agentgpt_adapter"
            }
        }
        
        # A2A Protocol patterns (for this project)
        self.patterns["a2a_protocol"] = {
            "imports": [
                r"from\s+a2a_protocol",
                r"import\s+a2a_protocol",
                r"A2AProtocol|AgentCard|AgentMessage",
                r"from\s+universal_agent_connector",
                r"AgentProtocolAdapter"
            ],
            "urls": [
                r"a2a-protocol",
                r"agent-to-agent"
            ],
            "keywords": [
                r"a2a.?protocol",
                r"agent.?card",
                r"adaptive.?bridge.?builder"
            ],
            "adapter": "a2a_adapter"
        }
    
    def detect_framework(self, input_text: str) -> FrameworkDetectionResult:
        """
        Detect AI framework from code snippet or URL.
        
        Args:
            input_text: Code snippet or URL to analyze
            
        Returns:
            FrameworkDetectionResult with detected framework, confidence, and suggested adapter
        """
        if not input_text or not input_text.strip():
            return FrameworkDetectionResult(
                framework="unknown",
                confidence=0.0,
                suggested_adapter="generic_adapter",
                details={"error": "Empty input"}
            )
        
        # Check if input is a URL
        if self._is_url(input_text):
            return self._detect_from_url(input_text)
        else:
            return self._detect_from_code(input_text)
    
    def _is_url(self, text: str) -> bool:
        """Check if the input text is a URL."""
        url_pattern = r'^https?://|^www\.|\.com|\.ai|\.io|/api/|/v\d+/'
        return bool(re.search(url_pattern, text, re.IGNORECASE))
    
    def _detect_from_url(self, url: str) -> FrameworkDetectionResult:
        """Detect framework from URL."""
        url_lower = url.lower()
        matches = []
        
        for framework, patterns in self.patterns.items():
            score = 0
            matched_patterns = []
            
            # Check URL patterns
            for pattern in patterns.get("urls", []):
                if re.search(pattern, url_lower):
                    score += 40
                    matched_patterns.append(f"url:{pattern}")
            
            # Check keywords in URL
            for pattern in patterns.get("keywords", []):
                if re.search(pattern, url_lower):
                    score += 20
                    matched_patterns.append(f"keyword:{pattern}")
            
            if score > 0:
                matches.append({
                    "framework": framework,
                    "score": score,
                    "patterns": matched_patterns
                })
        
        if not matches:
            return FrameworkDetectionResult(
                framework="unknown",
                confidence=0.0,
                suggested_adapter="generic_adapter",
                details={"url": url, "message": "No framework detected from URL"}
            )
        
        # Sort by score and get the best match
        matches.sort(key=lambda x: x["score"], reverse=True)
        best_match = matches[0]
        
        # Calculate confidence (0-1 scale)
        confidence = min(best_match["score"] / 100.0, 1.0)
        
        return FrameworkDetectionResult(
            framework=best_match["framework"],
            confidence=confidence,
            suggested_adapter=self.patterns[best_match["framework"]]["adapter"],
            details={
                "url": url,
                "matched_patterns": best_match["patterns"],
                "all_matches": matches[:3]  # Top 3 matches
            }
        )
    
    def _detect_from_code(self, code: str) -> FrameworkDetectionResult:
        """Detect framework from code snippet."""
        code_lower = code.lower()
        matches = []
        
        # Define orchestrator frameworks that get priority
        orchestrator_frameworks = {"langchain", "autogpt", "crewai", "autogen", "llamaindex"}
        
        for framework, patterns in self.patterns.items():
            score = 0
            matched_patterns = []
            
            # Check import patterns (highest weight)
            for pattern in patterns.get("imports", []):
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    score += 50
                    matched_patterns.append(f"import:{pattern}")
            
            # Check keywords (medium weight)
            for pattern in patterns.get("keywords", []):
                count = len(re.findall(pattern, code_lower))
                if count > 0:
                    score += min(20 * count, 60)  # Cap at 60
                    matched_patterns.append(f"keyword:{pattern}({count})")
            
            # Check URL patterns in code (low weight)
            for pattern in patterns.get("urls", []):
                if re.search(pattern, code_lower):
                    score += 10
                    matched_patterns.append(f"url_in_code:{pattern}")
            
            # Give bonus to orchestrator frameworks when multiple are detected
            if framework in orchestrator_frameworks and score > 0:
                score += 20
            
            if score > 0:
                matches.append({
                    "framework": framework,
                    "score": score,
                    "patterns": matched_patterns
                })
        
        if not matches:
            return FrameworkDetectionResult(
                framework="unknown",
                confidence=0.0,
                suggested_adapter="generic_adapter",
                details={"message": "No framework detected from code"}
            )
        
        # Sort by score and get the best match
        matches.sort(key=lambda x: x["score"], reverse=True)
        best_match = matches[0]
        
        # Calculate confidence (0-1 scale)
        # Higher scores mean higher confidence
        confidence = min(best_match["score"] / 150.0, 1.0)
        
        # If multiple frameworks have similar scores, reduce confidence
        if len(matches) > 1 and matches[1]["score"] > best_match["score"] * 0.8:
            confidence *= 0.8
        
        return FrameworkDetectionResult(
            framework=best_match["framework"],
            confidence=confidence,
            suggested_adapter=self.patterns[best_match["framework"]]["adapter"],
            details={
                "matched_patterns": best_match["patterns"],
                "all_matches": matches[:3],  # Top 3 matches
                "code_length": len(code)
            }
        )
    
    def detect_multiple_frameworks(self, code: str) -> List[FrameworkDetectionResult]:
        """
        Detect multiple frameworks that might be used together.
        
        Args:
            code: Code snippet to analyze
            
        Returns:
            List of detected frameworks, sorted by confidence
        """
        if self._is_url(code):
            # For URLs, just return single detection
            return [self.detect_framework(code)]
        
        results = []
        code_lower = code.lower()
        
        for framework, patterns in self.patterns.items():
            score = 0
            matched_patterns = []
            
            # Check import patterns
            for pattern in patterns.get("imports", []):
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    score += 50
                    matched_patterns.append(f"import:{pattern}")
            
            # Check keywords
            for pattern in patterns.get("keywords", []):
                count = len(re.findall(pattern, code_lower))
                if count > 0:
                    score += min(20 * count, 60)
                    matched_patterns.append(f"keyword:{pattern}({count})")
            
            if score >= 50:  # Threshold for considering a framework present
                confidence = min(score / 150.0, 1.0)
                results.append(FrameworkDetectionResult(
                    framework=framework,
                    confidence=confidence,
                    suggested_adapter=self.patterns[framework]["adapter"],
                    details={"matched_patterns": matched_patterns, "score": score}
                ))
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no frameworks detected, return unknown
        if not results:
            results.append(FrameworkDetectionResult(
                framework="unknown",
                confidence=0.0,
                suggested_adapter="generic_adapter",
                details={"message": "No frameworks detected"}
            ))
        
        return results


def detect_ai_framework(input_text: str) -> Dict[str, any]:
    """
    Convenience function to detect AI framework.
    
    Args:
        input_text: Code snippet or endpoint URL
        
    Returns:
        Dictionary with framework, confidence, and suggested_adapter
    """
    detector = AIFrameworkDetector()
    result = detector.detect_framework(input_text)
    
    return {
        "framework": result.framework,
        "confidence": result.confidence,
        "suggested_adapter": result.suggested_adapter,
        "details": result.details
    }


# Example usage
if __name__ == "__main__":
    # Test with various examples
    examples = [
        # LangChain example
        """
        from langchain.llms import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        llm = OpenAI(temperature=0.7)
        prompt = PromptTemplate(template="Tell me about {topic}")
        chain = LLMChain(llm=llm, prompt=prompt)
        """,
        
        # OpenAI example
        """
        import openai
        
        client = OpenAI(api_key="sk-...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        """,
        
        # URL examples
        "https://api.openai.com/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
        "https://api.mistral.ai/v1/chat/completions",
        
        # AutoGPT example
        """
        from autogpt.agents import Agent
        from autogpt.config import Config
        
        config = Config()
        agent = Agent(config=config)
        agent.run()
        """,
        
        # Multiple frameworks
        """
        from langchain.llms import OpenAI
        from langchain.agents import create_react_agent
        import anthropic
        
        # Using both OpenAI and Anthropic
        openai_llm = OpenAI()
        claude = anthropic.Anthropic()
        """
    ]
    
    detector = AIFrameworkDetector()
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}:")
        print(f"Input: {example[:100]}..." if len(example) > 100 else f"Input: {example}")
        
        result = detector.detect_framework(example)
        print(f"\nDetected Framework: {result.framework}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Suggested Adapter: {result.suggested_adapter}")
        if result.details:
            print(f"Details: {result.details}")
        
        # For code examples, also check for multiple frameworks
        if not detector._is_url(example):
            multi_results = detector.detect_multiple_frameworks(example)
            if len(multi_results) > 1:
                print("\nMultiple frameworks detected:")
                for r in multi_results:
                    if r.confidence >= 0.3:  # Only show reasonably confident detections
                        print(f"  - {r.framework}: {r.confidence:.2f}")
