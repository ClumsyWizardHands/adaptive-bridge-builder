import google.generativeai as genai
"""
Test AI Framework Detector

This module tests the AI framework detector with various code snippets and URLs.
"""

import json
from ai_framework_detector import AIFrameworkDetector, detect_ai_framework


def test_framework_detection() -> Tuple[Any, ...]:
    """Test the framework detection with various examples."""
    detector = AIFrameworkDetector()
    
    # Test cases with expected results
    test_cases = [
        {
            "name": "LangChain with OpenAI",
            "input": """
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
            """,
            "expected_framework": "langchain",
            "min_confidence": 0.7
        },
        {
            "name": "OpenAI Direct Usage",
            "input": """
import openai

client = openai.OpenAI(api_key="sk-...")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
            """,
            "expected_framework": "openai",
            "min_confidence": 0.8
        },
        {
            "name": "Anthropic Claude",
            "input": """
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)
            """,
            "expected_framework": "anthropic",
            "min_confidence": 0.8
        },
        {
            "name": "AutoGPT",
            "input": """
from autogpt.agents import Agent
from autogpt.config import Config
from autogpt.workspace import Workspace

config = Config()
workspace = Workspace()
agent = Agent(
    ai_name="TestAgent",
    memory=memory,
    config=config,
    workspace=workspace
)
agent.run()
            """,
            "expected_framework": "autogpt",
            "min_confidence": 0.7
        },
        {
            "name": "CrewAI",
            "input": """
from crewai import Agent, Task, Crew

researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI',
    backstory="You work at a leading tech think tank.",
    verbose=True,
    allow_delegation=False
)

task1 = Task(
    description="Investigate the latest AI trends",
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[task1],
    verbose=True
)
            """,
            "expected_framework": "crewai",
            "min_confidence": 0.8
        },
        {
            "name": "OpenAI API URL",
            "input": "https://api.openai.com/v1/chat/completions",
            "expected_framework": "openai",
            "min_confidence": 0.5
        },
        {
            "name": "Anthropic API URL",
            "input": "https://api.anthropic.com/v1/messages",
            "expected_framework": "anthropic",
            "min_confidence": 0.5
        },
        {
            "name": "Google Gemini",
            "input": """
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Tell me about machine learning")
            """,
            "expected_framework": "google",
            "min_confidence": 0.7
        },
        {
            "name": "Mistral AI",
            "input": """
from mistral import MistralClient

client = MistralClient(api_key="...")
response = client.chat(
    model="mistral-medium",
    messages=[{"role": "user", "content": "Hello Mistral!"}]
)
            """,
            "expected_framework": "mistral",
            "min_confidence": 0.7
        },
        {
            "name": "Multiple Frameworks",
            "input": """
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
import anthropic

# Using LangChain with OpenAI
openai_llm = OpenAI(temperature=0.7)

# Also using Anthropic directly
claude = anthropic.Anthropic()
            """,
            "expected_primary": "langchain",
            "expected_secondary": ["openai", "anthropic"],
            "is_multi": True
        },
        {
            "name": "Unknown Framework",
            "input": """
import some_random_module

def my_function():
    return "Hello World"
            """,
            "expected_framework": "unknown",
            "min_confidence": 0.0
        },
        {
            "name": "A2A Protocol (Project Specific)",
            "input": """
from universal_agent_connector import AgentProtocolAdapter
from a2a_protocol import AgentCard, AgentMessage

adapter = AgentProtocolAdapter()
agent_card = AgentCard(
    name="Bridge Builder",
    capabilities=["translation", "routing"]
)
            """,
            "expected_framework": "a2a_protocol",
            "min_confidence": 0.7
        }
    ]
    
    print("AI Framework Detector Test Results")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 40)
        
        # Truncate input for display
        input_display = test_case['input']
        if len(input_display) > 100:
            input_display = input_display[:97] + "..."
        print(f"Input: {input_display}")
        
        result = detector.detect_framework(test_case['input'])
        
        print(f"\nResult:")
        print(f"  Framework: {result.framework}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Adapter: {result.suggested_adapter}")
        
        # Check if it's a multi-framework test
        if test_case.get('is_multi', False):
            multi_results = detector.detect_multiple_frameworks(test_case['input'])
            print("\nMultiple Frameworks Detected:")
            for r in multi_results[:3]:
                print(f"  - {r.framework}: {r.confidence:.2f}")
            
            # Verify primary framework
            if result.framework == test_case['expected_primary']:
                # Check if secondary frameworks are detected
                detected_frameworks = [r.framework for r in multi_results]
                all_found = all(fw in detected_frameworks for fw in test_case['expected_secondary'])
                if all_found:
                    print("[PASS] PASS: All expected frameworks detected")
                    passed += 1
                else:
                    print("[FAIL] FAIL: Not all secondary frameworks detected")
                    failed += 1
            else:
                print(f"[FAIL] FAIL: Expected primary '{test_case['expected_primary']}', got '{result.framework}'")
                failed += 1
        else:
            # Single framework test
            if (result.framework == test_case['expected_framework'] and 
                result.confidence >= test_case['min_confidence']):
                print("[PASS] PASS")
                passed += 1
            else:
                print(f"[FAIL] FAIL: Expected '{test_case['expected_framework']}' with confidence >= {test_case['min_confidence']}")
                failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success Rate: {(passed/len(test_cases))*100:.1f}%")
    
    return passed, failed


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    print("\n\nEdge Case Tests")
    print("=" * 80)
    
    detector = AIFrameworkDetector()
    
    edge_cases = [
        ("Empty input", ""),
        ("None input", None),
        ("Whitespace only", "   \n\t  "),
        ("Non-code text", "This is just regular text without any code"),
        ("Malformed URL", "htp:/not-a-valid-url"),
        ("Mixed URL formats", "Check out https://api.openai.com and also anthropic.com"),
    ]
    
    for name, input_text in edge_cases:
        print(f"\nTest: {name}")
        try:
            result = detect_ai_framework(input_text if input_text is not None else "")
            print(f"  Framework: {result['framework']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print("  [PASS] Handled gracefully")
        except Exception as e:
            print(f"  [FAIL] Error: {e}")


def test_url_detection() -> None:
    """Test URL detection specifically."""
    print("\n\nURL Detection Tests")
    print("=" * 80)
    
    urls = [
        "https://api.openai.com/v1/chat/completions",
        "https://api.anthropic.com/v1/messages",
        "https://api.mistral.ai/v1/chat/completions",
        "https://generativelanguage.googleapis.com/v1/models",
        "https://api.cohere.ai/generate",
        "https://api.huggingface.co/models",
        "https://api.langchain.com/v1/chains",
        "www.autogpt.com/api/v1/agents",
        "http://localhost:8000/custom/endpoint",
    ]
    
    detector = AIFrameworkDetector()
    
    for url in urls:
        result = detector.detect_framework(url)
        print(f"\nURL: {url}")
        print(f"  Detected: {result.framework} (confidence: {result.confidence:.2f})")


def demo_usage() -> None:
    """Demonstrate how to use the detector in practice."""
    print("\n\nDemo: How to Use the AI Framework Detector")
    print("=" * 80)
    
    # Example 1: Simple detection
    print("\nExample 1: Simple detection with convenience function")
    code = """
import openai
client = openai.OpenAI()
response = client.chat.completions.create(model="gpt-4", messages=[])
    """
    
    result = detect_ai_framework(code)
    print(f"Code: {code.strip()}")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Example 2: Detailed detection with class
    print("\n\nExample 2: Detailed detection with multiple frameworks")
    code = """
from langchain.llms import OpenAI, Anthropic
from langchain.chains import ConversationChain
import cohere
from typing import Any, Tuple

# Multiple LLMs in use
openai_llm = OpenAI()
claude_llm = Anthropic()
co = cohere.Client()
    """
    
    detector = AIFrameworkDetector()
    single_result = detector.detect_framework(code)
    multi_results = detector.detect_multiple_frameworks(code)
    
    print(f"Primary detection: {single_result.framework} ({single_result.confidence:.2f})")
    print("\nAll detected frameworks:")
    for r in multi_results:
        if r.confidence >= 0.3:
            print(f"  - {r.framework}: {r.confidence:.2f} -> {r.suggested_adapter}")


if __name__ == "__main__":
    # Run all tests
    passed, failed = test_framework_detection()
    test_edge_cases()
    test_url_detection()
    demo_usage()
    
    # Final summary
    print("\n" + "=" * 80)
    print("All tests completed!")
    if failed == 0:
        print("[PASS] All framework detection tests passed!")
    else:
        print(f"[WARNING] {failed} tests failed. Please review the results above.")