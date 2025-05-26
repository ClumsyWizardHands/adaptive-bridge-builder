"""
Integration Code Generator

Generates integration code snippets for various agent frameworks.
"""

import textwrap
from typing import Dict, List, Optional, Any
from .models import AgentFramework, IntegrationCodeRequest, IntegrationCodeResponse


class IntegrationCodeGenerator:
    """Generates integration code for different agent frameworks"""
    
    def __init__(self) -> None:
        self.base_url = "http://localhost:8000"  # Default, can be configured
        
    def generate_code(self, request: IntegrationCodeRequest) -> IntegrationCodeResponse:
        """
        Generate integration code based on the request
        
        Args:
            request: Integration code request
            
        Returns:
            Integration code response with generated code
        """
        # Get the appropriate generator based on framework
        generator_map = {
            AgentFramework.LANGCHAIN: self._generate_langchain_code,
            AgentFramework.AUTOGPT: self._generate_autogpt_code,
            AgentFramework.OPENAI: self._generate_openai_code,
            AgentFramework.ANTHROPIC: self._generate_anthropic_code,
            AgentFramework.A2A_PROTOCOL: self._generate_a2a_protocol_code,
            AgentFramework.CUSTOM: self._generate_custom_code,
        }
        
        generator = generator_map.get(request.framework, self._generate_generic_code)
        
        # Generate the code
        code, imports, dependencies = generator(request)
        
        # Generate examples if requested
        examples = []
        if request.include_examples:
            examples = self._generate_examples(request.framework, request.agent_id)
            
        # Get documentation URL
        doc_url = self._get_documentation_url(request.framework)
        
        return IntegrationCodeResponse(
            agent_id=request.agent_id,
            framework=request.framework,
            language=request.language,
            code=code,
            imports=imports,
            dependencies=dependencies,
            examples=examples,
            documentation_url=doc_url
        )
        
    def _generate_langchain_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate LangChain integration code"""
        imports = [
            "from langchain.tools import Tool",
            "from langchain.agents import AgentExecutor, create_react_agent",
            "from langchain.memory import ConversationBufferMemory",
            "import requests",
            "import json"
        ]
        
        dependencies = [
            "langchain>=0.1.0",
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class AdaptiveBridgeBuilderTool:
            """Tool for interacting with the Adaptive Bridge Builder agent"""
            
            def __init__(self, agent_id: str, base_url: str = "{self.base_url}"):
                self.agent_id = agent_id
                self.base_url = base_url
                self.session = requests.Session()
                
            def invoke(self, input_data: dict) -> dict:
                """Execute a task through the Adaptive Bridge Builder"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/invoke"
                
                try:
                    response = self.session.post(
                        endpoint,
                        json={{"input": input_data}},
                        headers={{"Content-Type": "application/json"}}
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    return {{"error": str(e)}}
                    

            async def __aenter__(self):
                """Enter async context."""
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                """Exit async context and cleanup."""
                if hasattr(self, 'cleanup'):
                    await self.cleanup()
                elif hasattr(self, 'close'):
                    await self.close()
                return False
        # Create the LangChain tool
        bridge_builder = AdaptiveBridgeBuilderTool(agent_id="{request.agent_id}")
        
        langchain_tool = Tool(
            name="AdaptiveBridgeBuilder",
            description="Facilitates communication between different agent systems",
            func=lambda x: bridge_builder.invoke({{"query": x}})
        )
        ''')
        
        return code, imports, dependencies
        
    def _generate_autogpt_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate AutoGPT integration code"""
        imports = [
            "import requests",
            "import json",
            "from typing import Dict, Any"
        ]
        
        dependencies = [
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class AdaptiveBridgeBuilderPlugin:
            """AutoGPT plugin for Adaptive Bridge Builder integration"""
            
            def __init__(self):
                self.agent_id = "{request.agent_id}"
                self.base_url = "{self.base_url}"
                self.name = "adaptive_bridge_builder"
                self.description = "Connects to Adaptive Bridge Builder for inter-agent communication"
                
            def execute(self, command: str, arguments: Dict[str, Any]) -> str:
                """Execute a command through the Adaptive Bridge Builder"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/execute"
                
                payload = {{
                    "command": command,
                    "arguments": arguments
                }}
                
                try:
                    response = requests.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    return json.dumps(result, indent=2)
                except Exception as e:
                    return f"Error: {{str(e)}}"
                    
            def get_commands(self) -> Dict[str, Dict[str, Any]]:
                """Return available commands"""
                return {{
                    "bridge_communicate": {{
                        "description": "Communicate with another agent",
                        "parameters": {{
                            "target_agent": "The agent to communicate with",
                            "message": "The message to send"
                        }}
                    }},
                    "bridge_query": {{
                        "description": "Query agent capabilities",
                        "parameters": {{
                            "query": "What to query about"
                        }}
                    }}
                }}
        
        # Register the plugin
        bridge_plugin = AdaptiveBridgeBuilderPlugin()
        ''')
        
        return code, imports, dependencies
        
    def _generate_openai_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate OpenAI function calling integration code"""
        imports = [
            "import openai",
            "import requests",
            "import json",
            "from typing import Dict, Any"
        ]
        
        dependencies = [
            "openai>=1.0.0",
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class AdaptiveBridgeBuilderFunction:
            """OpenAI function for Adaptive Bridge Builder integration"""
            
            def __init__(self, agent_id: str = "{request.agent_id}"):
                self.agent_id = agent_id
                self.base_url = "{self.base_url}"
                
            def get_function_definition(self) -> Dict[str, Any]:
                """Get OpenAI function definition"""
                return {{
                    "name": "adaptive_bridge_builder",
                    "description": "Communicate with other agents through Adaptive Bridge Builder",
                    "parameters": {{
                        "type": "object",
                        "properties": {{
                            "action": {{
                                "type": "string",
                                "enum": ["communicate", "query", "register"],
                                "description": "The action to perform"
                            }},
                            "target_agent": {{
                                "type": "string",
                                "description": "Target agent identifier"
                            }},
                            "message": {{
                                "type": "string",
                                "description": "Message or query content"
                            }}
                        }},
                        "required": ["action", "message"]
                    }}
                }}
                
            def execute(self, action: str, message: str, target_agent: str = None) -> Dict[str, Any]:
                """Execute the function"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/function"
                
                payload = {{
                    "action": action,
                    "message": message,
                    "target_agent": target_agent
                }}
                
                try:
                    response = requests.post(endpoint, json=payload)
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    return {{"error": str(e)}}
        
        # Create the function instance
        bridge_function = AdaptiveBridgeBuilderFunction()
        
        # Add to OpenAI functions
        functions = [bridge_function.get_function_definition()]
        ''')
        
        return code, imports, dependencies
        
    def _generate_anthropic_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate Anthropic Claude integration code"""
        imports = [
            "import anthropic",
            "import requests",
            "import json",
            "from typing import Dict, Any"
        ]
        
        dependencies = [
            "anthropic>=0.3.0",
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class AdaptiveBridgeBuilderTool:
            """Anthropic Claude tool for Adaptive Bridge Builder integration"""
            
            def __init__(self, agent_id: str = "{request.agent_id}"):
                self.agent_id = agent_id
                self.base_url = "{self.base_url}"
                self.name = "adaptive_bridge_builder"
                self.description = "Bridge communication between different agent systems"
                
            def get_tool_definition(self) -> Dict[str, Any]:
                """Get tool definition for Claude"""
                return {{
                    "name": self.name,
                    "description": self.description,
                    "input_schema": {{
                        "type": "object",
                        "properties": {{
                            "operation": {{
                                "type": "string",
                                "enum": ["send", "receive", "query"],
                                "description": "Operation to perform"
                            }},
                            "data": {{
                                "type": "object",
                                "description": "Operation data"
                            }}
                        }},
                        "required": ["operation", "data"]
                    }}
                }}
                
            def use(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
                """Use the tool"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/tool"
                
                payload = {{
                    "tool": self.name,
                    "operation": operation,
                    "data": data
                }}
                
                try:
                    response = requests.post(endpoint, json=payload)
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    return {{"error": str(e), "success": False}}
        
        # Create tool instance
        bridge_tool = AdaptiveBridgeBuilderTool()
        
        # Use with Claude
        # tools = [bridge_tool.get_tool_definition()]
        ''')
        
        return code, imports, dependencies
        
    def _generate_a2a_protocol_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate A2A Protocol integration code"""
        imports = [
            "import json",
            "import requests",
            "from typing import Dict, Any, Optional",
            "from datetime import datetime", timezone
        ]
        
        dependencies = [
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class A2AProtocolClient:
            """A2A Protocol client for Adaptive Bridge Builder"""
            
            def __init__(self, agent_id: str = "{request.agent_id}"):
                self.agent_id = agent_id
                self.base_url = "{self.base_url}"
                self.session = requests.Session()
                
            def send_message(self, recipient_id: str, message_type: str, 
                           content: Dict[str, Any]) -> Dict[str, Any]:
                """Send A2A protocol message"""
                endpoint = f"{{self.base_url}}/api/a2a/send"
                
                message = {{
                    "jsonrpc": "2.0",
                    "method": message_type,
                    "params": {{
                        "from": self.agent_id,
                        "to": recipient_id,
                        "content": content,
                        "timestamp": datetime.utcnow().isoformat()
                    }},
                    "id": self._generate_message_id()
                }}
                
                response = self.session.post(endpoint, json=message)
                response.raise_for_status()
                return response.json()
                
            def register_capability(self, capability: Dict[str, Any]) -> bool:
                """Register agent capability"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/capabilities"
                
                response = self.session.post(endpoint, json=capability)
                return response.status_code == 200
                
            def query_agent(self, target_id: str, query: str) -> Dict[str, Any]:
                """Query another agent"""
                return self.send_message(
                    target_id,
                    "agent.query",
                    {{"query": query}}
                )
                
            def _generate_message_id(self) -> str:
                """Generate unique message ID"""
                import uuid
                return str(uuid.uuid4())
        

            async def __aenter__(self):
                """Enter async context."""
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                """Exit async context and cleanup."""
                if hasattr(self, 'cleanup'):
                    await self.cleanup()
                elif hasattr(self, 'close'):
                    await self.close()
                return False
        # Initialize A2A client
        a2a_client = A2AProtocolClient()
        ''')
        
        return code, imports, dependencies
        
    def _generate_custom_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate custom integration code"""
        return self._generate_generic_code(request)
        
    def _generate_generic_code(self, request: IntegrationCodeRequest) -> tuple:
        """Generate generic integration code"""
        imports = [
            "import requests",
            "import json",
            "from typing import Dict, Any"
        ]
        
        dependencies = [
            "requests>=2.28.0"
        ]
        
        code = textwrap.dedent(f'''
        class AdaptiveBridgeBuilderClient:
            """Generic client for Adaptive Bridge Builder integration"""
            
            def __init__(self, agent_id: str = "{request.agent_id}"):
                self.agent_id = agent_id
                self.base_url = "{self.base_url}"
                self.session = requests.Session()
                
            def register(self, agent_info: Dict[str, Any]) -> Dict[str, Any]:
                """Register with the Adaptive Bridge Builder"""
                endpoint = f"{{self.base_url}}/api/agents/register"
                
                payload = {{
                    "agent_id": self.agent_id,
                    **agent_info
                }}
                
                response = self.session.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
                
            def send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
                """Send a generic request"""
                url = f"{{self.base_url}}{{endpoint}}"
                
                response = self.session.post(url, json=data)
                response.raise_for_status()
                return response.json()
                
            def test_connection(self) -> bool:
                """Test connection to the bridge"""
                endpoint = f"{{self.base_url}}/api/agents/{{self.agent_id}}/test"
                
                try:
                    response = self.session.get(endpoint)
                    return response.status_code == 200
                except:
                    return False
        

            async def __aenter__(self):
                """Enter async context."""
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                """Exit async context and cleanup."""
                if hasattr(self, 'cleanup'):
                    await self.cleanup()
                elif hasattr(self, 'close'):
                    await self.close()
                return False
        # Create client instance
        bridge_client = AdaptiveBridgeBuilderClient()
        ''')
        
        return code, imports, dependencies
        
    def _generate_examples(self, framework: AgentFramework, agent_id: str) -> List[str]:
        """Generate usage examples for the framework"""
        examples = []
        
        if framework == AgentFramework.LANGCHAIN:
            examples.append(textwrap.dedent('''
            # Example: Using with LangChain agent
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            
            # Create agent with the tool
            llm = ChatOpenAI(temperature=0)
            tools = [langchain_tool]
            
            # Use in a chain
            response = langchain_tool.run("Query capabilities of agent-123")
            print(response)
            '''))
            
        elif framework == AgentFramework.A2A_PROTOCOL:
            examples.append(textwrap.dedent('''
            # Example: Send A2A message
            response = a2a_client.send_message(
                recipient_id="agent-456",
                message_type="task.request",
                content={
                    "task": "translate",
                    "input": "Hello world",
                    "target_language": "es"
                }
            )
            print(f"Response: {response}")
            
            # Example: Query agent capabilities
            capabilities = a2a_client.query_agent(
                "agent-456",
                "What are your translation capabilities?"
            )
            '''))
            
        # Add a generic example for all frameworks
        examples.append(textwrap.dedent(f'''
        # Example: Test connection
        if bridge_client.test_connection():
            print("Successfully connected to Adaptive Bridge Builder")
        else:
            print("Connection failed")
            
        # Example: Register agent
        registration = bridge_client.register({{
            "name": "My Agent",
            "capabilities": ["translation", "summarization"],
            "version": "1.0.0"
        }})
        print(f"Registered with ID: {{registration.get('agent_id')}}")
        '''))
        
        return examples
        
    def _get_documentation_url(self, framework: AgentFramework) -> Optional[str]:
        """Get documentation URL for the framework"""
        doc_urls = {
            AgentFramework.LANGCHAIN: "https://docs.langchain.com/docs/",
            AgentFramework.AUTOGPT: "https://docs.agpt.co/",
            AgentFramework.OPENAI: "https://platform.openai.com/docs/",
            AgentFramework.ANTHROPIC: "https://docs.anthropic.com/",
            AgentFramework.A2A_PROTOCOL: "https://github.com/hyperledger/aries-rfcs/tree/main/features",
        }
        
        return doc_urls.get(framework, "https://github.com/adaptive-bridge-builder/docs")
