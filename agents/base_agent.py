"""
Base Agent Class for Neural Options Oracle++
OpenAI Agents SDK v0.3.0 Implementation
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from config.logging import get_agents_logger

logger = get_agents_logger()


class BaseAgent(ABC):
    """Base class for all AI agents in the system"""
    
    def __init__(self, client: OpenAI, name: str, model: str = "gpt-4o"):
        self.client = client
        self.name = name
        self.model = model
        self.initialized = False
        self.system_instructions = ""
        
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.system_instructions = self._get_system_instructions()
            self.initialized = True
            logger.info(f"{self.name} agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name} agent: {e}")
            return False
    
    @abstractmethod
    def _get_system_instructions(self) -> str:
        """Get system instructions for the agent"""
        pass
    
    @abstractmethod
    async def analyze(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """Main analysis method - must be implemented by subclasses"""
        pass
    
    async def _make_completion(
        self, 
        messages: list, 
        tools: Optional[list] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Make a completion request to OpenAI"""
        
        try:
            # Prepare the request
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Make the completion
            response = self.client.chat.completions.create(**request_params)
            
            # Extract response content
            message = response.choices[0].message
            content = message.content or ""
            
            # Handle tool calls if present
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    try:
                        tool_calls.append({
                            'id': tool_call.id,
                            'function': tool_call.function.name,
                            'arguments': json.loads(tool_call.function.arguments)
                        })
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool call arguments: {tool_call.function.arguments}")
            
            return {
                'content': content,
                'tool_calls': tool_calls,
                'usage': response.usage.dict() if response.usage else {},
                'model': response.model
            }
            
        except Exception as e:
            logger.error(f"{self.name} completion failed: {e}")
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': self.name,
            'model': self.model,
            'initialized': self.initialized,
            'healthy': True,
            'last_check': datetime.now().isoformat()
        }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from agent"""
        try:
            # Try to extract JSON from the response
            if '```json' in content:
                start = content.find('```json') + 7
                end = content.find('```', start)
                json_str = content[start:end].strip()
            elif '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
            else:
                json_str = content
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"{self.name} failed to parse JSON response: {e}")
            return {'error': 'Failed to parse response', 'raw_content': content}
    
    def _validate_confidence(self, confidence: float) -> float:
        """Validate and normalize confidence score"""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5  # Default confidence