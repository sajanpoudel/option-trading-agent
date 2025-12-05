"""
Base Agent Class for Neural Options Oracle++
OpenAI Agents SDK v0.3.0 Implementation
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from backend.config.logging import get_agents_logger

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
        temperature: float = 0.7,
        response_schema: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make a completion request to OpenAI"""
        
        try:
            # Check if client is available
            if not self.client:
                logger.warning(f"{self.name} client not available, returning fallback response")
                return {
                    'content': json.dumps(self._get_fallback_response()),
                    'tool_calls': []
                }
            
            # Prepare the request
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000
            }
            
            # Use structured outputs if schema provided, otherwise use basic json_object
            if response_schema:
                request_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "analysis_response",
                        "strict": True,
                        "schema": response_schema
                    }
                }
            else:
                request_params["response_format"] = {"type": "json_object"}
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            # Make the completion
            logger.info(f"{self.name} making OpenAI API call with model: {self.model}")
            logger.debug(f"{self.name} request params: {request_params}")
            
            response = self.client.chat.completions.create(**request_params)
            
            # Extract response content
            message = response.choices[0].message
            content = message.content or ""
            
            logger.info(f"{self.name} OpenAI API response received")
            logger.debug(f"{self.name} response content length: {len(content)}")
            logger.debug(f"{self.name} response content preview: {content[:200]}...")
            logger.debug(f"{self.name} full response: {response}")
            
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
            # Return fallback response instead of raising
            return {
                'content': json.dumps(self._get_fallback_response()),
                'tool_calls': []
            }
    
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
            # Handle empty or None content
            if not content or content.strip() == "":
                logger.warning(f"{self.name} received empty response, returning fallback")
                return self._get_fallback_response()
            
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
            
            # Try to parse the JSON
            parsed = json.loads(json_str)
            
            # Validate that we got a proper response
            if not isinstance(parsed, dict):
                logger.warning(f"{self.name} parsed response is not a dict: {type(parsed)}")
                return self._get_fallback_response()
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"{self.name} failed to parse JSON response: {e}")
            logger.error(f"Raw content length: {len(content)}")
            logger.error(f"Raw content (first 500 chars): {repr(content[:500])}")
            if len(content) > 500:
                logger.error(f"Raw content (last 200 chars): {repr(content[-200:])}")
            return self._get_fallback_response()
        except Exception as e:
            logger.error(f"{self.name} unexpected error parsing response: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> Dict[str, Any]:
        """Get a fallback response when parsing fails"""
        return {
            'summary': f'{self.name} analysis completed with limited data',
            'confidence': 0.3,
            'recommendation': 'HOLD',
            'reasoning': 'Analysis completed with fallback data due to API limitations',
            'fallback': True
        }
    
    def _validate_confidence(self, confidence: float) -> float:
        """Validate and normalize confidence score"""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5  # Default confidence