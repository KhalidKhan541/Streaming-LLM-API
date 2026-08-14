import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

class ToolCallStatus(Enum):
    PENDING = 'pending'
    EXECUTING = 'executing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

@dataclass
class ToolCall:
    """Parsed tool call from stream."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Any = None
    error: str = None
    
    def to_dict(self) -> Dict:
        return {
            'call_id': self.call_id,
            'tool': self.tool_name,
            'args': self.arguments,
            'status': self.status.value,
            'result': self.result,
        }

class ToolInterceptor:
    """Mid-stream tool call interception and execution."""
    
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.pending_calls: Dict[str, ToolCall] = {}
        self.interception_patterns: List[re.Pattern] = []
        self.logger = logging.getLogger(__name__)
        
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default tool call detection patterns."""
        self.interception_patterns = [
            re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL),
            re.compile(r'```json\s*\{.*?"tool".*?\}\s*```', re.DOTALL),
            re.compile(r'\{"tool_call":\s*\{.*?\}\}', re.DOTALL),
        ]
    
    def register_tool(self, name: str, handler: Callable,
                     schema: Dict = None):
        """Register a tool handler."""
        self.handlers[name] = {
            'handler': handler,
            'schema': schema or {},
        }
        self.logger.info(f"Registered tool: {name}")
    
    def detect_tool_call(self, accumulated: str) -> Optional[ToolCall]:
        """Detect if accumulated text contains a tool call."""
        for pattern in self.interception_patterns:
            match = pattern.search(accumulated)
            if match:
                return self._parse_tool_call(match.group(0))
        
        return None
    
    def _parse_tool_call(self, text: str) -> Optional[ToolCall]:
        """Parse tool call from text."""
        try:
            # Try JSON parsing
            text = text.strip()
            
            # Remove markdown code blocks
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            
            # Remove <tool_call> tags
            text = re.sub(r'<tool_call>|</tool_call>', '', text)
            
            data = json.loads(text)
            
            if 'tool' in data:
                tool_name = data['tool']
                args = data.get('arguments', data.get('args', {}))
            elif 'tool_call' in data:
                tool_name = data['tool_call'].get('name', '')
                args = data['tool_call'].get('arguments', {})
            else:
                return None
            
            call_id = data.get('call_id', f"call_{id(text)}")
            
            return ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=args,
            )
            
        except json.JSONDecodeError:
            return None
    
    async def intercept_and_execute(self, accumulated: str,
                                   stream_callback: Callable = None) -> Optional[ToolCall]:
        """Intercept tool call and execute it."""
        tool_call = self.detect_tool_call(accumulated)
        
        if not tool_call:
            return None
        
        self.pending_calls[tool_call.call_id] = tool_call
        
        # Execute tool
        tool_call.status = ToolCallStatus.EXECUTING
        
        try:
            result = await self._execute_tool(tool_call)
            tool_call.result = result
            tool_call.status = ToolCallStatus.COMPLETED
            
            # Notify stream
            if stream_callback:
                await stream_callback(tool_call)
            
            return tool_call
            
        except Exception as e:
            tool_call.error = str(e)
            tool_call.status = ToolCallStatus.FAILED
            self.logger.error(f"Tool execution failed: {e}")
            
            return tool_call
    
    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute a tool call."""
        handler_info = self.handlers.get(tool_call.tool_name)
        
        if not handler_info:
            raise ValueError(f"Tool '{tool_call.tool_name}' not registered")
        
        handler = handler_info['handler']
        
        if asyncio.iscoroutinefunction(handler):
            return await handler(**tool_call.arguments)
        else:
            return handler(**tool_call.arguments)
    
    def cancel_call(self, call_id: str):
        """Cancel a pending tool call."""
        call = self.pending_calls.get(call_id)
        if call:
            call.status = ToolCallStatus.CANCELLED
    
    def extract_clean_response(self, text: str) -> str:
        """Remove tool call markup from response text."""
        for pattern in self.interception_patterns:
            text = pattern.sub('', text)
        
        return text.strip()
    
    def get_pending_calls(self) -> List[Dict]:
        """Get all pending tool calls."""
        return [
            call.to_dict() for call in self.pending_calls.values()
            if call.status in (ToolCallStatus.PENDING, ToolCallStatus.EXECUTING)
        ]
    
    def get_call_history(self) -> List[Dict]:
        """Get history of all tool calls."""
        return [call.to_dict() for call in self.pending_calls.values()]

class StreamingToolExecutor:
    """Execute tools during streaming with cancellation support."""
    
    def __init__(self, interceptor: ToolInterceptor):
        self.interceptor = interceptor
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def execute_with_streaming(self, tool_call: ToolCall,
                                    progress_callback: Callable = None) -> Any:
        """Execute tool with streaming progress updates."""
        async def _execute():
            if progress_callback:
                await progress_callback({
                    'status': 'starting',
                    'tool': tool_call.tool_name,
                })
            
            result = await self.interceptor._execute_tool(tool_call)
            
            if progress_callback:
                await progress_callback({
                    'status': 'completed',
                    'tool': tool_call.tool_name,
                    'result': result,
                })
            
            return result
        
        task = asyncio.create_task(_execute())
        self.active_executions[tool_call.call_id] = task
        
        try:
            return await task
        finally:
            self.active_executions.pop(tool_call.call_id, None)
    
    def cancel_execution(self, call_id: str):
        """Cancel an active execution."""
        task = self.active_executions.get(call_id)
        if task and not task.done():
            task.cancel()
            self.interceptor.cancel_call(call_id)
