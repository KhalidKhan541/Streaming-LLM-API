import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, List, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from .sse_protocol import SSEStream, SSEEvent, EventType

@dataclass
class StreamSession:
    """Active streaming session."""
    session_id: str
    stream: SSEStream
    model: str
    messages: List[Dict]
    tools: List[Dict] = field(default_factory=list)
    tool_executor: Callable = None
    created_at: str = None
    last_active: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.last_active = self.created_at

class StreamEngine:
    """Production streaming engine with backpressure and tool interception."""
    
    def __init__(self, max_concurrent_streams: int = 100,
                 stream_timeout: float = 300.0):
        self.max_concurrent_streams = max_concurrent_streams
        self.stream_timeout = stream_timeout
        self.active_sessions: Dict[str, StreamSession] = {}
        self.tool_registry: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._semaphore = asyncio.Semaphore(max_concurrent_streams)
    
    def register_tool(self, name: str, handler: Callable):
        """Register a tool handler for mid-stream execution."""
        self.tool_registry[name] = handler
    
    async def create_session(self, session_id: str, model: str,
                            messages: List[Dict],
                            tools: List[Dict] = None) -> StreamSession:
        """Create a new streaming session."""
        async with self._semaphore:
            stream = SSEStream()
            session = StreamSession(
                session_id=session_id,
                stream=stream,
                model=model,
                messages=messages,
                tools=tools or [],
            )
            self.active_sessions[session_id] = session
            self.logger.info(f"Created session: {session_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[StreamSession]:
        return self.active_sessions.get(session_id)
    
    def cancel_session(self, session_id: str):
        """Cancel an active session."""
        session = self.active_sessions.get(session_id)
        if session:
            session.stream.cancel()
            self.logger.info(f"Cancelled session: {session_id}")
    
    def cleanup_session(self, session_id: str):
        """Remove session from active list."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Cleaned up session: {session_id}")
    
    async def stream_completion(self, session_id: str,
                               prompt: str = None,
                               **kwargs) -> AsyncGenerator[str, None]:
        """Stream LLM completion with tool call interception."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            # Send initial heartbeat
            await session.stream.send(SSEEvent(
                event_type=EventType.HEARTBEAT,
                data={'status': 'connected'}
            ))
            
            # Simulate LLM streaming (replace with actual LLM call)
            async for chunk in self._simulate_llm_stream(prompt, session):
                if session.stream.is_cancelled():
                    await session.stream.send_done('cancelled')
                    break
                
                # Check for tool calls in chunk
                if self._is_tool_call(chunk):
                    tool_result = await self._handle_tool_call(chunk, session)
                    await session.stream.send_tool_result(
                        chunk.get('call_id', ''),
                        tool_result
                    )
                else:
                    await session.stream.send_token(
                        chunk.get('token', ''),
                        metadata={'model': session.model}
                    )
            
            if not session.stream.is_cancelled():
                await session.stream.send_done()
            
        except Exception as e:
            await session.stream.send_error(str(e))
            self.logger.error(f"Stream error: {e}")
        finally:
            self.cleanup_session(session_id)
    
    async def _simulate_llm_stream(self, prompt: str,
                                   session: StreamSession) -> AsyncGenerator[Dict, None]:
        """Simulate LLM token streaming (replace with real LLM)."""
        response = f"This is a simulated streaming response for: {prompt[:50]}..."
        
        for i, char in enumerate(response):
            yield {'token': char, 'position': i}
            await asyncio.sleep(0.02)  # Simulate latency
            
            # Simulate tool call at position 10
            if i == 10 and session.tools:
                yield {
                    'token': '',
                    'is_tool_call': True,
                    'tool': session.tools[0].get('name', 'default_tool'),
                    'args': {'input': prompt},
                    'call_id': f'call_{i}',
                }
    
    def _is_tool_call(self, chunk: Dict) -> bool:
        """Check if chunk contains a tool call."""
        return chunk.get('is_tool_call', False)
    
    async def _handle_tool_call(self, chunk: Dict,
                                session: StreamSession) -> Any:
        """Handle mid-stream tool call execution."""
        tool_name = chunk.get('tool', '')
        tool_args = chunk.get('args', {})
        call_id = chunk.get('call_id', '')
        
        await session.stream.send_tool_call(tool_name, tool_args, call_id)
        
        handler = self.tool_registry.get(tool_name)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**tool_args)
                else:
                    result = handler(**tool_args)
                return result
            except Exception as e:
                return {'error': str(e)}
        
        return {'error': f'Tool {tool_name} not found'}
    
    async def reconnect_session(self, session_id: str,
                               cursor: str = None) -> AsyncGenerator[str, None]:
        """Reconnect to session with cursor resumption."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Send missed events
        missed_events = session.stream.get_events_since(cursor)
        for evt_data in missed_events:
            event = SSEEvent(
                event_type=EventType(evt_data['type']),
                data={'replayed': True},
                id=evt_data['id']
            )
            yield event.to_sse_string()
        
        # Continue streaming
        async for sse_string in session.stream.event_generator():
            yield sse_string
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get info about active sessions."""
        return {
            sid: {
                'model': s.model,
                'created_at': s.created_at,
                'is_cancelled': s.stream.is_cancelled(),
                'cursor': s.stream.cursor,
            }
            for sid, s in self.active_sessions.items()
        }
