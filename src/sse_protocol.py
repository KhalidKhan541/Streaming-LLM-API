import json
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class EventType(Enum):
    """SSE event types."""
    TOKEN = 'token'
    TOOL_CALL = 'tool_call'
    TOOL_RESULT = 'tool_result'
    DONE = 'done'
    ERROR = 'error'
    HEARTBEAT = 'heartbeat'
    CURSOR = 'cursor'

@dataclass
class SSEEvent:
    """Single SSE event."""
    event_type: EventType
    data: Dict[str, Any]
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"evt_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    def to_sse_string(self) -> str:
        """Convert to SSE wire format."""
        lines = []
        
        if self.event_type:
            lines.append(f"event: {self.event_type.value}")
        
        if self.id:
            lines.append(f"id: {self.id}")
        
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        
        data_json = json.dumps(self.data)
        for line in data_json.split('\n'):
            lines.append(f"data: {line}")
        
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)

class SSEStream:
    """Manage SSE stream with backpressure and cancellation."""
    
    def __init__(self, max_buffer_size: int = 1000,
                 heartbeat_interval: float = 30.0):
        self.max_buffer_size = max_buffer_size
        self.heartbeat_interval = heartbeat_interval
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_buffer_size)
        self.cancel_token = asyncio.Event()
        self.cursor: str = None
        self.event_history: list = []
        self.is_complete = False
    
    async def send(self, event: SSEEvent):
        """Send event to stream with backpressure."""
        if self.cancel_token.is_set():
            return
        
        try:
            await asyncio.wait_for(
                self.queue.put(event),
                timeout=5.0
            )
            self.event_history.append({
                'id': event.id,
                'type': event.event_type.value,
                'timestamp': datetime.now().isoformat(),
            })
            self.cursor = event.id
        except asyncio.TimeoutError:
            raise Exception("Stream backpressure timeout")
    
    async def send_token(self, token: str, metadata: Dict = None):
        """Send token event."""
        data = {'token': token}
        if metadata:
            data.update(metadata)
        
        event = SSEEvent(event_type=EventType.TOKEN, data=data)
        await self.send(event)
    
    async def send_tool_call(self, tool_name: str, tool_args: Dict,
                            call_id: str = None):
        """Send tool call event."""
        event = SSEEvent(
            event_type=EventType.TOOL_CALL,
            data={
                'tool': tool_name,
                'args': tool_args,
                'call_id': call_id,
            }
        )
        await self.send(event)
    
    async def send_tool_result(self, call_id: str, result: Any):
        """Send tool result event."""
        event = SSEEvent(
            event_type=EventType.TOOL_RESULT,
            data={
                'call_id': call_id,
                'result': result,
            }
        )
        await self.send(event)
    
    async def send_done(self, finish_reason: str = 'stop'):
        """Send done event."""
        event = SSEEvent(
            event_type=EventType.DONE,
            data={'finish_reason': finish_reason}
        )
        await self.send(event)
        self.is_complete = True
    
    async def send_error(self, error: str):
        """Send error event."""
        event = SSEEvent(
            event_type=EventType.ERROR,
            data={'error': error}
        )
        await self.send(event)
    
    def cancel(self):
        """Cancel the stream."""
        self.cancel_token.set()
    
    def is_cancelled(self) -> bool:
        return self.cancel_token.is_set()
    
    async def get_event(self) -> Optional[SSEEvent]:
        """Get next event from queue."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def event_generator(self) -> AsyncGenerator[str, None]:
        """Generate SSE formatted strings."""
        while not self.is_complete:
            if self.cancel_token.is_set():
                break
            
            event = await self.get_event()
            if event:
                yield event.to_sse_string()
    
    def get_events_since(self, cursor: str) -> list:
        """Get events since a cursor for reconnection."""
        if not cursor:
            return self.event_history
        
        cursor_idx = None
        for i, evt in enumerate(self.event_history):
            if evt['id'] == cursor:
                cursor_idx = i
                break
        
        if cursor_idx is not None:
            return self.event_history[cursor_idx + 1:]
        
        return self.event_history
