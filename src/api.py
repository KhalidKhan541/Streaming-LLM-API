from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import uuid
import logging

from .stream_engine import StreamEngine
from .tool_interceptor import ToolInterceptor, ToolCall
from .backpressure import BackpressureManager
from .sse_protocol import SSEEvent, EventType

app = FastAPI(title="Streaming LLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
stream_engine = StreamEngine()
tool_interceptor = ToolInterceptor()
backpressure = BackpressureManager()

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    model: str = "gpt-4"
    messages: List[Dict[str, str]]
    tools: Optional[List[Dict]] = None
    stream: bool = True

class ToolRegistration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    endpoint: Optional[str] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    """Stream chat completions with SSE."""
    session_id = str(uuid.uuid4())
    
    # Create session
    session = await stream_engine.create_session(
        session_id=session_id,
        model=body.model,
        messages=body.messages,
        tools=body.tools or [],
    )
    
    if not body.stream:
        # Non-streaming response
        return await _get_completion(session_id, body)
    
    # Streaming response
    async def event_generator():
        try:
            async for sse_string in stream_engine.stream_completion(
                session_id,
                prompt=body.messages[-1].get('content', '') if body.messages else ''
            ):
                yield sse_string
        except asyncio.CancelledError:
            stream_engine.cancel_session(session_id)
        except Exception as e:
            error_event = SSEEvent(
                event_type=EventType.ERROR,
                data={'error': str(e)}
            )
            yield error_event.to_sse_string()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        }
    )

@app.get("/v1/chat/stream/{session_id}")
async def reconnect_stream(session_id: str, cursor: Optional[str] = None):
    """Reconnect to an existing stream with cursor."""
    session = stream_engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_generator():
        async for sse_string in stream_engine.reconnect_session(session_id, cursor):
            yield sse_string
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/v1/chat/cancel/{session_id}")
async def cancel_stream(session_id: str):
    """Cancel an active stream."""
    stream_engine.cancel_session(session_id)
    return {"status": "cancelled", "session_id": session_id}

@app.get("/v1/chat/sessions")
async def list_sessions():
    """List active streaming sessions."""
    return stream_engine.get_active_sessions()

@app.post("/v1/tools/register")
async def register_tool(tool: ToolRegistration):
    """Register a tool for mid-stream execution."""
    async def tool_handler(**kwargs):
        # Placeholder - implement actual tool execution
        return {"result": f"Executed {tool.name}"}
    
    tool_interceptor.register_tool(
        name=tool.name,
        handler=tool_handler,
        schema=tool.parameters,
    )
    
    stream_engine.register_tool(tool.name, tool_handler)
    
    return {"status": "registered", "tool": tool.name}

@app.get("/v1/tools")
async def list_tools():
    """List registered tools."""
    return {"tools": list(tool_interceptor.handlers.keys())}

@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(stream_engine.active_sessions),
        "registered_tools": len(tool_interceptor.handlers),
    }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the client demo page."""
    return get_client_html()

def get_client_html() -> str:
    """Return client-side HTML/JS for SSE consumption."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Streaming LLM API Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #6366f1; }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui; background: var(--bg); color: var(--text); padding: 2rem; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: var(--accent); margin-bottom: 1rem; }
        .chat-box { background: var(--card); border-radius: 8px; padding: 1rem; min-height: 300px; margin: 1rem 0; }
        .message { margin: 0.5rem 0; padding: 0.75rem; border-radius: 6px; }
        .user { background: #334155; }
        .assistant { background: #1e3a5f; }
        .input-row { display: flex; gap: 0.5rem; }
        input { flex: 1; padding: 0.75rem; border-radius: 6px; border: 1px solid #334155; background: var(--card); color: var(--text); }
        button { padding: 0.75rem 1.5rem; border-radius: 6px; border: none; background: var(--accent); color: white; cursor: pointer; }
        button:disabled { opacity: 0.5; }
        .tool-call { background: #3730a3; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-family: monospace; font-size: 0.9em; }
        .status { color: #10b981; font-size: 0.9em; margin: 0.5rem 0; }
        .cursor { color: #64748b; font-size: 0.8em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Streaming LLM API</h1>
        <div class="chat-box" id="chat"></div>
        <div class="status" id="status"></div>
        <div class="cursor" id="cursor"></div>
        <div class="input-row">
            <input type="text" id="input" placeholder="Type a message..." onkeypress="if(event.key==='Enter')send()">
            <button onclick="send()" id="sendBtn">Send</button>
            <button onclick="cancelStream()" id="cancelBtn" disabled>Cancel</button>
        </div>
    </div>
    <script>
        let currentEventSource = null;
        let currentSessionId = null;
        
        async function send() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            document.getElementById('sendBtn').disabled = true;
            document.getElementById('cancelBtn').disabled = false;
            document.getElementById('status').textContent = 'Connecting...';
            
            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: 'gpt-4',
                        messages: [{ role: 'user', content: message }],
                        stream: true
                    })
                });
                
                currentSessionId = response.headers.get('X-Session-Id');
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let buffer = '';
                let assistantDiv = null;
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data === '[DONE]') {
                                document.getElementById('status').textContent = 'Complete';
                                break;
                            }
                            
                            try {
                                const event = JSON.parse(data);
                                handleEvent(event, assistantDiv);
                            } catch (e) {}
                        } else if (line.startsWith('event: ')) {
                            const eventType = line.slice(7);
                            document.getElementById('status').textContent = `Event: ${eventType}`;
                        }
                    }
                }
            } catch (error) {
                document.getElementById('status').textContent = `Error: ${error.message}`;
            }
            
            document.getElementById('sendBtn').disabled = false;
            document.getElementById('cancelBtn').disabled = true;
        }
        
        function handleEvent(event, assistantDiv) {
            if (event.token) {
                if (!assistantDiv) {
                    assistantDiv = addMessage('assistant', '');
                }
                assistantDiv.textContent += event.token;
                updateCursor(event.id);
            }
            if (event.tool) {
                const toolDiv = document.createElement('div');
                toolDiv.className = 'tool-call';
                toolDiv.textContent = `Tool: ${event.tool}(${JSON.stringify(event.args)})`;
                document.getElementById('chat').appendChild(toolDiv);
            }
        }
        
        function addMessage(role, content) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }
        
        function updateCursor(eventId) {
            document.getElementById('cursor').textContent = `Cursor: ${eventId}`;
        }
        
        async function cancelStream() {
            if (currentSessionId) {
                await fetch(`/v1/chat/cancel/${currentSessionId}`, { method: 'POST' });
                document.getElementById('status').textContent = 'Cancelled';
            }
        }
        
        function reconnect() {
            if (currentSessionId) {
                const cursor = document.getElementById('cursor').textContent.split(': ')[1];
                window.location.href = `/v1/chat/stream/${currentSessionId}?cursor=${cursor}`;
            }
        }
    </script>
</body>
</html>"""
