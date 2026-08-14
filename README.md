# Production Streaming LLM API + SSE Engine

Token-level SSE streaming from LLM to client with tool call interception, cancellation, backpressure, and client reconnect.

## Features

| Feature | Description |
|---------|-------------|
| **Token-level SSE** | Real-time token streaming via Server-Sent Events |
| **Tool Call Interception** | Mid-stream detection and execution of tool calls |
| **Cancellation** | Propagate cancellation tokens through stream |
| **Backpressure** | Async flow control with adaptive rate limiting |
| **Client Reconnect** | Cursor-based resumption for dropped connections |
| **JS Stream Renderer** | Client-side markdown parser with live rendering |

## Quick Start

```bash
pip install -r requirements.txt

# Start server
python run.py

# Or with auto-reload
python run.py --reload --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Stream chat completions |
| `/v1/chat/stream/{session_id}` | GET | Reconnect to stream |
| `/v1/chat/cancel/{session_id}` | POST | Cancel active stream |
| `/v1/chat/sessions` | GET | List active sessions |
| `/v1/tools/register` | POST | Register tool handler |
| `/v1/tools` | GET | List registered tools |
| `/v1/health` | GET | Health check |

## Client Usage

### JavaScript SSE Client

```javascript
const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model: 'gpt-4',
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    // Parse SSE events...
}
```

### Python Client

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream(
        'POST',
        'http://localhost:8000/v1/chat/completions',
        json={
            'model': 'gpt-4',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'stream': True
        }
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                print(line[6:])
```

## Architecture

```
Streaming-LLM-API/
├── src/
│   ├── sse_protocol.py        # SSE wire format and events
│   ├── stream_engine.py       # Core streaming engine
│   ├── backpressure.py        # Async flow control
│   ├── tool_interceptor.py    # Mid-stream tool call handling
│   └── api.py                 # FastAPI endpoints
├── run.py                     # Server entry point
├── requirements.txt
└── README.md
```

## Dependencies

- fastapi, uvicorn - ASGI server
- sse-starlette - SSE support
- pydantic - Data validation
