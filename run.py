import uvicorn
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description='Streaming LLM API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'],
                       default='info', help='Log level')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print(f"Starting Streaming LLM API on {args.host}:{args.port}")
    print(f"Dashboard: http://localhost:{args.port}/")
    
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )

if __name__ == '__main__':
    main()
