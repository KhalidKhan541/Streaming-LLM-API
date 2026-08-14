import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

@dataclass
class FlowMetrics:
    """Flow control metrics."""
    tokens_sent: int = 0
    tokens_dropped: int = 0
    avg_latency_ms: float = 0
    queue_depth: int = 0
    backpressure_events: int = 0
    last_flush_time: float = 0

class BackpressureManager:
    """Async backpressure management for streaming."""
    
    def __init__(self, max_queue_size: int = 1000,
                 target_latency_ms: float = 100.0,
                 adaptive: bool = True):
        self.max_queue_size = max_queue_size
        self.target_latency_ms = target_latency_ms
        self.adaptive = adaptive
        
        self.metrics: Dict[str, FlowMetrics] = {}
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self.flush_events: Dict[str, asyncio.Event] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def create_flow(self, flow_id: str, rate_limit: int = None):
        """Create a new flow with backpressure control."""
        self.metrics[flow_id] = FlowMetrics()
        self.rate_limiters[flow_id] = asyncio.Semaphore(
            rate_limit or self.max_queue_size
        )
        self.flush_events[flow_id] = asyncio.Event()
    
    def remove_flow(self, flow_id: str):
        """Remove a flow."""
        self.metrics.pop(flow_id, None)
        self.rate_limiters.pop(flow_id, None)
        self.flush_events.pop(flow_id, None)
    
    async def acquire(self, flow_id: str, timeout: float = 5.0) -> bool:
        """Acquire send permission with backpressure."""
        if flow_id not in self.rate_limiters:
            self.create_flow(flow_id)
        
        try:
            await asyncio.wait_for(
                self.rate_limiters[flow_id].acquire(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            self.metrics[flow_id].backpressure_events += 1
            self.logger.warning(f"Backpressure timeout for flow {flow_id}")
            return False
    
    def release(self, flow_id: str):
        """Release send permission."""
        if flow_id in self.rate_limiters:
            self.rate_limiters[flow_id].release()
    
    async def send_with_backpressure(self, flow_id: str,
                                    data: Any,
                                    send_fn: Callable) -> bool:
        """Send data with backpressure control."""
        start_time = time.time()
        
        if not await self.acquire(flow_id):
            self.metrics[flow_id].tokens_dropped += 1
            return False
        
        try:
            await send_fn(data)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self._update_metrics(flow_id, elapsed_ms)
            
            return True
        finally:
            self.release(flow_id)
    
    def _update_metrics(self, flow_id: str, latency_ms: float):
        """Update flow metrics."""
        metrics = self.metrics.get(flow_id)
        if not metrics:
            return
        
        metrics.tokens_sent += 1
        metrics.last_flush_time = time.time()
        
        # Exponential moving average of latency
        alpha = 0.1
        metrics.avg_latency_ms = alpha * latency_ms + (1 - alpha) * metrics.avg_latency_ms
        
        # Adaptive rate limiting
        if self.adaptive and metrics.avg_latency_ms > self.target_latency_ms * 2:
            self._throttle_flow(flow_id)
    
    def _throttle_flow(self, flow_id: str):
        """Throttle flow when latency is too high."""
        limiter = self.rate_limiters.get(flow_id)
        if limiter and limiter._value > 1:
            # Reduce available slots
            pass
    
    def get_metrics(self, flow_id: str) -> Optional[Dict]:
        """Get metrics for a flow."""
        metrics = self.metrics.get(flow_id)
        if not metrics:
            return None
        
        return {
            'tokens_sent': metrics.tokens_sent,
            'tokens_dropped': metrics.tokens_dropped,
            'avg_latency_ms': metrics.avg_latency_ms,
            'backpressure_events': metrics.backpressure_events,
            'drop_rate': metrics.tokens_dropped / max(metrics.tokens_sent + metrics.tokens_dropped, 1),
        }
    
    def should_flush(self, flow_id: str, batch_size: int = 10) -> bool:
        """Determine if buffer should be flushed."""
        metrics = self.metrics.get(flow_id)
        if not metrics:
            return True
        
        time_since_flush = time.time() - metrics.last_flush_time
        latency_factor = metrics.avg_latency_ms / self.target_latency_ms
        
        if latency_factor > 1.5:
            return time_since_flush > 0.1
        
        return time_since_flush > 0.05 or metrics.queue_depth >= batch_size

class TokenBucket:
    """Token bucket rate limiter for smooth flow control."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket."""
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now
