import logging
import queue
import threading
from logging.handlers import QueueHandler, QueueListener
from typing import Optional

from veeksha.dashboard.events import DashboardEvent
from veeksha.dashboard.state import DashboardState

class DashboardEventHandler(logging.Handler):
    """Converts log records to dashboard events and applies them to state"""
    
    def __init__(self, dashboard_state: DashboardState):
        super().__init__()
        self.dashboard_state = dashboard_state
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if hasattr(record, 'dashboard_event'):
                event = record.dashboard_event
                if isinstance(event, DashboardEvent):
                    self.dashboard_state.apply(event)
        except Exception:
            self.handleError(record) # handle all errors so we don't crash the main application

class DashboardEventHandlerProcessor:
    """Main pipeline for dashboard event processing"""
    
    def __init__(self, max_queue_size: int = 1000, max_live_requests: int = 50):
        self.event_queue = queue.Queue(maxsize=max_queue_size)
        self.dashboard_state = DashboardState(max_live_requests=max_live_requests)
        self.event_handler = DashboardEventHandler(self.dashboard_state)
        self.queue_listener: Optional[QueueListener] = None
        self.is_running = False
        
        self.logger = logging.getLogger('veeksha.dashboard')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(QueueHandler(self.event_queue))
        
    def start(self) -> DashboardState:
        """Start the dashboard pipeline and return the state object"""
        if self.is_running:
            return self.dashboard_state
            
        self.queue_listener = QueueListener(
            self.event_queue, 
            self.event_handler,
            respect_handler_level=True
        )
        self.queue_listener.start()
        self.is_running = True
        
        return self.dashboard_state
    
    def stop(self) -> None:
        """Stop the dashboard event processor"""
        if self.queue_listener and self.is_running:
            self.queue_listener.stop()
            self.is_running = False
    
    def emit_event(self, event: DashboardEvent) -> None:
        """Emit a dashboard event. Non-blocking."""
        if self.is_running:
            try:
                self.logger.debug("Dashboard event", extra={'dashboard_event': event})
            except queue.Full:
                pass # simply drop events if dashboard can't keep up

# Global event processor singleton
_dashboard_event_processor: Optional[DashboardEventHandlerProcessor] = None

def get_dashboard_event_processor() -> Optional[DashboardEventHandlerProcessor]:
    """Get the global dashboard processor instance"""
    return _dashboard_event_processor

def init_dashboard_event_processor(enabled: bool = True, **kwargs) -> Optional[DashboardState]:
    """Initialize the global dashboard event processor"""
    global _dashboard_event_processor
    
    if not enabled:
        _dashboard_event_processor = None
        return None
        
    _dashboard_event_processor = DashboardEventHandlerProcessor(**kwargs)
    return _dashboard_event_processor.start()

def emit_dashboard_event(event: DashboardEvent) -> None:
    """Emit a dashboard event to the global pipeline."""
    if _dashboard_event_processor:
        _dashboard_event_processor.emit_event(event)