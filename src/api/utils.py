"""
Utility functions for the API module.
"""

import time
import datetime
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple in-memory rate limiter for API requests."""
    
    def __init__(self, limit: int = 100, window: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.requests = {}  # Maps client ID to list of request timestamps
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request from the client is allowed.
        
        Args:
            client_id: Unique identifier for the client (e.g., IP address)
            
        Returns:
            bool: True if the request is allowed, False otherwise
        """
        now = time.time()
        
        # Initialize if client not seen before
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove expired timestamps
        self.requests[client_id] = [ts for ts in self.requests[client_id] if now - ts < self.window]
        
        # Check if limit is reached
        if len(self.requests[client_id]) >= self.limit:
            return False
        
        # Add current timestamp and allow request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """
        Get the number of remaining requests for a client.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            int: Number of remaining requests
        """
        now = time.time()
        
        # Initialize if client not seen before
        if client_id not in self.requests:
            return self.limit
        
        # Remove expired timestamps
        self.requests[client_id] = [ts for ts in self.requests[client_id] if now - ts < self.window]
        
        # Return remaining requests
        return max(0, self.limit - len(self.requests[client_id]))
    
    def get_reset_time(self, client_id: str) -> float:
        """
        Get the time in seconds until the client's limit resets.
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            float: Time in seconds until reset
        """
        if client_id not in self.requests or not self.requests[client_id]:
            return 0
        
        now = time.time()
        oldest_request = min(self.requests[client_id])
        return max(0, self.window - (now - oldest_request))


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """
    Format a timestamp as an ISO 8601 string.
    
    Args:
        timestamp: UNIX timestamp to format
        
    Returns:
        str: ISO 8601 formatted date-time string
    """
    if timestamp is None:
        timestamp = time.time()
    
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.isoformat()


def estimate_completion_time(total_items: int, completed_items: int, 
                           elapsed_time: float) -> datetime.datetime:
    """
    Estimate job completion time based on current progress.
    
    Args:
        total_items: Total number of items to process
        completed_items: Number of items processed so far
        elapsed_time: Time spent processing so far in seconds
        
    Returns:
        datetime.datetime: Estimated completion datetime
    """
    if completed_items <= 0:
        # If no items completed yet, use a conservative estimate
        time_per_item = 0.5  # Default assumption: 0.5 seconds per item
    else:
        time_per_item = elapsed_time / completed_items
    
    remaining_items = total_items - completed_items
    remaining_time = remaining_items * time_per_item
    
    # Add a 10% buffer for unexpected delays
    remaining_time *= 1.1
    
    return datetime.datetime.fromtimestamp(time.time() + remaining_time)


def cleanup_records(records: Dict[str, Any], max_age_seconds: int = 86400) -> int:
    """
    Remove old records from a dictionary to prevent memory leaks.
    
    Args:
        records: Dictionary of records with timestamps
        max_age_seconds: Maximum age of records in seconds
        
    Returns:
        int: Number of records removed
    """
    now = time.time()
    keys_to_remove = []
    
    for key, record in records.items():
        # Check if record has a timestamp field
        timestamp = record.get('created_at') or record.get('timestamp')
        if timestamp and now - timestamp > max_age_seconds:
            keys_to_remove.append(key)
    
    # Remove expired records
    for key in keys_to_remove:
        del records[key]
    
    return len(keys_to_remove)
