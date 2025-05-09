"""
Configuration module for the multilingual emotion detection system.

This module loads and manages configuration settings from environment variables
and configuration files, providing centralized access to configuration parameters.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Base directory of the project
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default configuration values
DEFAULT_CONFIG = {
    # Model settings
    "MODEL": {
        "PATH": os.path.join("models", "emotion_model"),
        "BATCH_SIZE": 16,
        "AUTO_ADJUST_BATCH": True,
    },
    
    # API settings
    "API": {
        "HOST": "0.0.0.0",
        "PORT": 8000,
        "WORKERS": 4,
        "RATE_LIMIT": 100,  # Requests per minute
        "RATE_LIMIT_WINDOW": 60,  # Seconds
        "JOB_TTL": 86400,  # 24 hours in seconds
        "CLEANUP_INTERVAL": 3600,  # 1 hour in seconds
        "MAX_BATCH_SIZE": 1000,
    },
    
    # Logging settings
    "LOGGING": {
        "LEVEL": "INFO",
        "FILE_PATH": os.path.join(BASE_DIR, "logs", "app.log"),
        "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "MAX_SIZE": 10485760,  # 10 MB
        "BACKUP_COUNT": 5,
    },
    
    # Languages
    "LANGUAGES": {
        "SUPPORTED": ["en", "hi"],
        "DEFAULT": "auto",
    },
    
    # File paths
    "PATHS": {
        "LOGS_DIR": os.path.join(BASE_DIR, "logs"),
        "DATA_DIR": os.path.join(BASE_DIR, "data"),
        "OUTPUT_DIR": os.path.join(BASE_DIR, "output"),
    },
    
    # Processing settings
    "PROCESSING": {
        "DEFAULT_BATCH_SIZE": 16,
        "MAX_WAIT_FOR_RESULTS_BATCH": 50,
    }
}


class Config:
    """Configuration manager for the multilingual emotion detection system."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the configuration if not already initialized."""
        if self._initialized:
            return
            
        self._config = DEFAULT_CONFIG.copy()
        self._load_from_env()
        self._load_from_file()
        self._ensure_directories()
        
        self._initialized = True
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # MODEL settings
        if os.environ.get("MODEL_PATH"):
            self._config["MODEL"]["PATH"] = os.environ.get("MODEL_PATH")
        if os.environ.get("MODEL_BATCH_SIZE"):
            self._config["MODEL"]["BATCH_SIZE"] = int(os.environ.get("MODEL_BATCH_SIZE"))
            
        # API settings
        if os.environ.get("API_HOST"):
            self._config["API"]["HOST"] = os.environ.get("API_HOST")
        if os.environ.get("API_PORT"):
            self._config["API"]["PORT"] = int(os.environ.get("API_PORT"))
        if os.environ.get("API_WORKERS"):
            self._config["API"]["WORKERS"] = int(os.environ.get("API_WORKERS"))
        if os.environ.get("API_RATE_LIMIT"):
            self._config["API"]["RATE_LIMIT"] = int(os.environ.get("API_RATE_LIMIT"))
            
        # Logging settings
        if os.environ.get("LOG_LEVEL"):
            self._config["LOGGING"]["LEVEL"] = os.environ.get("LOG_LEVEL")
        if os.environ.get("LOG_FILE"):
            self._config["LOGGING"]["FILE_PATH"] = os.environ.get("LOG_FILE")
            
        # Processing settings
        if os.environ.get("DEFAULT_BATCH_SIZE"):
            self._config["PROCESSING"]["DEFAULT_BATCH_SIZE"] = int(os.environ.get("DEFAULT_BATCH_SIZE"))
            
    def _load_from_file(self):
        """Load configuration from config file if it exists."""
        config_file = os.environ.get("CONFIG_FILE", os.path.join(BASE_DIR, "config.json"))
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                    
                # Deep merge file_config into self._config
                self._merge_configs(self._config, file_config)
                    
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {str(e)}")
    
    def _merge_configs(self, target: Dict, source: Dict):
        """Recursively merge source config into target config."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_configs(target[key], value)
            else:
                target[key] = value
                
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self._config["PATHS"]["LOGS_DIR"],
            self._config["PATHS"]["DATA_DIR"],
            self._config["PATHS"]["OUTPUT_DIR"],
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")
                
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: The configuration section (e.g., "MODEL", "API")
            key: The configuration key within the section
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default value
        """
        try:
            return self._config[section][key]
        except KeyError:
            return default
            
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: The configuration section (e.g., "MODEL", "API")
            
        Returns:
            A dictionary containing the section configuration or an empty dict
        """
        return self._config.get(section, {}).copy()
        
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the entire configuration.
        
        Returns:
            A copy of the complete configuration dictionary
        """
        return self._config.copy()
        
    def update(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value (runtime only, not persisted).
        
        Args:
            section: The configuration section (e.g., "MODEL", "API")
            key: The configuration key within the section
            value: The new value
        """
        if section in self._config:
            self._config[section][key] = value
        else:
            self._config[section] = {key: value}
            
            
# Global configuration instance
config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        The singleton Config instance
    """
    return config

