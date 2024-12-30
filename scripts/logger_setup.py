import logging
from scripts.config import load_config

def setup_logging():
    """Setup logging configuration from config file"""
    config = load_config("./config.yaml")
    logging_config = config.get("logging", {})
    
    # Set log level from config
    level = logging_config.get("level", "INFO")
    level = getattr(logging, level.upper())
    
    # Set log format from config
    log_format = logging_config.get("format", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format
    )
    
    # Create and return logger
    logger = logging.getLogger('trellis-api')
    logger.debug("Logging configured with level: %s", level)
    
    return logger
