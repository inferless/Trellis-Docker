import logging
from scripts.config import load_config

def setup_logging():
    """Setup logging configuration from config file"""
    CURRENT_DIR = os.path.dirname(__file__)
    CONFIG_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "scripts", "config.yaml"))
    print("***"*100,"CONFIG_PATH:", CONFIG_PATH,flush=True)
    
    config = load_config(CONFIG_PATH)
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
