import logging
import sys
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from .database import SessionLocal
from .models import Log


# Custom DB Handler para escribir logs en la base de datos
class DBHandler(logging.Handler):
    def __init__(self, process_name: str):
        super().__init__()
        self.process_name = process_name

    def emit(self, record):
        db = SessionLocal()
        try:
            log_entry = Log(
                level=record.levelname,
                message=self.format(record),
                process_name=self.process_name
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            # Log error to separate file to avoid infinite recursion
            import sys
            import traceback
            error_msg = f"DBHandler error: {str(e)}\n{traceback.format_exc()}"
            
            # Escribir error a stderr
            print(f"[DBHandler ERROR] {error_msg}", file=sys.stderr)
            
            # Also write to specific error file
            try:
                with open('db_handler_errors.log', 'a', encoding='utf-8') as f:
                    from datetime import datetime
                    timestamp = datetime.now().isoformat()
                    f.write(f"[{timestamp}] {error_msg}\n")
            except:
                # Si incluso esto falla, no hacer nada para evitar crashear la app
                pass
        finally:
            db.close()


def setup_logging(process_name: str, log_level: str = "INFO"):
    """
    Configures logging system with multiple handlers and best practices.
    
    Args:
        process_name: Process name to identify logs
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Crear directorio de logs si no existe
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # More detailed formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Logger principal
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # 1. Handler para la consola
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    # 2. Handler for general file (size rotation)
    general_log_path = os.path.join(log_dir, f'{process_name}_general.log')
    file_handler = RotatingFileHandler(
        general_log_path, 
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # 3. Handler for errors (daily rotation)
    error_log_path = os.path.join(log_dir, f'{process_name}_error.log')
    error_handler = TimedRotatingFileHandler(
        error_log_path,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding='utf-8'
    )
    error_handler.setFormatter(detailed_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # 4. Handler for security (validation and security logs)
    security_log_path = os.path.join(log_dir, f'{process_name}_security.log')
    security_handler = TimedRotatingFileHandler(
        security_log_path,
        when='midnight',
        interval=1,
        backupCount=90,  # Keep 90 days for audit
        encoding='utf-8'
    )
    security_handler.setFormatter(detailed_formatter)
    security_handler.addFilter(SecurityLogFilter())
    logger.addHandler(security_handler)
    
    # 5. Database handler (important messages only)
    try:
        db_handler = DBHandler(process_name=process_name)
        db_handler.setLevel(logging.WARNING)  # Only warnings and errors to DB
        db_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(db_handler)
    except Exception as e:
        # If DB connection fails, continue without DB handler
        print(f"Warning: Could not initialize database logging handler: {e}", file=sys.stderr)
    
    # Log inicial
    logger.info(f"Logging initialized for process: {process_name}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log directory: {os.path.abspath(log_dir)}")
    logger.debug(f"Detailed logging enabled for debugging")
    
    # Configure specific loggers for external modules
    _configure_external_loggers()


class SecurityLogFilter(logging.Filter):
    """
    Filtro para capturar logs relacionados con seguridad.
    """
    def filter(self, record):
        security_keywords = [
            'security', 'validation', 'malicious', 'suspicious', 
            'unauthorized', 'injection', 'sanitiz', 'mime', 'file upload'
        ]
        message_lower = record.getMessage().lower()
        return any(keyword in message_lower for keyword in security_keywords)


def _configure_external_loggers():
    """
    Configures loggers for external libraries with appropriate levels.
    """
    # Reduce verbosity of external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('celery').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Mantener logs importantes de FastAPI
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    

def get_security_logger(name: str = None) -> logging.Logger:
    """
    Gets a specific logger for security events.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger configured for security
    """
    if name:
        logger = logging.getLogger(f"security.{name}")
    else:
        logger = logging.getLogger("security")
    
    return logger
