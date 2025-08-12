import logging
import sys
from logging.handlers import RotatingFileHandler
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
        except Exception:
            # Si falla la escritura en la BD, no podemos hacer mucho más,
            # pero evitamos que la aplicación crashee.
            pass
        finally:
            db.close()


def setup_logging(process_name: str):
    # Formato del log
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Handler para la consola
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    # Handler para el archivo (con rotación, 5MB por archivo, guarda 5 archivos)
    file_handler = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Handler para la base de datos
    db_handler = DBHandler(process_name=process_name)
    db_handler.setLevel(logging.INFO)
    db_handler.setFormatter(logging.Formatter('%(message)s'))  # Guardamos solo el mensaje en la DB
    logger.addHandler(db_handler)
