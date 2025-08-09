import logging

# ============================== CONSTANTES ===================================
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"

GREEN_B = "\033[1;32m"   # vert gras
YELLOW_B = "\033[1;33m"  # jaune gras
RED_B = "\033[1;31m"     # rouge gras

GREEN_I = "\033[3;32m"   # vert italique
YELLOW_I = "\033[3;33m"  # jaune italique
RED_I = "\033[3;31m"     # rouge italique

RESET = "\033[0m"

# ================================ CLASSE =====================================
class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.ERROR:
            self._style._fmt = f'{RED_B}[%(levelname)s]{RESET}{RED} %(message)s {RESET}{RED_I}(%(filename)s:%(lineno)d){RESET}'

        elif record.levelno >= logging.WARNING:
            self._style._fmt = f'{YELLOW_B}[%(levelname)s]{RESET}{YELLOW} %(message)s {RESET}{YELLOW_I}(%(filename)s:%(lineno)d){RESET}'

        elif record.levelno >= logging.INFO:
            self._style._fmt = f'{GREEN_B}[%(levelname)s]{RESET} %(message)s'
            
        else:
            self._style._fmt = '[%(levelname)s] %(message)s'
        return super().format(record)

# ============================== FONCTIONS ====================================
def setup_logger(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = ColorFormatter('[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []
    logger.addHandler(handler)

    return logger
