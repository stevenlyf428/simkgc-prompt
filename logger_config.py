import logging
from config import args
import os

def _setup_logger():
    if args.is_test:
        log_file = os.path.join(args.model_dir, 'test.log')
    else:
        log_file = os.path.join(args.model_dir, 'train.log')
        

    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    # logger.handlers = [console_handler,file_handler]
    logger.addHandler(console_handler)

    return logger

logger = _setup_logger()