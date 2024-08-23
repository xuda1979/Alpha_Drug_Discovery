# utils/logging_config.py

import logging

def setup_logging(log_file='project.log'):
    """
    Setup logging configuration to log messages to both a file and the console.
    
    Parameters:
    log_file (str): The path of the log file.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
