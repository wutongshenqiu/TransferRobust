import toml

import logging.config
from pathlib import PurePath

from src import settings


def get_logger(logger_name: str, filename: str):
    assert logger_name in {"StreamLogger", "FileLogger"}

    with open(settings.logger_config_file, "r", encoding="utf8") as f:
        config = toml.loads(f.read())

    config["handlers"]["file"]["filename"] = filename
    logging.config.dictConfig(config)

    return logging.getLogger(logger_name)


logger = get_logger(settings.logger_name, settings.log_file)