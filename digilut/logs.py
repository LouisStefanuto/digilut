import datetime
import json
import logging.config
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def __init__(self, fmt) -> None:
        super().__init__(fmt)
        self._ignore_keys = {"msg", "args"}

    def format(self, record: logging.LogRecord) -> str:
        message = record.__dict__.copy()
        message["message"] = record.getMessage()
        for key in self._ignore_keys:
            message.pop(key, None)
        if record.exc_info and record.exc_text is None:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message["exc_info"] = record.exc_text
        if record.stack_info:
            message["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(message)


def setup_logging():
    # Determine the root directory of the repository
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "consoleFormatter": {
                "format": "%(asctime)s %(name)s - %(levelname)s: %(message)s"
            },
            "jsonFormatter": {
                "()": JSONFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
        },
        "handlers": {
            "consoleHandler": {
                "level": "INFO",
                "formatter": "consoleFormatter",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "fileHandler": {
                "level": "INFO",
                "formatter": "jsonFormatter",
                "class": "logging.FileHandler",
                "filename": Path(
                    logs_dir,
                    datetime.datetime.now().strftime("%Y%m%d_run.log"),
                ),
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["consoleHandler", "fileHandler"],
                "level": "INFO",
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)


# Set up logging when the module is imported
setup_logging()


def get_logger(name):
    return logging.getLogger(name)
