import importlib.metadata
import json
import logging
from logging.config import dictConfig
from typing import Any

from vllm.envs import VLLM_CONFIGURE_LOGGING, VLLM_LOGGING_CONFIG_PATH
from vllm.logger import DEFAULT_LOGGING_CONFIG

__version__ = importlib.metadata.version("vllm_spyre")
logger = logging.getLogger(__name__)

_connector_registered = False


def register():
    """Register the Spyre platform and Spyre-specific KV connector."""
    _register_kv_connector()
    return "vllm_spyre.platform.SpyrePlatform"


def _register_kv_connector() -> None:
    global _connector_registered
    if _connector_registered:
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        KVConnectorFactory.register_connector(
            "InMemorySpyreConnector",
            "vllm_spyre.distributed.kv_transfer.kv_connector.v1.inmemory_spyre_connector",
            "InMemorySpyreConnector",
        )
        _connector_registered = True
    except ValueError:
        # Another import path may have registered it first.
        _connector_registered = True
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        logger.debug("Deferring InMemorySpyreConnector registration: %s", exc)


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = dict[str, Any]()

    if VLLM_CONFIGURE_LOGGING:
        config = {**DEFAULT_LOGGING_CONFIG}

    if VLLM_LOGGING_CONFIG_PATH:
        # Error checks must be done already in vllm.logger.py
        with open(VLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            config = json.loads(file.read())

    if VLLM_CONFIGURE_LOGGING:
        # Copy the vLLM logging configurations for our package
        if "vllm_spyre" not in config["formatters"]:
            if "vllm" in config["formatters"]:
                config["formatters"]["vllm_spyre"] = config["formatters"]["vllm"]
            else:
                config["formatters"]["vllm_spyre"] = DEFAULT_LOGGING_CONFIG["formatters"]["vllm"]

        if "vllm_spyre" not in config["handlers"]:
            if "vllm" in config["handlers"]:
                handler_config = config["handlers"]["vllm"]
            else:
                handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
            handler_config["formatter"] = "vllm_spyre"
            config["handlers"]["vllm_spyre"] = handler_config

        if "vllm_spyre" not in config["loggers"]:
            if "vllm" in config["loggers"]:
                logger_config = config["loggers"]["vllm"]
            else:
                logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
            logger_config["handlers"] = ["vllm_spyre"]
            config["loggers"]["vllm_spyre"] = logger_config

    dictConfig(config)


_init_logging()
