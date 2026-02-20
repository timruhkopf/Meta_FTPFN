import signal

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GracefulExit(Exception):
    """Custom exception to trigger clean shutdown on signals."""

    pass


def signal_handler(signum, frame):
    signame = signal.Signals(signum).name
    logger.info(f"Signal {signame} received. Triggering graceful shutdown...")
    raise GracefulExit(f"Received {signame}")
