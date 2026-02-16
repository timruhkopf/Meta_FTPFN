import subprocess

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        logger.warning("Could not retrieve git hash.")
        return "not-a-git-repo"