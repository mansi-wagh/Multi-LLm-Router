import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

with (ROOT_DIR / "config.yaml").open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)


def get_api_key(name: str | None) -> str | None:
    return os.getenv(name) if name else None
