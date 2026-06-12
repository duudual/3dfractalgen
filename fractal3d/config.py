from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config_defaults(config_path: str | Path | None) -> dict[str, Any]:
  if config_path is None:
    return {}
  path = Path(config_path)
  with path.open("r", encoding="utf-8") as f:
    config = json.load(f)
  if not isinstance(config, dict):
    raise ValueError(f"Config must contain a JSON object: {path}")
  return {str(key).replace("-", "_"): value for key, value in config.items()}


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
  config_parser = argparse.ArgumentParser(add_help=False)
  config_parser.add_argument(
    "--config",
    default=None,
    help="Optional JSON config file. Command-line arguments override config values.",
  )
  config_args, remaining = config_parser.parse_known_args()
  defaults = load_config_defaults(config_args.config)
  if defaults:
    parser.set_defaults(**defaults)
  parser.add_argument(
    "--config",
    default=config_args.config,
    help="Optional JSON config file. Command-line arguments override config values.",
  )
  return parser.parse_args(remaining)
