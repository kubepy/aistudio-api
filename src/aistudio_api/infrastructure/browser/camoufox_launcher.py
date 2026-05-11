"""Compatibility launcher for Camoufox Playwright server.

Camoufox's bundled `camoufox.server.launch_server()` currently forwards
`None`-valued options such as `proxy=None`, which breaks startup in some
versions with errors like `proxy: expected object, got null`.

This module launches the same underlying Node script, but prunes `None`
values first so the generated config matches what the browser expects.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path
from typing import Any, Optional

import orjson
from camoufox.server import LAUNCH_SCRIPT, get_nodejs, to_camel_case_dict
from camoufox.utils import launch_options


def _prune_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _prune_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_prune_none(item) for item in value]
    return value


def launch_camoufox_server(*, port: int, headless: bool, proxy: Optional[dict[str, str]] = None):
    cfg = launch_options(port=port, headless=headless, main_world_eval=True, proxy=proxy)
    cfg = _prune_none(cfg)
    nodejs = get_nodejs()
    data = orjson.dumps(to_camel_case_dict(cfg))

    process = subprocess.Popen(
        [nodejs, str(LAUNCH_SCRIPT)],
        cwd=Path(nodejs).parent / "package",
        stdin=subprocess.PIPE,
        text=True,
    )
    if process.stdin:
        process.stdin.write(base64.b64encode(data).decode())
        process.stdin.close()
    process.wait()
    raise RuntimeError("Server process terminated unexpectedly")


def main():
    from aistudio_api.config import build_camoufox_proxy, settings

    parser = argparse.ArgumentParser(description="Launch Camoufox server with sanitized config")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    launch_camoufox_server(
        port=args.port,
        headless=args.headless,
        proxy=build_camoufox_proxy(settings.proxy_url),
    )


if __name__ == "__main__":
    main()
