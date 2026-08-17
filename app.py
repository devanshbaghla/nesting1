#!/usr/bin/env python3
"""Start the web app.

    python app.py                      http://127.0.0.1:8000
    python app.py --port 8080
    python app.py --host 0.0.0.0       reachable from other machines
    python app.py --reload --open      auto-restart on edits, open a browser

Equivalent to `uvicorn app.main:app`, with friendlier defaults and errors.
The command-line nester is a separate entry point: `python cli.py`.
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="app.py",
        description="Run the STL nesting web app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 2)[2],
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default: %(default)s; "
                        "use 0.0.0.0 to expose on your network)")
    p.add_argument("--port", type=int, default=8000,
                   help="port to listen on (default: %(default)s)")
    p.add_argument("--reload", action="store_true",
                   help="restart automatically when source files change")
    p.add_argument("--open", dest="open_browser", action="store_true",
                   help="open the app in your default browser once it is up")
    p.add_argument("--log-level", default="info",
                   choices=["critical", "error", "warning", "info", "debug", "trace"],
                   help="uvicorn log level (default: %(default)s)")
    return p.parse_args(argv)


def port_is_free(host: str, port: int) -> bool:
    """Bind-test the port so a clash fails with a readable message.

    uvicorn's own error for this is a bare OSError traceback, and a stale
    server from an earlier run is the most common way to hit it.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run:\n"
              "    pip install -r requirements.txt", file=sys.stderr)
        return 1

    if not port_is_free(args.host, args.port):
        print(f"port {args.port} is already in use on {args.host}.\n"
              f"Something else is listening there — stop it, or pick another "
              f"port:\n    python app.py --port {args.port + 1}", file=sys.stderr)
        return 1

    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}"
    print(f"\n  STL Nesting  ->  {url}\n  Ctrl-C to stop\n")

    if args.open_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    try:
        uvicorn.run("app.main:app", host=args.host, port=args.port,
                    reload=args.reload, log_level=args.log_level)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
