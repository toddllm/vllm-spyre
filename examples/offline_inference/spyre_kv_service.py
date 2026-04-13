from __future__ import annotations

import argparse
import json

from vllm_spyre.distributed.kv_transfer.kv_connector.v1.persistent_kv_service import (
    PersistentKVServiceClient,
    run_persistent_kv_service,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage the persistent local Spyre KV service."
    )
    parser.add_argument(
        "command",
        choices=("serve", "stats", "clear", "shutdown"),
    )
    parser.add_argument(
        "--socket-path",
        type=str,
        default="/tmp/spyre-kv-persistent.sock",
    )
    args = parser.parse_args()

    if args.command == "serve":
        run_persistent_kv_service(args.socket_path)
        return 0

    client = PersistentKVServiceClient(args.socket_path)
    try:
        if args.command == "stats":
            print(json.dumps(client.stats(), indent=2, sort_keys=True))
        elif args.command == "clear":
            client.clear()
            print(json.dumps({"ok": True, "cleared": True}, indent=2, sort_keys=True))
        elif args.command == "shutdown":
            client.shutdown_service()
            print(json.dumps({"ok": True, "shutdown": True}, indent=2, sort_keys=True))
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
