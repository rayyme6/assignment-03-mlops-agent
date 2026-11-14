"""Console interface. Confirmation is a separate operator-supplied capability."""

from __future__ import annotations

import argparse
import json
import logging

from agent.executor import MUTATIONS
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="?")
    parser.add_argument("--mode", choices=["rules", "llm"], default="rules")
    parser.add_argument("--confirm", action="append", choices=sorted(MUTATIONS), default=[])
    parser.add_argument("--list-tools", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = Orchestrator(mode=args.mode)
    if args.list_tools:
        print(json.dumps(agent.executor.definitions(), indent=2))
        return
    if args.message:
        request = AgentRequest(message=args.message, confirmed_tools=args.confirm)
        print(agent.chat(request).model_dump_json(indent=2))
        return
    print(
        "MLOps agent. Enter a request; /quit exits. Interactive reads only; use --confirm for writes."
    )
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if message == "/quit":
            return
        if message:
            print(agent.chat(AgentRequest(message=message)).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
