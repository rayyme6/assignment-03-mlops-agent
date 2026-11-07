import argparse
import logging

from agent.orchestrator import Agent


def main():
    parser = argparse.ArgumentParser(description="Evidence-first MLOps agent")
    parser.add_argument("question")
    parser.add_argument("--llm", action="store_true", help="Use OpenAI structured planning")
    parser.add_argument(
        "--approve",
        action="append",
        default=[],
        help="Exact action:target, e.g. airflow.trigger:iris_training",
    )
    args = parser.parse_args()
    # Approval originates from the human CLI, never from generated tool arguments.
    if args.approve:
        print("Requested authorization: " + ", ".join(args.approve))
        if input("Type CONFIRM to authorize these operations: ") != "CONFIRM":
            parser.error("Authorization declined")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = Agent().ask(args.question, use_llm=args.llm, approvals=frozenset(args.approve))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
