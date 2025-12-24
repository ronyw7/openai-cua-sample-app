import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agent import Agent
from computers import computers_config
from computers.config import *
from computers.default import *

# Import shared utilities from utils.py
from utils import (
    create_evaluator,
    fetch_final_state,
    get_next_run_number,
    get_task_config,
    make_safety_check_callback,
    run_evaluation,
    save_agent_logs,
    usage_tracker,
)


def main():
    parser = argparse.ArgumentParser(description="Select a computer environment from the available options.")
    parser.add_argument(
        "--computer",
        choices=computers_config.keys(),
        help="Choose the computer environment to use.",
        default="local-playwright",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Initial input to use instead of asking the user.",
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show images during the execution.",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Start the browsing session with a specific URL (only for browser environments).",
        default="https://bing.com",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Run autonomously without asking for user confirmation before actions.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode: execute single task then exit with metrics.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Task ID from tasks.yaml (e.g., 'dashdish-custom-3'). Loads goal and start-url from registry.",
        default=None,
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of evaluation runs to perform (default: 1).",
        default=1,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (no visible window). Screenshots still work.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Browser window width")
    parser.add_argument("--height", type=int, default=768, help="Browser window height")
    args = parser.parse_args()

    # If task-id is provided, load task config from registry
    if args.task_id:
        task_config = get_task_config(args.task_id)
        if task_config:
            # Override with registry values if not explicitly provided
            if not args.input:
                args.input = task_config.get("goal")
            if args.start_url == "https://bing.com":  # default value
                args.start_url = task_config.get("initial_url", args.start_url)
            # Auto-enable eval mode if task has evaluate: true
            if task_config.get("evaluate", False):
                args.eval = True
            print(f"Loaded task from registry: {args.task_id}")
            print(f"  Goal: {args.input[:80]}..." if len(args.input or "") > 80 else f"  Goal: {args.input}")
            print(f"  URL: {args.start_url}")
        else:
            print(f"Warning: Task '{args.task_id}' not found in tasks.yaml")
    ComputerClass = computers_config[args.computer]

    # Build kwargs for computer class (headless, width, height only supported by local-playwright)
    computer_kwargs = {}
    if args.computer == "local-playwright":
        computer_kwargs["headless"] = args.headless
        computer_kwargs["width"] = args.width
        computer_kwargs["height"] = args.height

    # Eval mode: run task(s) and exit
    if args.eval:
        if not args.input:
            print("Error: --eval requires --input or --task-id to specify the task")
            return

        # Create evaluator once (before runs)
        evaluator = None
        if args.task_id:
            evaluator = create_evaluator(args.task_id)
            if evaluator:
                print(f"Evaluator created for task: {args.task_id}")
            else:
                print(f"Warning: Could not create evaluator for task: {args.task_id}")

        # Track results across runs
        all_results = []

        # Determine starting run number by checking existing runs
        output_dir = Path("results") / (args.task_id or "unknown")
        start_run = get_next_run_number(output_dir)

        for run_num in range(start_run, start_run + args.runs):
            print(f"\n{'=' * 50}")
            print(f"RUN {run_num} (session: {run_num - start_run + 1}/{args.runs})")
            print(f"{'=' * 50}\n")

            # Create fresh browser session for each run
            with ComputerClass(**computer_kwargs) as computer:
                agent = Agent(
                    computer=computer,
                    acknowledge_safety_check_callback=make_safety_check_callback(args.autonomous),
                )
                items = []

                # Add system prompt for autonomous mode
                if args.autonomous:
                    items.append(
                        {
                            "role": "system",
                            "content": "You are an autonomous agent. Complete tasks independently on the given site without asking for user confirmation or approval. Execute actions directly and only report back when the task is complete or if you encounter an unrecoverable error. Do not ask clarifying questions - make reasonable assumptions and proceed. You must use the provided site, i.e., the one that is already opened, and not any other site.",
                        }
                    )

                # Navigate to start URL
                if args.computer in ["browserbase", "local-playwright"]:
                    start_url = args.start_url
                    if not start_url.startswith("http"):
                        start_url = "https://" + start_url
                    agent.computer.goto(start_url)

                # Reset usage tracker for this run
                usage_tracker.reset()
                final_message = None
                error = None

                # Start timing right before agent execution
                start_time = time.time()

                try:
                    items.append({"role": "user", "content": args.input})
                    output_items = agent.run_full_turn(
                        items,
                        print_steps=True,
                        show_images=args.show,
                        debug=args.debug,
                    )
                    end_time = time.time()

                    items += output_items

                    # Extract final message from agent
                    for item in reversed(output_items):
                        if item.get("type") == "message" and item.get("content"):
                            content = item["content"]
                            if isinstance(content, list) and len(content) > 0:
                                final_message = content[0].get("text", "")
                            elif isinstance(content, str):
                                final_message = content
                            break

                except Exception as e:
                    end_time = time.time()
                    error = str(e)
                    print(f"Error during execution: {e}")

                total_time = end_time - start_time

                # Fetch final state
                print("\nFetching final state...")
                final_state = fetch_final_state(computer, args.start_url)

                # Auto-generate output path based on task-id
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save agent logs (full conversation history with screenshots)
                log_path = save_agent_logs(items, output_dir, run_num, timestamp_str, save_screenshots=True)

                # Save final_state to a separate file
                final_state_path = None
                if final_state:
                    final_state_path = output_dir / f"run{run_num}_{timestamp_str}_final_state.json"
                    with open(final_state_path, "w") as f:
                        json.dump(final_state, f, indent=2)
                    print(f"Final state saved to: {final_state_path}")

                # Run evaluation if evaluator is available and no error occurred
                evaluated_success = None
                if evaluator and not error:
                    print("Running evaluation...")
                    if final_state:
                        eval_result = run_evaluation(evaluator, final_state, final_message or "")
                        evaluated_success = eval_result.get("evaluated_success")
                        print(f"Evaluation result: {eval_result.get('message', 'N/A')}")
                        print(f"Evaluated success: {evaluated_success}")
                    else:
                        print("Warning: Could not evaluate - no final state")

                # Build simplified results
                results = {
                    "task_id": args.task_id,
                    "run": run_num,
                    "timestamp": datetime.now().isoformat(),
                    "total_time_seconds": round(total_time, 2),
                    "evaluated_success": evaluated_success,
                    "final_result": final_message,
                    "final_state_path": str(final_state_path) if final_state_path else None,
                    "log_path": str(log_path),
                    "error": error,
                    "usage": usage_tracker.get_summary(),
                }

                # Save results
                results_path = output_dir / f"run{run_num}_{timestamp_str}_results.json"
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {results_path}")

                all_results.append(results)

        # Print summary across all runs
        print("\n" + "=" * 50)
        print(f"SUMMARY ({args.runs} runs)")
        print("=" * 50)
        successes = sum(1 for r in all_results if r["evaluated_success"] is True)
        failures = sum(1 for r in all_results if r["evaluated_success"] is False)
        errors = sum(1 for r in all_results if r["error"] is not None)
        total_time = sum(r["total_time_seconds"] for r in all_results)
        total_cost = sum(r["usage"]["estimated_cost_usd"] for r in all_results)

        print(f"Task ID: {args.task_id}")
        print(f"Success Rate: {successes}/{args.runs} ({100 * successes / args.runs:.0f}%)")
        print(f"Failures: {failures}, Errors: {errors}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Total Cost: ${total_cost:.4f}")

        return

    # Interactive mode
    with ComputerClass(**computer_kwargs) as computer:
        agent = Agent(
            computer=computer,
            acknowledge_safety_check_callback=make_safety_check_callback(args.autonomous),
        )
        items = []

        # Add system prompt for autonomous mode
        if args.autonomous:
            items.append(
                {
                    "role": "system",
                    "content": "You are an autonomous agent. Complete tasks independently without asking for user confirmation or approval. Execute actions directly and only report back when the task is complete or if you encounter an unrecoverable error. Do not ask clarifying questions - make reasonable assumptions and proceed.",
                }
            )

        # Navigate to start URL
        if args.computer in ["browserbase", "local-playwright"]:
            start_url = args.start_url
            if not start_url.startswith("http"):
                start_url = "https://" + start_url
            agent.computer.goto(start_url)

        # Reset usage tracker
        usage_tracker.reset()
        start_time = time.time()
        cumulative_agent_time = 0.0

        while True:
            try:
                user_input = args.input or input("> ")
                if user_input == "exit":
                    break
            except EOFError as e:
                print(f"An error occurred: {e}")
                break
            items.append({"role": "user", "content": user_input})

            # Time each agent turn
            turn_start = time.time()
            output_items = agent.run_full_turn(
                items,
                print_steps=True,
                show_images=args.show,
                debug=args.debug,
            )
            turn_end = time.time()
            cumulative_agent_time += turn_end - turn_start

            items += output_items
            args.input = None

        # Print usage summary on exit
        end_time = time.time()
        total_time = end_time - start_time
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Total session time: {total_time:.2f} seconds")
        print(f"Agent execution time: {cumulative_agent_time:.2f} seconds")
        summary = usage_tracker.get_summary()
        print(f"API calls: {summary['api_calls']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"  Input tokens: {summary['input_tokens']}")
        print(f"  Output tokens: {summary['output_tokens']}")
        print(f"Estimated cost: ${summary['estimated_cost_usd']:.4f}")

        # Save session logs if there were any interactions
        if len(items) > 0:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / "sessions"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_agent_logs(items, output_dir, 1, timestamp_str, save_screenshots=True)


if __name__ == "__main__":
    main()
