import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import yaml
from agent.agent import Agent
from computers import computers_config
from computers.config import *
from computers.default import *
from utils import usage_tracker

_agisdk_path = Path(__file__).parent.parent / "experiments" / "agisdk" / "src"
if _agisdk_path.exists():
    sys.path.insert(0, str(_agisdk_path))

FINISH_PAGE_WAIT_TIME = 5  # seconds to wait for finish page to load
TASKS_FILE = Path(__file__).parent / "tasks.yaml"


def load_task_registry() -> Dict[str, Dict[str, Any]]:
    """Load tasks from tasks.yaml and return as a dict keyed by task ID."""
    if not TASKS_FILE.exists():
        return {}

    with open(TASKS_FILE) as f:
        tasks_list = yaml.safe_load(f) or []

    # Convert list to dict keyed by ID
    return {task["id"]: task for task in tasks_list if isinstance(task, dict) and "id" in task}


def get_task_config(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task configuration from registry by ID."""
    registry = load_task_registry()
    return registry.get(task_id)


def create_evaluator(task_id: str, version: str = "custom"):
    """Create an evaluator for the task, similar to experiments/runner.py."""
    try:
        from agisdk.REAL.browsergym.webclones.evaluate import WebCloneEvaluator
        from agisdk.REAL.browsergym.webclones.task_config import TaskConfig

        task_config = TaskConfig(task_id, version)
        evaluator = WebCloneEvaluator(task_config)
        return evaluator
    except Exception as e:
        import traceback

        print(f"Failed to create evaluator: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None


def fetch_final_state(computer, initial_url: str) -> Optional[Dict[str, Any]]:
    """Fetch the final state by navigating to the finish page."""
    try:
        finish_url = urljoin(initial_url, "finish")
        print(f"Fetching final state from: {finish_url}")

        computer.goto(finish_url)
        time.sleep(FINISH_PAGE_WAIT_TIME)

        # Get the JSON from the <pre> element
        env_state = computer._page.evaluate("() => document.querySelector('pre')?.textContent || ''")
        if env_state:
            return json.loads(env_state)
        else:
            print("Warning: finish page returned empty state")
            return None

    except Exception as e:
        print(f"Failed to fetch final state: {e}")
        return None


def run_evaluation(evaluator, final_state: Dict[str, Any], final_result: str) -> Dict[str, Any]:
    """Run the evaluator and return results."""
    try:
        env_state = {"final_state": final_state, "final_result": final_result}
        reward, _, message, info = evaluator.evaluate(env_state, final_result)

        evaluated_success = all(result[0] for result in info["results"])

        return {
            "evaluated_success": evaluated_success,
            "reward": reward,
            "message": message,
            "details": info,
        }
    except Exception as e:
        print(f"Failed to run evaluation: {e}")
        return {
            "evaluated_success": None,
            "reward": None,
            "message": None,
            "error": str(e),
        }


def save_agent_logs(
    items: List[Dict[str, Any]],
    output_dir: Path,
    run_num: int,
    timestamp_str: str,
    save_screenshots: bool = True,
) -> Path:
    """
    Save the agent conversation log, optionally extracting screenshots to separate files.

    Args:
        items: The full conversation history from the agent
        output_dir: Directory to save logs
        run_num: Run number for filename
        timestamp_str: Timestamp string for filename
        save_screenshots: Whether to extract and save screenshots separately

    Returns:
        Path to the saved log file
    """
    screenshots_dir = output_dir / f"run{run_num}_{timestamp_str}_screenshots"
    screenshot_count = 0

    # Process items to optionally extract screenshots
    processed_items = []
    for idx, item in enumerate(items):
        processed_item = item.copy()

        # Handle computer_call_output with base64 screenshots
        if item.get("type") == "computer_call_output":
            output = item.get("output", {})
            if isinstance(output, dict) and output.get("type") == "input_image":
                image_url = output.get("image_url", "")
                if image_url.startswith("data:image/png;base64,"):
                    screenshot_count += 1
                    if save_screenshots:
                        # Extract and save screenshot
                        screenshots_dir.mkdir(parents=True, exist_ok=True)
                        base64_data = image_url.replace("data:image/png;base64,", "")
                        screenshot_path = screenshots_dir / f"screenshot_{screenshot_count:03d}.png"

                        import base64

                        with open(screenshot_path, "wb") as f:
                            f.write(base64.b64decode(base64_data))

                        # Replace base64 with file reference in log
                        processed_item = item.copy()
                        processed_item["output"] = output.copy()
                        processed_item["output"]["image_url"] = f"file://{screenshot_path}"
                    else:
                        # Just truncate the base64 data
                        processed_item = item.copy()
                        processed_item["output"] = output.copy()
                        processed_item["output"]["image_url"] = "[base64 screenshot truncated]"

        processed_items.append(processed_item)

    # Save the log file
    log_path = output_dir / f"run{run_num}_{timestamp_str}_log.json"
    with open(log_path, "w") as f:
        json.dump(processed_items, f, indent=2, default=str)

    print(f"Agent log saved to: {log_path}")
    if save_screenshots and screenshot_count > 0:
        print(f"Screenshots saved to: {screenshots_dir}/ ({screenshot_count} files)")

    return log_path


def make_safety_check_callback(autonomous: bool):
    def acknowledge_safety_check_callback(message: str) -> bool:
        if autonomous:
            print(f"Safety Check Warning (auto-acknowledged): {message}")
            return True
        response = input(f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): ").lower()
        return response.lower().strip() == "y"

    return acknowledge_safety_check_callback


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
        default=True,
        help="Run browser in headless mode (no visible window). Screenshots still work.",
    )
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

    # Build kwargs for computer class (headless only supported by local-playwright)
    computer_kwargs = {}
    if args.computer == "local-playwright" and args.headless:
        computer_kwargs["headless"] = True

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

        for run_num in range(1, args.runs + 1):
            print(f"\n{'=' * 50}")
            print(f"RUN {run_num}/{args.runs}")
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
                            "content": "You are an autonomous agent. Complete tasks independently without asking for user confirmation or approval. Execute actions directly and only report back when the task is complete or if you encounter an unrecoverable error. Do not ask clarifying questions - make reasonable assumptions and proceed.",
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
                output_dir = Path("results") / (args.task_id or "unknown")
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
