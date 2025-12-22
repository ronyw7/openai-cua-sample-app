"""
Anthropic Computer Use CLI
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

# Import the adapter from anthropic_cua_loop.py
from anthropic_cua_loop import AnthropicComputerAdapter
from computers.computer import Computer
from computers.default import LocalPlaywrightBrowser

# Import shared utilities
from utils import (
    UsageTracker,
    create_evaluator,
    fetch_final_state,
    get_task_config,
    run_evaluation,
    save_agent_logs,
)


def run_agent_loop_with_tracking(
    computer: Computer,
    task: str,
    model: str = "claude-sonnet-4-20250514",
    start_url: Optional[str] = None,
    usage_tracker: Optional[UsageTracker] = None,
) -> Dict[str, Any]:
    """
    Run the Anthropic computer use agent loop with full tracking.

    Runs until the agent completes (no more tool calls), matching OpenAI behavior.
    Returns dict with results, outputs, usage, and timing.
    """
    client = Anthropic()
    adapter = AnthropicComputerAdapter(computer)

    if usage_tracker is None:
        usage_tracker = UsageTracker(model=model)

    # Navigate to start URL
    if start_url and hasattr(computer, "goto"):
        print(f"Navigating to: {start_url}")
        computer.goto(start_url)
        computer.wait(2000)

    width, height = computer.get_dimensions()

    tools = [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": width,
            "display_height_px": height,
            "display_number": 1,
        }
    ]

    system = f"""You are an autonomous agent. Complete tasks independently on the given site without asking for user confirmation or approval. Execute actions directly and only report back when the task is complete or if you encounter an unrecoverable error. Do not ask clarifying questions - make reasonable assumptions and proceed."""

    messages: list[BetaMessageParam] = [{"role": "user", "content": task}]

    step = 0
    final_response = None
    all_outputs = []

    print(f"Starting agent with task: {task}")
    print(f"Model: {model}")
    print("=" * 60)

    while True:  # Run until agent completes (no more tool calls)
        step += 1
        print(f"\n[Step {step}]")

        try:
            response = client.beta.messages.create(
                model=model,
                max_tokens=4096,
                messages=messages,
                tools=tools,
                system=system,
                betas=["computer-use-2025-01-24"],
            )
        except Exception as e:
            print(f"API Error: {e}")
            all_outputs.append({"type": "error", "error": str(e), "step": step})
            break

        # Track usage
        usage_tracker.add_usage(response.usage, model=model)

        print(f"  Tokens: {response.usage.input_tokens}in/{response.usage.output_tokens}out")

        # Convert response to params format
        content_blocks = []
        for block in response.content:
            if hasattr(block, "text"):
                content_blocks.append(BetaTextBlockParam(type="text", text=block.text))
            elif hasattr(block, "name"):
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        messages.append({"role": "assistant", "content": content_blocks})

        # Process response
        tool_results = []
        has_tool_use = False

        for block in response.content:
            if hasattr(block, "text") and block.text:
                text_preview = block.text[:200] + "..." if len(block.text) > 200 else block.text
                print(f"  Assistant: {text_preview}")
                final_response = block.text
                all_outputs.append(
                    {
                        "type": "text",
                        "text": block.text,
                        "step": step,
                    }
                )

            elif hasattr(block, "name") and block.name == "computer":
                has_tool_use = True
                action = block.input.get("action", "screenshot")
                action_params = {k: v for k, v in block.input.items() if k != "action"}

                print(f"  Action: {action}({action_params})")

                # Execute action
                try:
                    adapter.execute_action(action, **action_params)
                except Exception as e:
                    print(f"  Action error: {e}")
                    tool_results.append(
                        BetaToolResultBlockParam(
                            type="tool_result",
                            tool_use_id=block.id,
                            content=f"Error: {e}",
                            is_error=True,
                        )
                    )
                    all_outputs.append(
                        {
                            "type": "tool_error",
                            "action": action,
                            "error": str(e),
                            "step": step,
                        }
                    )
                    continue

                # Take screenshot
                screenshot_b64 = computer.screenshot()

                tool_results.append(
                    BetaToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=block.id,
                        content=[
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": screenshot_b64,
                                },
                            }
                        ],
                    )
                )

                all_outputs.append(
                    {
                        "type": "tool_use",
                        "action": action,
                        "params": action_params,
                        "step": step,
                        "has_screenshot": True,
                        "screenshot_b64": screenshot_b64,
                    }
                )

        if not has_tool_use:
            print("\n[Agent completed - no more tool calls]")
            break

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    print("=" * 60)
    print(f"Completed in {step} steps")

    return {
        "status": "completed",
        "steps": step,
        "final_response": final_response,
        "outputs": all_outputs,
        "messages": messages,
        "usage": usage_tracker.get_summary(),
    }


def main():
    parser = argparse.ArgumentParser(description="Anthropic Computer Use CLI")
    parser.add_argument("--url", "-u", default="https://www.google.com", help="Starting URL")
    parser.add_argument("--input", "-i", type=str, help="Task input (or use --task-id)", default=None)
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--width", type=int, default=1024, help="Browser window width")
    parser.add_argument("--height", type=int, default=768, help="Browser window height")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation mode")
    parser.add_argument("--task-id", type=str, help="Task ID from tasks.yaml", default=None)
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Load task from registry if provided
    if args.task_id:
        task_config = get_task_config(args.task_id)
        if task_config:
            if not args.input:
                args.input = task_config.get("goal")
            if args.url == "https://www.google.com":
                args.url = task_config.get("initial_url", args.url)
            if task_config.get("evaluate", False):
                args.eval = True
            print(f"Loaded task: {args.task_id}")
            print(f"  Goal: {args.input[:80]}..." if len(args.input or "") > 80 else f"  Goal: {args.input}")
            print(f"  URL: {args.url}")
        else:
            print(f"Warning: Task '{args.task_id}' not found in tasks.yaml")

    # Eval mode
    if args.eval:
        if not args.input:
            print("Error: --eval requires --input or --task-id")
            return

        evaluator = None
        if args.task_id:
            evaluator = create_evaluator(args.task_id)
            if evaluator:
                print(f"Evaluator created for: {args.task_id}")

        all_results = []

        for run_num in range(1, args.runs + 1):
            print(f"\n{'=' * 50}")
            print(f"RUN {run_num}/{args.runs}")
            print(f"{'=' * 50}\n")

            usage_tracker = UsageTracker(model=args.model)
            start_time = time.time()

            with LocalPlaywrightBrowser(headless=args.headless, width=args.width, height=args.height) as computer:
                try:
                    result = run_agent_loop_with_tracking(
                        computer=computer,
                        task=args.input,
                        model=args.model,
                        start_url=args.url,
                        usage_tracker=usage_tracker,
                    )
                    error = None
                except Exception as e:
                    result = {"status": "error", "final_response": None, "outputs": [], "steps": 0}
                    error = str(e)
                    print(f"Error: {e}")

                end_time = time.time()
                total_time = end_time - start_time

                # Fetch final state for evaluation
                final_state = None
                evaluated_success = None
                if evaluator and not error:
                    print("\nFetching final state...")
                    final_state = fetch_final_state(computer, args.url)
                    if final_state:
                        eval_result = run_evaluation(evaluator, final_state, result.get("final_response", ""))
                        evaluated_success = eval_result.get("evaluated_success")
                        print(f"Evaluated success: {evaluated_success}")

                # Save logs using shared function
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("results") / f"anthropic-{args.task_id or 'unknown'}"
                output_dir.mkdir(parents=True, exist_ok=True)

                log_path = save_agent_logs(result.get("outputs", []), output_dir, run_num, timestamp_str)

                if final_state:
                    final_state_path = output_dir / f"run{run_num}_{timestamp_str}_final_state.json"
                    with open(final_state_path, "w") as f:
                        json.dump(final_state, f, indent=2)

                # Build results
                run_results = {
                    "task_id": args.task_id,
                    "run": run_num,
                    "model": args.model,
                    "timestamp": datetime.now().isoformat(),
                    "total_time_seconds": round(total_time, 2),
                    "steps": result.get("steps", 0),
                    "status": result.get("status", "error"),
                    "evaluated_success": evaluated_success,
                    "final_result": result.get("final_response"),
                    "log_path": str(log_path),
                    "error": error,
                    "usage": usage_tracker.get_summary(),
                }

                results_path = output_dir / f"run{run_num}_{timestamp_str}_results.json"
                with open(results_path, "w") as f:
                    json.dump(run_results, f, indent=2)
                print(f"Results saved to: {results_path}")

                all_results.append(run_results)

        # Print summary
        print("\n" + "=" * 50)
        print(f"SUMMARY ({args.runs} runs) - Anthropic {args.model}")
        print("=" * 50)
        successes = sum(1 for r in all_results if r["evaluated_success"] is True)
        failures = sum(1 for r in all_results if r["evaluated_success"] is False)
        errors = sum(1 for r in all_results if r["error"] is not None)
        total_time = sum(r["total_time_seconds"] for r in all_results)
        avg_time = total_time / len(all_results) if all_results else 0
        total_cost = sum(r["usage"]["estimated_cost_usd"] for r in all_results)
        total_tokens = sum(r["usage"]["total_tokens"] for r in all_results)

        print(f"Task ID: {args.task_id}")
        print(f"Success Rate: {successes}/{args.runs} ({100 * successes / args.runs:.0f}%)")
        print(f"Failures: {failures}, Errors: {errors}")
        print(f"Total E2E Time: {total_time:.2f}s (avg: {avg_time:.2f}s)")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Total Tokens: {total_tokens:,}")
        return

    # Interactive mode
    with LocalPlaywrightBrowser(headless=args.headless, width=args.width, height=args.height) as computer:
        usage_tracker = UsageTracker(model=args.model)

        print("Anthropic Computer Use Agent (Interactive Mode)")
        print(f"Model: {args.model}")
        print(f"Starting URL: {args.url}")
        print("Type 'quit' to exit\n")

        if hasattr(computer, "goto"):
            computer.goto(args.url)
            computer.wait(2000)

        while True:
            try:
                user_input = args.input or input("> ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                result = run_agent_loop_with_tracking(
                    computer=computer,
                    task=user_input,
                    model=args.model,
                    usage_tracker=usage_tracker,
                )
                print(f"\nFinal: {result['final_response']}\n")
                args.input = None

            except KeyboardInterrupt:
                print("\nInterrupted")
                break

        # Print session summary
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        summary = usage_tracker.get_summary()
        print(f"API calls: {summary['api_calls']}")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"  Input: {summary['input_tokens']:,}")
        print(f"  Output: {summary['output_tokens']:,}")
        print(f"Estimated cost: ${summary['estimated_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
