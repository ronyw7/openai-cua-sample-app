"""
Anthropic Computer Use Agent loop using the openai-cua-sample-app's Computer abstraction.
"""

import copy
import os
import sys
import time
from typing import Any, List

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from computers.computer import Computer
from computers.default import LocalPlaywrightBrowser

# See anthropic-quickstarts/computer-use-demo for more details
DEFAULT_MAX_RECENT_IMAGES = 10  # Default number of recent images to keep
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"  # Prompt caching beta flag


def filter_to_n_most_recent_images(
    messages: List[BetaMessageParam],
    n: int,
) -> List[BetaMessageParam]:
    """
    Filter messages to keep only the N most recent images.

    This mirrors the implementation in anthropic-quickstarts/computer-use-demo.
    Older images are replaced with placeholder text to preserve conversation
    structure while reducing token count.

    Args:
        messages: The conversation messages list
        n: Number of most recent images to keep

    Returns:
        A new messages list with older images replaced by placeholders
    """
    if n <= 0:
        return messages  # No filtering

    # Deep copy to avoid modifying original
    messages = copy.deepcopy(messages)

    # Count total images and find their locations
    image_locations = []  # List of (msg_idx, content_idx, item_idx or None)

    for msg_idx, message in enumerate(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if not isinstance(content, list):
            continue

        for content_idx, content_item in enumerate(content):
            # Handle tool_result blocks (which contain images as nested content)
            if isinstance(content_item, dict) and content_item.get("type") == "tool_result":
                tool_content = content_item.get("content")
                if isinstance(tool_content, list):
                    for item_idx, item in enumerate(tool_content):
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_locations.append((msg_idx, content_idx, item_idx))
            # Handle direct image blocks
            elif isinstance(content_item, dict) and content_item.get("type") == "image":
                image_locations.append((msg_idx, content_idx, None))

    # Keep only the N most recent images
    images_to_remove = image_locations[:-n] if len(image_locations) > n else []

    # Replace old images with placeholder text
    for msg_idx, content_idx, item_idx in images_to_remove:
        content = messages[msg_idx]["content"]
        if item_idx is not None:
            # Image is nested inside a tool_result
            tool_content = content[content_idx]["content"]
            tool_content[item_idx] = {
                "type": "text",
                "text": "[Screenshot omitted to reduce context length]",
            }
        else:
            # Direct image block
            content[content_idx] = {
                "type": "text",
                "text": "[Screenshot omitted to reduce context length]",
            }

    return messages


def inject_prompt_caching(messages: List[BetaMessageParam]) -> None:
    """
    Set cache breakpoints for the 3 most recent turns.

    This mirrors the implementation in anthropic-quickstarts/computer-use-demo.
    One cache breakpoint is left for tools/system prompt, to be shared across sessions.

    Modifies messages in-place.
    """
    breakpoints_remaining = 3
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list) and len(content) > 0:
                if breakpoints_remaining:
                    breakpoints_remaining -= 1
                    # Add cache_control to the last content item
                    last_item = content[-1]
                    if isinstance(last_item, dict):
                        last_item["cache_control"] = {"type": "ephemeral"}  # type: ignore
                else:
                    # Remove any existing cache_control from older messages
                    last_item = content[-1]
                    if isinstance(last_item, dict) and "cache_control" in last_item:
                        del last_item["cache_control"]  # type: ignore
                    # We'll only ever have one extra turn per loop
                    break


class AnthropicComputerAdapter:
    """
    Adapts Anthropic's computer_20250124 actions to the Computer interface.

    Supported actions (per https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool):
    - screenshot: Capture the current display
    - left_click: Click at coordinates [x, y]
    - right_click: Right-click at coordinates [x, y]
    - middle_click: Middle-click at coordinates [x, y]
    - double_click: Double-click at coordinates [x, y]
    - triple_click: Triple-click at coordinates [x, y]
    - type: Type a text string
    - key: Press a key or key combination (e.g., "ctrl+s")
    - mouse_move: Move the cursor to specified coordinates
    - scroll: Scroll in a direction at coordinates
    - left_click_drag: Click and drag between coordinates
    - left_mouse_down: Press and hold left mouse button
    - left_mouse_up: Release left mouse button
    - hold_key: Hold a key while performing other actions
    - wait: Pause between actions
    """

    def __init__(self, computer: Computer):
        self.computer = computer
        self._last_mouse_position = (0, 0)
        self._page = getattr(computer, "_page", None)
        self._held_keys: set[str] = set()  # Track keys currently held down (for hold_key action)

    def _release_held_keys(self) -> None:
        """Release all currently held keys."""
        if self._page and self._held_keys:
            for key in list(self._held_keys):
                self._page.keyboard.up(key)
            self._held_keys.clear()

    def execute_action(self, action: str, **params) -> str | None:
        """
        Execute an Anthropic computer action.
        """
        if action == "screenshot":
            # Screenshot is handled by returning the image after this call
            return None

        elif action == "key":
            # Press a key or key combination
            key = params.get("key") or params.get("text", "")
            # Anthropic uses X11 keysym names (xdotool style) like "Return", "Page_Down"
            # Map these to Playwright key names
            key_map = {
                # Enter/Return key
                "Return": "Enter",
                "KP_Enter": "Enter",
                # Backspace
                "BackSpace": "Backspace",
                # Navigation keys (X11 uses underscores)
                "Page_Up": "PageUp",
                "Page_Down": "PageDown",
                "Prior": "PageUp",  # Alternative X11 name
                "Next": "PageDown",  # Alternative X11 name
                # Arrow keys (X11 style)
                "Left": "ArrowLeft",
                "Right": "ArrowRight",
                "Up": "ArrowUp",
                "Down": "ArrowDown",
                # Escape
                "Escape": "Escape",
                # Tab
                "Tab": "Tab",
                "ISO_Left_Tab": "Tab",  # Shift+Tab in X11
                # Space
                "space": " ",
                # Delete/Insert/Home/End
                "Delete": "Delete",
                "Insert": "Insert",
                "Home": "Home",
                "End": "End",
                # Modifier keys (lowercase as used in combos like "ctrl+s")
                "ctrl": "Control",
                "Control_L": "Control",
                "Control_R": "Control",
                "alt": "Alt",
                "Alt_L": "Alt",
                "Alt_R": "Alt",
                "shift": "Shift",
                "Shift_L": "Shift",
                "Shift_R": "Shift",
                "super": "Meta",
                "Super_L": "Meta",
                "Super_R": "Meta",
                "cmd": "Meta",
                "win": "Meta",
                # Function keys are usually the same but ensure consistency
                "F1": "F1",
                "F2": "F2",
                "F3": "F3",
                "F4": "F4",
                "F5": "F5",
                "F6": "F6",
                "F7": "F7",
                "F8": "F8",
                "F9": "F9",
                "F10": "F10",
                "F11": "F11",
                "F12": "F12",
                # Caps Lock
                "Caps_Lock": "CapsLock",
                # Num Lock
                "Num_Lock": "NumLock",
                # Print Screen
                "Print": "PrintScreen",
                # Scroll Lock
                "Scroll_Lock": "ScrollLock",
                # Pause
                "Pause": "Pause",
                # Menu key
                "Menu": "ContextMenu",
            }
            # Split and map each key part
            keys = key.split("+") if "+" in key else [key]
            keys = [key_map.get(k, k) for k in keys]
            self.computer.keypress(keys)

        elif action == "type":
            # Type text
            text = params.get("text", "")
            self.computer.type(text)

        elif action == "mouse_move":
            # Move mouse to coordinate
            coord = params.get("coordinate", [0, 0])
            x, y = coord[0], coord[1]
            self.computer.move(x, y)
            self._last_mouse_position = (x, y)

        elif action == "left_click":
            # Left click at specified coordinate
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
            else:
                x, y = self._last_mouse_position
            self.computer.click(x, y, "left")
            self._last_mouse_position = (x, y)

        elif action == "right_click":
            # Right click at specified coordinate
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
            else:
                x, y = self._last_mouse_position
            self.computer.click(x, y, "right")
            self._last_mouse_position = (x, y)

        elif action == "middle_click":
            # Middle click at specified coordinate
            # Note: Need direct Playwright access as Computer interface doesn't support middle
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
            else:
                x, y = self._last_mouse_position

            if self._page:
                self._page.mouse.click(x, y, button="middle")
            else:
                # Fallback: move to position (can't click middle without direct access)
                self.computer.move(x, y)
                print("Warning: middle_click requires direct Playwright access")
            self._last_mouse_position = (x, y)

        elif action == "double_click":
            # Double click at specified coordinate
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
            else:
                x, y = self._last_mouse_position
            self.computer.double_click(x, y)
            self._last_mouse_position = (x, y)

        elif action == "triple_click":
            # Triple click at specified coordinate (for selecting paragraphs, etc.)
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
            else:
                x, y = self._last_mouse_position

            if self._page:
                self._page.mouse.click(x, y, click_count=3)
            else:
                # Fallback: three rapid clicks
                self.computer.click(x, y, "left")
                self.computer.click(x, y, "left")
                self.computer.click(x, y, "left")
            self._last_mouse_position = (x, y)

        elif action == "left_click_drag":
            # Drag from start to end coordinate
            start = params.get("start_coordinate", [0, 0])
            end = params.get("end_coordinate", [0, 0])
            path = [
                {"x": start[0], "y": start[1]},
                {"x": end[0], "y": end[1]},
            ]
            self.computer.drag(path)
            self._last_mouse_position = (end[0], end[1])

        elif action == "left_mouse_down":
            # Press and hold left mouse button at coordinate
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
                self.computer.move(x, y)
                self._last_mouse_position = (x, y)

            if self._page:
                self._page.mouse.down()
            else:
                print("Warning: left_mouse_down requires direct Playwright access")

        elif action == "left_mouse_up":
            # Release left mouse button at coordinate
            coord = params.get("coordinate")
            if coord:
                x, y = coord[0], coord[1]
                self.computer.move(x, y)
                self._last_mouse_position = (x, y)

            if self._page:
                self._page.mouse.up()
            else:
                print("Warning: left_mouse_up requires direct Playwright access")

        elif action == "scroll":
            coord = params.get("coordinate", [0, 0])
            direction = params.get("direction", params.get("scroll_direction", "down"))
            amount = params.get("amount", params.get("scroll_amount", 3))

            x, y = coord[0], coord[1]

            delta_x, delta_y = 0, 0
            pixels_per_click = 100

            if direction == "down":
                delta_y = amount * pixels_per_click
            elif direction == "up":
                delta_y = -amount * pixels_per_click
            elif direction == "right":
                delta_x = amount * pixels_per_click
            elif direction == "left":
                delta_x = -amount * pixels_per_click

            if self._page:
                self._page.mouse.move(x, y)
                self._page.mouse.wheel(delta_x, delta_y)
                time.sleep(0.1)
            else:
                self.computer.scroll(x, y, delta_x, delta_y)

            self._last_mouse_position = (x, y)

        elif action == "hold_key":
            key = params.get("key", "")
            if self._page:
                key_map = {
                    "ctrl": "Control",
                    "cmd": "Meta",
                    "alt": "Alt",
                    "shift": "Shift",
                    "super": "Meta",
                    "win": "Meta",
                }
                mapped_key = key_map.get(key.lower(), key)
                self._page.keyboard.down(mapped_key)
                self._held_keys.add(mapped_key)
            else:
                print("Warning: hold_key requires direct Playwright access")

        elif action == "wait":
            ms = params.get("duration", 1000)
            self.computer.wait(ms)

        else:
            print(f"Warning: Unknown action '{action}', ignoring")

        # Release any held keys after each action
        if action != "hold_key":
            self._release_held_keys()

        return None


def create_anthropic_response(
    client: Anthropic,
    model: str,
    messages: list[BetaMessageParam],
    tools: list[dict],
    system: str | None = None,
    max_tokens: int = 4096,
) -> tuple[Any, list[BetaContentBlockParam]]:
    """Call Anthropic API and return (response, content_blocks)."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        system=system or "",
        betas=["computer-use-2025-01-24"],
    )

    content_blocks = []
    for block in response.content:
        if hasattr(block, "text"):
            content_blocks.append(BetaTextBlockParam(type="text", text=block.text))
        elif hasattr(block, "name"):
            # Tool use block - only include API-accepted fields
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )

    return response, content_blocks


def run_agent_loop(
    computer: Computer,
    task: str,
    model: str = "claude-sonnet-4-20250514",
    start_url: str | None = None,
) -> dict:
    """
    Run the Anthropic computer use agent loop.

    Runs until the agent completes (no more tool calls), matching OpenAI behavior.

    Args:
        computer: Computer interface to use
        task: The task to perform
        model: Anthropic model to use
        start_url: Optional URL to navigate to first

    Returns:
        Dict with results including messages, steps, and final response
    """
    client = Anthropic()
    adapter = AnthropicComputerAdapter(computer)

    # Navigate to start URL if provided
    if start_url and hasattr(computer, "goto"):
        print(f"Navigating to: {start_url}")
        computer.goto(start_url)
        computer.wait(2000)  # Wait for page load

    # Get display dimensions
    width, height = computer.get_dimensions()

    # Define tools
    tools = [
        {
            "type": "computer_20250124",
            "name": "computer",
            "display_width_px": width,
            "display_height_px": height,
            "display_number": 1,
        }
    ]

    # System prompt
    system = f"""You are an autonomous agent. Complete tasks independently on the given site without asking for user confirmation or approval. Execute actions directly and only report back when the task is complete or if you encounter an unrecoverable error. Do not ask clarifying questions - make reasonable assumptions and proceed. You must use the provided site, i.e., the one that is already opened, and not any other site."""

    # Initial message with task
    messages: list[BetaMessageParam] = [{"role": "user", "content": task}]

    step = 0
    final_response = None
    all_outputs = []
    error_message = None

    print(f"Starting agent with task: {task}")
    print(f"Model: {model}")
    print("=" * 60)

    while True:  # Run until agent completes i.e., no more tool calls
        step += 1
        print(f"\n[Step {step}]")

        try:
            response, content_blocks = create_anthropic_response(
                client=client,
                model=model,
                messages=messages,
                tools=tools,
                system=system,
            )
        except Exception as e:
            error_message = str(e)
            print(f"API Error: {error_message}")
            break

        messages.append(
            {
                "role": "assistant",
                "content": content_blocks,
            }
        )

        tool_results = []
        has_tool_use = False

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"  Assistant: {block.text[:200]}...")
                final_response = block.text
                all_outputs.append({"type": "text", "text": block.text, "step": step})

            elif hasattr(block, "name") and block.name == "computer":
                has_tool_use = True
                action = block.input.get("action", "screenshot")
                action_params = {k: v for k, v in block.input.items() if k != "action"}

                print(f"  Action: {action}({action_params})")
                all_outputs.append(
                    {
                        "type": "tool_use",
                        "action": action,
                        "params": action_params,
                        "step": step,
                    }
                )

                # Execute the action
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
                    continue

                # Take screenshot after action
                screenshot_b64 = computer.screenshot()

                # Build tool result with screenshot
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

        # If no tool use, we're done
        if not has_tool_use:
            print("\n[Agent completed - no more tool calls]")
            break

        # Add tool results to messages
        if tool_results:
            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

    print("=" * 60)
    print(f"Completed in {step} steps")

    if error_message:
        return {
            "status": "error",
            "error": error_message,
            "steps": step,
            "final_response": final_response,
            "outputs": all_outputs,
            "messages": messages,
        }

    return {
        "status": "completed",
        "steps": step,
        "final_response": final_response,
        "outputs": all_outputs,
        "messages": messages,
    }


def main():
    """Interactive CLI for Anthropic computer use agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Anthropic Computer Use Agent")
    parser.add_argument("--task", "-t", help="Task to perform (or interactive if not provided)")
    parser.add_argument("--url", "-u", default=os.getenv("START_URL", "https://www.google.com"), help="Starting URL")
    parser.add_argument("--model", "-m", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--width", type=int, default=1024, help="Browser window width")
    parser.add_argument("--height", type=int, default=768, help="Browser window height")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Create the computer environment
    with LocalPlaywrightBrowser(headless=args.headless, width=args.width, height=args.height) as computer:
        if args.task:
            # Run single task
            result = run_agent_loop(
                computer=computer,
                task=args.task,
                model=args.model,
                start_url=args.url,
            )
            print(f"\nFinal response: {result['final_response']}")
        else:
            # Interactive mode
            print("Anthropic Computer Use Agent (Interactive Mode)")
            print(f"Starting URL: {args.url}")
            print("Type 'quit' to exit\n")

            # Navigate to start URL
            if hasattr(computer, "goto"):
                computer.goto(args.url)
                computer.wait(2000)

            while True:
                try:
                    user_input = input("> ").strip()
                    if user_input.lower() in ("quit", "exit", "q"):
                        break
                    if not user_input:
                        continue

                    result = run_agent_loop(
                        computer=computer,
                        task=user_input,
                        model=args.model,
                    )
                    print(f"\nFinal: {result['final_response']}\n")
                except KeyboardInterrupt:
                    print("\nInterrupted")
                    break


if __name__ == "__main__":
    main()
