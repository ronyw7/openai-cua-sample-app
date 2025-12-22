import base64
import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)


# Fetch pricing from LiteLLM
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CACHE_DIR = Path.home() / ".cache" / "openai-cua"
CACHE_DURATION = timedelta(days=1)

# Cache for pricing data
_pricing_cache: Optional[Dict[str, Any]] = None


def _get_cache_file() -> Path:
    """Get the cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "model_pricing.json"


def _is_cache_valid(cache_file: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_file.exists():
        return False
    try:
        with open(cache_file) as f:
            data = json.load(f)
        cached_time = datetime.fromisoformat(data.get("_cached_at", "1970-01-01"))
        return datetime.now() - cached_time < CACHE_DURATION
    except Exception:
        return False


def _fetch_pricing_data() -> Dict[str, Any]:
    """Fetch pricing data from LiteLLM GitHub, with caching."""
    global _pricing_cache

    if _pricing_cache is not None:
        return _pricing_cache

    cache_file = _get_cache_file()

    # Try to load from cache first
    if _is_cache_valid(cache_file):
        try:
            with open(cache_file) as f:
                data = json.load(f)
            _pricing_cache = data.get("pricing", {})
            return _pricing_cache
        except Exception:
            pass

    # Fetch from GitHub
    try:
        response = requests.get(LITELLM_PRICING_URL, timeout=10)
        response.raise_for_status()
        _pricing_cache = response.json()

        # Save to cache with timestamp
        cache_data = {"_cached_at": datetime.now().isoformat(), "pricing": _pricing_cache}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        return _pricing_cache
    except Exception as e:
        print(f"Warning: Failed to fetch pricing data: {e}. Using fallback pricing.")
        _pricing_cache = {}
        return _pricing_cache


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing for a model (input/output cost per 1M tokens).

    Fetches from LiteLLM's pricing JSON and converts per-token costs to per-1M-token costs.
    Falls back to hardcoded pricing if fetch fails or model not found.
    """
    pricing_data = _fetch_pricing_data()

    # Try exact match first
    if model in pricing_data:
        data = pricing_data[model]
        input_cost = data.get("input_cost_per_token", 0) * 1_000_000
        output_cost = data.get("output_cost_per_token", 0) * 1_000_000
        if input_cost > 0 or output_cost > 0:
            return {"input": input_cost, "output": output_cost}

    # Try with provider prefixes (LiteLLM convention)
    for prefix in ["openai/", "anthropic/"]:
        prefixed_model = f"{prefix}{model}"
        if prefixed_model in pricing_data:
            data = pricing_data[prefixed_model]
            input_cost = data.get("input_cost_per_token", 0) * 1_000_000
            output_cost = data.get("output_cost_per_token", 0) * 1_000_000
            if input_cost > 0 or output_cost > 0:
                return {"input": input_cost, "output": output_cost}

    # Fallback pricing for known models
    fallback_pricing = {
        # Anthropic models
        "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
        # OpenAI models
        "computer-use-preview": {"input": 3.0, "output": 12.0},
    }

    if model in fallback_pricing:
        return fallback_pricing[model]

    raise ValueError(f"Pricing not found for model: {model}")


@dataclass
class UsageTracker:
    """Tracks token usage and costs across API calls.

    Handles both OpenAI and Anthropic formats:
    - OpenAI Chat API: prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens
    - OpenAI Responses API: input_tokens, output_tokens
    - Anthropic: input_tokens, output_tokens

    Cached tokens (OpenAI) are 50% cheaper than regular input tokens.

    Note: E2E latency should be tracked separately (total_time_seconds in results).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # OpenAI cached tokens (50% cheaper)
    total_tokens: int = 0
    api_calls: int = 0
    model: str = "computer-use-preview"
    call_history: list = field(default_factory=list)

    def add_usage(self, usage: Any, model: Optional[str] = None):
        """Add usage from an API response (works with both dict and object)."""
        if not usage:
            return

        # Extract tokens based on format (dict vs object, OpenAI vs Anthropic)
        if isinstance(usage, dict):
            # OpenAI Chat API uses prompt_tokens/completion_tokens
            # OpenAI Responses API uses input_tokens/output_tokens
            input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", 0)

            # Extract cached tokens from prompt_tokens_details (OpenAI)
            prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            if isinstance(prompt_details, dict):
                cached_tokens = prompt_details.get("cached_tokens", 0)
            else:
                cached_tokens = 0
        else:
            # Object format (Anthropic, or parsed OpenAI)
            input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0)

            # Try to get cached tokens from details
            prompt_details = getattr(usage, "prompt_tokens_details", None) or getattr(
                usage, "input_tokens_details", None
            )
            if prompt_details is not None:
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
            else:
                cached_tokens = 0

        total = input_tokens + output_tokens

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        self.total_tokens += total
        self.api_calls += 1

        if model:
            self.model = model

        call_record = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        }
        if cached_tokens > 0:
            call_record["cached_tokens"] = cached_tokens

        self.call_history.append(call_record)

    def get_cost(self) -> float:
        """Calculate estimated cost in USD.

        For OpenAI, cached tokens are 50% cheaper than regular input tokens.
        Cost = (uncached_input * input_price) + (cached * input_price * 0.5) + (output * output_price)
        """
        try:
            pricing = get_model_pricing(self.model)
        except ValueError:
            # Fallback if pricing not found
            pricing = {"input": 3.0, "output": 15.0}

        # Cached tokens are included in input_tokens, so subtract for uncached
        uncached_input = self.input_tokens - self.cached_tokens

        # Uncached input at full price, cached at 50%
        input_cost = (uncached_input / 1_000_000) * pricing["input"]
        cached_cost = (self.cached_tokens / 1_000_000) * pricing["input"] * 0.5
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]

        return input_cost + cached_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of usage and costs."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,  # Always show for transparency
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "estimated_cost_usd": round(self.get_cost(), 6),
            "model": self.model,
        }

    def reset(self):
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self.call_history.clear()


# Global usage tracker instance
usage_tracker = UsageTracker()

BLOCKED_DOMAINS = [
    "maliciousbook.com",
    "evilvideos.com",
    "darkwebforum.com",
    "shadytok.com",
    "suspiciouspins.com",
    "ilanbigio.com",
]


def pp(obj):
    print(json.dumps(obj, indent=4))


def show_image(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(BytesIO(image_data))
    image.show()


def calculate_image_dimensions(base_64_image):
    image_data = base64.b64decode(base_64_image)
    image = Image.open(io.BytesIO(image_data))
    return image.size


def sanitize_message(msg: dict) -> dict:
    """Return a copy of the message with image_url omitted for computer_call_output messages."""
    if msg.get("type") == "computer_call_output":
        output = msg.get("output", {})
        if isinstance(output, dict):
            sanitized = msg.copy()
            sanitized["output"] = {**output, "image_url": "[omitted]"}
            return sanitized
    return msg


def create_response(**kwargs):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    openai_org = os.getenv("OPENAI_ORG")
    if openai_org:
        headers["Openai-Organization"] = openai_org

    response = requests.post(url, headers=headers, json=kwargs)

    if response.status_code != 200:
        print(f"Error: {response.status_code} {response.text}")

    result = response.json()

    # Track usage if present in response
    if "usage" in result:
        usage_tracker.add_usage(result["usage"], model=kwargs.get("model"))

    return result


def check_blocklisted_url(url: str) -> None:
    """Raise ValueError if the given URL (including subdomains) is in the blocklist."""
    hostname = urlparse(url).hostname or ""
    if any(hostname == blocked or hostname.endswith(f".{blocked}") for blocked in BLOCKED_DOMAINS):
        raise ValueError(f"Blocked URL: {url}")


# ============================================================================
# Shared Evaluation and Logging Functions
# ============================================================================

import sys
import time
from urllib.parse import urljoin

import yaml

# Try to import agisdk for evaluation
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

    return {task["id"]: task for task in tasks_list if isinstance(task, dict) and "id" in task}


def get_task_config(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task configuration from registry by ID."""
    registry = load_task_registry()
    return registry.get(task_id)


def create_evaluator(task_id: str, version: str = "custom"):
    """Create an evaluator for the task, using agisdk if available."""
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
    items: list,
    output_dir: Path,
    run_num: int,
    timestamp_str: str,
    save_screenshots: bool = True,
    screenshot_key: str = "image_url",  # "image_url" for OpenAI, "screenshot_b64" for Anthropic
) -> Path:
    """
    Save the agent conversation log, optionally extracting screenshots to separate files.

    Works with both OpenAI and Anthropic log formats.
    """
    screenshots_dir = output_dir / f"run{run_num}_{timestamp_str}_screenshots"
    screenshot_count = 0

    processed_items = []
    for idx, item in enumerate(items):
        processed_item = item.copy() if isinstance(item, dict) else item

        # Handle OpenAI format: computer_call_output with base64 screenshots
        if isinstance(item, dict) and item.get("type") == "computer_call_output":
            output = item.get("output", {})
            if isinstance(output, dict) and output.get("type") == "input_image":
                image_url = output.get("image_url", "")
                if image_url.startswith("data:image/png;base64,"):
                    screenshot_count += 1
                    if save_screenshots:
                        screenshots_dir.mkdir(parents=True, exist_ok=True)
                        base64_data = image_url.replace("data:image/png;base64,", "")
                        screenshot_path = screenshots_dir / f"screenshot_{screenshot_count:03d}.png"
                        with open(screenshot_path, "wb") as f:
                            f.write(base64.b64decode(base64_data))
                        processed_item = item.copy()
                        processed_item["output"] = output.copy()
                        processed_item["output"]["image_url"] = f"file://{screenshot_path}"
                    else:
                        processed_item = item.copy()
                        processed_item["output"] = output.copy()
                        processed_item["output"]["image_url"] = "[base64 screenshot truncated]"

        # Handle Anthropic format: tool_use with screenshot_b64
        elif isinstance(item, dict) and item.get("type") == "tool_use" and item.get("has_screenshot"):
            screenshot_count += 1
            if save_screenshots and "screenshot_b64" in item:
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshots_dir / f"screenshot_{screenshot_count:03d}.png"
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(item["screenshot_b64"]))
                processed_item = item.copy()
                processed_item["screenshot_path"] = str(screenshot_path)
                if "screenshot_b64" in processed_item:
                    del processed_item["screenshot_b64"]
            elif "screenshot_b64" in item:
                processed_item = item.copy()
                del processed_item["screenshot_b64"]
                processed_item["screenshot"] = "[base64 truncated]"

        processed_items.append(processed_item)

    log_path = output_dir / f"run{run_num}_{timestamp_str}_log.json"
    with open(log_path, "w") as f:
        json.dump(processed_items, f, indent=2, default=str)

    print(f"Agent log saved to: {log_path}")
    if save_screenshots and screenshot_count > 0:
        print(f"Screenshots saved to: {screenshots_dir}/ ({screenshot_count} files)")

    return log_path


def make_safety_check_callback(autonomous: bool):
    """Create a safety check callback that auto-acknowledges in autonomous mode."""

    def acknowledge_safety_check_callback(message: str) -> bool:
        if autonomous:
            print(f"Safety Check Warning (auto-acknowledged): {message}")
            return True
        response = input(f"Safety Check Warning: {message}\nDo you want to acknowledge and proceed? (y/n): ").lower()
        return response.lower().strip() == "y"

    return acknowledge_safety_check_callback
