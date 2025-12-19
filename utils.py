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

    if model in pricing_data:
        data = pricing_data[model]
        # LiteLLM uses cost per token, convert to per 1M tokens
        input_cost = data.get("input_cost_per_token", 0) * 1_000_000
        output_cost = data.get("output_cost_per_token", 0) * 1_000_000
        if input_cost > 0 or output_cost > 0:
            return {"input": input_cost, "output": output_cost}

    # Try with openai/ prefix (LiteLLM convention)
    prefixed_model = f"openai/{model}"
    if prefixed_model in pricing_data:
        data = pricing_data[prefixed_model]
        input_cost = data.get("input_cost_per_token", 0) * 1_000_000
        output_cost = data.get("output_cost_per_token", 0) * 1_000_000
        if input_cost > 0 or output_cost > 0:
            return {"input": input_cost, "output": output_cost}

    raise ValueError(f"Pricing not found for model: {model}")


@dataclass
class UsageTracker:
    """Tracks token usage and costs across API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    model: str = "computer-use-preview"
    call_history: list = field(default_factory=list)

    def add_usage(self, usage: Dict[str, Any], model: Optional[str] = None):
        """Add usage from an API response."""
        if not usage:
            return

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total = usage.get("total_tokens", input_tokens + output_tokens)

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += total
        self.api_calls += 1

        if model:
            self.model = model

        self.call_history.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
            }
        )

    def get_cost(self) -> float:
        """Calculate estimated cost in USD."""
        pricing = get_model_pricing(self.model)
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of usage and costs."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "estimated_cost_usd": round(self.get_cost(), 6),
            "model": self.model,
        }

    def reset(self):
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
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
