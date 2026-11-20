"""Route IRIS's LiteLLM calls to the local Claude Code CLI.

IRIS normally reaches a hosted provider (Gemini by default) through
``litellm.completion``.  This shim keeps that exact interface but answers each
completion by running the Claude Code CLI in headless print mode, so IRIS runs
on an existing Claude Code installation instead of a provider API key.

Install the patch *before* the first completion call::

    from src.models.claude_code_llm import install
    install()

Environment variables:
    CLAUDE_CODE_BIN     path to the ``claude`` executable (auto-detected)
    IRIS_CLAUDE_MODEL   model alias passed to ``--model`` (default: sonnet)
    IRIS_CLAUDE_TIMEOUT per-call timeout in seconds (default: 600)
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Optional

import litellm
from litellm import ModelResponse
from litellm.types.utils import Choices, Message, Usage

_PATCHED = False
_LOCK = threading.Lock()

# Serialise CLI calls: each one spawns a process, and IRIS's MCTS expansion can
# fan out.  A modest cap keeps memory and rate limits sane.
_SEMAPHORE = threading.Semaphore(int(os.environ.get("IRIS_CLAUDE_CONCURRENCY", "3")))

_CALL_LOG: List[Dict[str, Any]] = []


def _find_cli() -> str:
    """Locate the Claude Code executable."""
    explicit = os.environ.get("CLAUDE_CODE_BIN")
    if explicit and os.path.exists(explicit):
        return explicit

    on_path = shutil.which("claude")
    if on_path:
        return on_path

    # Desktop app keeps versioned copies under ~/.config/Claude/claude-code/
    pattern = os.path.expanduser("~/.config/Claude/claude-code/*/*/claude")
    candidates = sorted(glob.glob(pattern))
    if candidates:
        return candidates[-1]

    raise RuntimeError(
        "Could not find the Claude Code CLI. Set CLAUDE_CODE_BIN to its path."
    )


def _split_messages(messages: List[Dict[str, str]]) -> tuple[Optional[str], str]:
    """Split a chat transcript into (system prompt, user prompt)."""
    system_parts: List[str] = []
    convo_parts: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):  # multimodal blocks -> text only
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"[Previous assistant turn]\n{content}")
        else:
            convo_parts.append(content)

    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, "\n\n".join(convo_parts)


def _run_cli(system_prompt: Optional[str], user_prompt: str, model: str) -> Dict[str, Any]:
    """Run one headless Claude Code call and return its parsed JSON result."""
    cli = _find_cli()
    timeout = int(os.environ.get("IRIS_CLAUDE_TIMEOUT", "600"))

    cmd = [
        cli,
        "--print",
        "--output-format", "json",
        "--model", model,
        # Pure text generation: the shim stands in for a chat completion API,
        # so the sub-session gets no tools and no file access.
        "--allowed-tools", "",
        "--strict-mcp-config",
    ]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    # Run in an empty directory so project files and CLAUDE.md cannot leak into
    # the ideation prompts.
    with tempfile.TemporaryDirectory(prefix="iris-llm-") as workdir:
        with _SEMAPHORE:
            proc = subprocess.run(
                cmd,
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Claude Code CLI exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:800]}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse CLI output as JSON: {proc.stdout[:500]!r}"
        ) from exc


def claude_code_completion(
    messages: List[Dict[str, str]],
    model: str = "claude-code/sonnet",
    **kwargs: Any,
) -> ModelResponse:
    """LiteLLM-compatible completion backed by the Claude Code CLI.

    Sampling kwargs (temperature, top_p, max_tokens, ...) are accepted and
    ignored: the CLI does not expose them.
    """
    cli_model = os.environ.get("IRIS_CLAUDE_MODEL", "sonnet")

    system_prompt, user_prompt = _split_messages(messages)
    payload = _run_cli(system_prompt, user_prompt, cli_model)

    text = payload.get("result") or ""
    if payload.get("is_error"):
        raise RuntimeError(f"Claude Code reported an error: {text[:500]}")
    if not text.strip():
        raise RuntimeError("Claude Code returned an empty completion")

    usage_in = payload.get("usage", {}) or {}
    prompt_tokens = int(usage_in.get("input_tokens", 0) or 0)
    completion_tokens = int(usage_in.get("output_tokens", 0) or 0)

    _CALL_LOG.append(
        {
            "requested_model": model,
            "cli_model": cli_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "duration_ms": payload.get("duration_ms"),
        }
    )

    return ModelResponse(
        id=f"claudecode-{uuid.uuid4().hex[:12]}",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=text.strip()),
            )
        ],
        model=f"claude-code/{cli_model}",
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def claude_code_batch_completion(
    messages: List[List[Dict[str, str]]],
    model: str = "claude-code/sonnet",
    **kwargs: Any,
) -> List[ModelResponse]:
    """Sequential stand-in for ``litellm.batch_completion``."""
    kwargs.pop("fallbacks", None)
    return [claude_code_completion(m, model=model, **kwargs) for m in messages]


def get_call_log() -> List[Dict[str, Any]]:
    """Return a copy of the per-call usage log."""
    return list(_CALL_LOG)


def install() -> str:
    """Patch litellm so every IRIS completion goes through Claude Code."""
    global _PATCHED
    with _LOCK:
        if _PATCHED:
            return _find_cli()
        litellm.completion = claude_code_completion
        litellm.batch_completion = claude_code_batch_completion
        litellm.drop_params = True
        _PATCHED = True
        cli = _find_cli()
        print(
            f"[claude_code_llm] litellm.completion -> Claude Code CLI\n"
            f"                  binary: {cli}\n"
            f"                  model:  {os.environ.get('IRIS_CLAUDE_MODEL', 'sonnet')}"
        )
        return cli
