"""Safety controls for the RAG chain.

Was an empty placeholder package until 2026-08-16. See prompt_guard for the
threat it closes and, importantly, for what it deliberately does not claim.
"""

from .prompt_guard import (
    CONTEXT_FENCE_CLOSE,
    CONTEXT_FENCE_OPEN,
    Finding,
    GuardResult,
    PromptGuardError,
    Severity,
    inspect_output,
    inspect_user_input,
    sanitize_context,
)

__all__ = [
    "CONTEXT_FENCE_CLOSE",
    "CONTEXT_FENCE_OPEN",
    "Finding",
    "GuardResult",
    "PromptGuardError",
    "Severity",
    "inspect_output",
    "inspect_user_input",
    "sanitize_context",
]
