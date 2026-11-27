"""Explicit workflow choices for the optional held-out open-pose check."""

RETURN_OPEN_SKIP_POLICY = "explicit-user-skip-v1"
RETURN_OPEN_OMIT_POLICY = "simplified-workflow-omission-v1"


def is_return_open_skipped(validation) -> bool:
    """A missing or failed check is never an implicit user-requested skip."""
    return (isinstance(validation, dict)
            and validation.get("passed") is False
            and validation.get("status") == "skipped"
            and validation.get("policy") in (RETURN_OPEN_SKIP_POLICY, RETURN_OPEN_OMIT_POLICY)
            and validation.get("used_for_fit") is False)
