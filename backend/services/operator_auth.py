import hmac
import os


def is_authorized_physio_operator(authorization_header: str | None) -> bool:
    configured_token = os.getenv("HAWKEYE_PHYSIO_CONTEXT_TOKEN", "").strip()
    provided_token = (authorization_header or "").removeprefix("Bearer ").strip()
    return bool(
        configured_token
        and hmac.compare_digest(provided_token, configured_token)
    )
