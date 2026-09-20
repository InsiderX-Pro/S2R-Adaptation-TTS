from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProxyProfile:
    name: str
    url: str | None

    @property
    def requests_proxies(self) -> dict[str, str]:
        if not self.url:
            return {}
        return {"http": self.url, "https": self.url}


def _proxy_is_usable(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = str(parsed.hostname or "").lower()
        _ = parsed.port
    except (TypeError, ValueError):
        return False
    allowed_schemes = {"http", "https", "socks4", "socks4a", "socks5", "socks5h"}
    if parsed.scheme.lower() not in allowed_schemes or not host:
        return False
    return True


def discover_proxy_profiles(explicit: str | None = None) -> list[ProxyProfile]:
    """Use only explicitly configured connectivity; never probe private hosts."""
    raw_explicit = str(explicit or "").strip()
    if raw_explicit.lower() == "direct":
        return [ProxyProfile("direct", None)]
    if raw_explicit:
        if not _proxy_is_usable(raw_explicit):
            raise ValueError("The explicitly configured Gemini proxy URL is invalid.")
        return [ProxyProfile("explicit", raw_explicit)]

    return [ProxyProfile("direct", None)]
