from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import anyio

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeURL(ValueError):
    """Raised when a URL is malformed or resolves to a blocked address range."""


def _is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURL(f"url must be http or https, got '{parsed.scheme or url}'")
    host = parsed.hostname
    if not host:
        raise UnsafeURL("url must include a host")
    try:
        infos = await anyio.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"host '{host}' could not be resolved") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _is_blocked(ip):
            raise UnsafeURL(
                f"host '{host}' resolves to a disallowed "
                "(private/loopback/link-local) address"
            )
