"""Small network-address helpers independent of Flower's private modules."""

from __future__ import annotations


def parse_address(address: str) -> tuple[str, int, bool] | None:
    """Parse ``host:port`` or ``[ipv6]:port`` into Flower's legacy shape."""

    value = str(address).strip()
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0 or value[closing + 1 : closing + 2] != ":":
            return None
        host = value[1:closing]
        raw_port = value[closing + 2 :]
        is_v6 = True
    else:
        host, separator, raw_port = value.rpartition(":")
        if not separator or not host:
            return None
        is_v6 = ":" in host
    try:
        port = int(raw_port)
    except ValueError:
        return None
    if not 0 <= port <= 65535:
        return None
    return host, port, is_v6
