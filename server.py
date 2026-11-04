"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311.server")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
READ_ONLY = {"readOnlyHint": True}

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]


HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def _mask_address(address: str) -> str:
    """Cut a street address to its block: "994 HARRISON ST, ..." -> "900 block of HARRISON ST".

    Addresses without a house number, such as intersections, are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address or "")
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """The row a tool may return: every field, with the address masked to the block."""
    row = request.as_dict()
    row["address"] = _mask_address(row["address"])
    return row


def _check_service(service: str | None) -> None:
    if service is not None and service not in SERVICES:
        hint = closest(service, SERVICES)
        suggestion = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise ToolError(f"Unknown service {service!r}.{suggestion} Call dataset_info for the exact service names.")


def _check_neighborhood(neighborhood: str | None) -> None:
    if neighborhood is None or neighborhood in NEIGHBORHOODS:
        return
    hint = closest(neighborhood, NEIGHBORHOODS)
    suggestion = f" Did you mean: {', '.join(hint)}?" if hint else ""
    local = ""
    if neighborhood in LOCAL_NAMES:
        spans = sorted({r.neighborhood for r in REQUESTS if r.local_name == neighborhood and r.neighborhood})
        local = f" {neighborhood!r} is a local name that spans these official neighborhoods: {', '.join(spans)}."
    raise ToolError(
        f"Unknown neighborhood {neighborhood!r}.{local}{suggestion} "
        "Call dataset_info for the 41 official neighborhood names."
    )


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_service(service)
    _check_neighborhood(neighborhood)
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations=READ_ONLY)
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations=READ_ONLY)
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally filtered by service and neighborhood.

    Without group_by, returns a single number. With group_by, returns a dict
    of group to count, largest first, at most 50 groups (ties break
    alphabetically). Use this for any "how many" or "which has the most"
    question; do not page through search_requests to count.

    service and neighborhood must match the exact names from dataset_info.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def median_close_hours(
    service: str | None = None,
    group_by: MedianGroup | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests, rounded to 0.1.

    Without group_by, returns one number (None if nothing has closed). With
    group_by ("neighborhood" or "agency"), returns a dict fastest first,
    leaving out groups with fewer than 10 closed requests. Use this for
    "how fast" or "which is slowest/fastest" questions. Open requests are
    ignored; use count_requests with group_by="status" to see how many are
    still open.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {g: round(median(v), 1) for g, v in hours.items() if len(v) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def open_backlog(service: str | None = None, neighborhood: str | None = None) -> dict:
    """How many 311 requests opened this week are still open, for a service and/or neighborhood.

    Returns open, closed, and total counts, the percent still open, and the
    oldest request that is still open (id, opened time, and block-level
    address). Use this for "how many are still open", "what's the backlog",
    or "what share is unresolved" questions. For open counts broken down
    across many services or neighborhoods, use count_requests with
    group_by="status" per filter; for close times, use median_close_hours.

    service and neighborhood must match the exact names from dataset_info.
    """
    rows = _matching(service, neighborhood)
    still_open = [r for r in rows if r.status == "Open"]
    oldest = min(still_open, key=lambda r: r.opened, default=None)
    return {
        "service": service,
        "neighborhood": neighborhood,
        "open": len(still_open),
        "closed": len(rows) - len(still_open),
        "total": len(rows),
        "percent_open": round(100 * len(still_open) / len(rows), 1) if rows else None,
        "oldest_open": (
            {"request_id": oldest.request_id, "opened": oldest.opened.isoformat(), "address": _mask_address(oldest.address)}
            if oldest
            else None
        ),
    }


@mcp.tool(annotations=READ_ONLY)
def count_in_neighborhood(name: str, ctx: Context, boundary: str | None = None) -> int | str:
    """Count 311 requests for a neighborhood name people use, official or local.

    Accepts the 41 official neighborhoods ("Mission") and local names
    ("Lower Nob Hill", "Duboce Triangle"). Returns a single count. When a
    local name overlaps several official neighborhoods, the user is asked
    which part to count unless `boundary` is given: pass one of the
    overlapped official neighborhoods, or "all" for the whole local area.
    If the user declines, returns a message and counts nothing; call again
    with boundary="all" or a specific part. For counts by service or other
    groupings within official neighborhoods, use count_requests.
    """
    name = name.strip()
    if name in NEIGHBORHOODS:
        return sum(1 for r in REQUESTS if r.neighborhood == name)
    if name not in LOCAL_NAMES:
        hint = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
        suggestion = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise ToolError(f"Unknown neighborhood {name!r}.{suggestion} Call dataset_info for the official names.")

    area = [r for r in REQUESTS if r.local_name == name]
    parts = sorted({r.neighborhood for r in area if r.neighborhood})
    all_option = f"All of {name}"

    if boundary is None and len(parts) > 1:
        choice = read_choice(ctx, "boundary")
        if choice is None:
            return ask_choice(
                "boundary",
                f"{name} overlaps {len(parts)} official neighborhoods. Which part should I count?",
                parts + [all_option],
            )
        if not choice.accepted:
            return (
                f"Nothing was counted: the user did not choose a part of {name}. "
                f"Call again with boundary set to one of {', '.join(parts)}, or 'all'."
            )
        boundary = choice.value

    if boundary is None or boundary.strip().lower() == "all" or boundary == all_option:
        return len(area)
    boundary = boundary.strip()
    if boundary not in parts:
        raise ToolError(
            f"{boundary!r} is not part of {name}. Use one of {', '.join(parts)}, or 'all' for the whole area."
        )
    return sum(1 for r in area if r.neighborhood == boundary)


@mcp.tool(annotations=READ_ONLY)
def local_name_breakdown(name: str) -> dict:
    """Split a local neighborhood name's 311 requests across the official neighborhoods it overlaps.

    Returns the total for the local area and, for each overlapped official
    neighborhood, its count and percent share, largest first. Use this for
    "which part holds the most" or "how is <local name> split" questions.
    For one number, use count_in_neighborhood.
    """
    name = name.strip()
    if name not in LOCAL_NAMES:
        hint = closest(name, LOCAL_NAMES)
        suggestion = f" Did you mean: {', '.join(hint)}?" if hint else ""
        official = " That is an official neighborhood; use count_requests." if name in NEIGHBORHOODS else ""
        raise ToolError(f"{name!r} is not a local neighborhood name.{official}{suggestion}")
    area = [r for r in REQUESTS if r.local_name == name]
    counts = Counter(r.neighborhood for r in area if r.neighborhood)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS]
    return {
        "local_name": name,
        "total": len(area),
        "parts": [
            {"neighborhood": n, "requests": c, "percent": round(100 * c / len(area), 1)} for n, c in ranked
        ],
    }


@mcp.tool(annotations=READ_ONLY)
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests, newest first, up to `limit` rows (1 to 50).

    Addresses are given only to the block ("900 block of HARRISON ST").

    Use this to look at examples or details of specific requests. For counts,
    rankings, or timing, use count_requests or median_close_hours instead:
    this tool returns a sample, not every match. Use get_request when you
    already have a request_id.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. For totals, use count_requests instead.")
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def get_request(request_id: str) -> dict:
    """Return one 311 request by its request_id, with every field.

    The address is given only to the block ("900 block of HARRISON ST").
    Use search_requests to find request IDs.
    """
    log.info("get_request request_id=%s caller=%s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find valid request IDs.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
