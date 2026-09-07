"""Presentation helpers for monetary values."""


def format_currency(amount: float, currency: str) -> str:
    """Format a monetary amount with its ISO currency code, e.g. '100.00 USD'."""
    return f"{amount:.2f} {currency}"


def render_chart(points: list[float]) -> str:
    """Render a tiny ASCII sparkline (unrelated to transactions — a retrieval distractor)."""
    return "".join(chr(0x2581 + min(7, int(p))) for p in points)
