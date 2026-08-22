from amounts import as_decimal


def format_total(cents: int, currency: str) -> str:
    return f"{currency} {as_decimal(cents)}"
