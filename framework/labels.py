"""Exact charge vocabulary handling shared by preparation, prompting and scoring."""


def normalize_charge(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("A charge must be a string")
    # CAIL uses square brackets *inside official labels*, including at the
    # beginning: "[走私、贩卖、运输、制造]毒品". Preserve them exactly.
    # Do not use fuzzy, substring, or gold-dependent alias conversion.
    value = value.strip()
    if not value:
        raise ValueError("A charge cannot be empty")
    return value


def validate_labels(values: list[str]) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("Labels must be a nonempty JSON array")
    labels = [normalize_charge(value) for value in values]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate charges after normalization")
    return labels
