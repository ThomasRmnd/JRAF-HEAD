import re


def extract_window(stem: str, unit: str) -> float:
    """
    Extract a window encoded as

        _3m_      -> 3.0
        _1_5m_    -> 1.5
        _12_75m_  -> 12.75

    or

        _2s_      -> 2.0
        _1_2s_    -> 1.2
        _0_75s_   -> 0.75
    """
    pattern = rf"_(\d+(?:_\d+)?)({unit})_"
    match = re.search(pattern, stem)
    if match is None:
        raise ValueError(
            f"Cannot infer {unit} window from {stem!r}."
        )

    return float(match.group(1).replace("_", "."))