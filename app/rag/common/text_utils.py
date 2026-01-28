from typing import Iterable, List, TypeVar

T = TypeVar("T")


def unique_in_order(items: Iterable[T]) -> List[T]:
    seen: set[T] = set()
    out: List[T] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
