"""Reference answers for 16 - typing, dataclasses and protocols."""

from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


def safe_divide(a: float, b: float) -> Optional[float]:
    """Divide a by b, or return None instead of raising on b == 0.

    The return type is `Optional[float]` - `float | None` - because failure
    is modelled as a value the caller must check, not an exception they must
    remember to catch. This is the same shape as dict.get's default of None.
    """
    if b == 0:
        return None
    return a / b


class Stack(Generic[T]):
    """A LIFO stack, generic over the element type.

    `Generic[T]` is what lets a type checker infer `Stack[int]` from
    `Stack()` used with ints, and flag `stack.push("x")` on a `Stack[int]`
    as an error - all at zero runtime cost, since Python itself never
    checks T.
    """

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)


def first_match(items: Iterable[T], predicate: Callable[[T], bool]) -> Optional[T]:
    """Return the first item for which predicate(item) is True, else None.

    Generic over T via the TypeVar bound in both the Iterable and the
    Callable - a type checker ties the element type of `items` to the
    argument type `predicate` expects, so passing a `list[int]` with a
    `Callable[[str], bool]` predicate is flagged before you run anything.
    """
    for item in items:
        if predicate(item):
            return item
    return None


@dataclass(frozen=True)
class Point:
    """An immutable 2D point.

    frozen=True makes __setattr__ raise, which is what lets a Point be
    hashable (dataclass only generates __hash__ when frozen=True or
    eq=False) and safe to use as a dict key or put in a set.
    """

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


@dataclass(order=True)
class SemVer:
    """A semantic version, orderable by (major, minor, patch) only.

    `field(compare=False)` excludes `prerelease` from both __eq__ and the
    ordering methods order=True generates - so 1.2.3-rc1 and 1.2.3 compare
    equal and sort adjacently, which matches how semver defines precedence
    for the release core (real semver also orders prereleases below their
    release, which this simplified version does not model).
    """

    major: int
    minor: int
    patch: int
    prerelease: str = field(default="", compare=False)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        return f"{base}-{self.prerelease}" if self.prerelease else base


def sort_versions(versions: list[SemVer]) -> list[SemVer]:
    """Sort SemVer instances using the dataclass-generated ordering."""
    return sorted(versions)


@dataclass
class ShoppingCart:
    """A cart whose item list defaults to empty, per-instance.

    `items: list = []` as a default would share ONE list across every
    instance that doesn't pass its own - the classic mutable-default bug.
    `field(default_factory=list)` calls list() fresh for each instance
    instead, which is why dataclasses forbids a bare mutable literal
    default and raises ValueError at class-definition time if you try.
    """

    items: list = field(default_factory=list)

    def add_item(self, name: str) -> None:
        self.items.append(name)

    def total_items(self) -> int:
        return len(self.items)


@dataclass
class PositiveInt:
    """A dataclass that validates its own field after construction.

    __post_init__ runs once, right after the generated __init__ finishes
    assigning fields - the standard place to add validation a plain
    dataclass field declaration cannot express.
    """

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError(f"value must be positive, got {self.value}")


@runtime_checkable
class SupportsArea(Protocol):
    """Structural type: anything with a no-arg `area() -> float` method.

    Nothing has to inherit from SupportsArea to satisfy it - Circle and
    Rectangle below are plain classes. @runtime_checkable is what makes
    isinstance(obj, SupportsArea) actually work; without it isinstance
    raises TypeError, since Protocol is a static-only tool by default.
    """

    def area(self) -> float:
        ...


@dataclass
class Circle:
    radius: float

    def area(self) -> float:
        return 3.141592653589793 * self.radius ** 2


@dataclass
class Rectangle:
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height


def total_area(shapes: Iterable[SupportsArea]) -> float:
    """Sum area() across any mix of objects that structurally satisfy it.

    Circle and Rectangle share no base class - this function only ever
    calls .area(), so it does not need one. That is what "structural"
    means: the Protocol describes a shape, not a lineage.
    """
    return sum(shape.area() for shape in shapes)


def satisfies_supports_area(obj: object) -> bool:
    """True if obj structurally satisfies SupportsArea, via isinstance.

    Works only because SupportsArea is decorated @runtime_checkable -
    isinstance checks for the *presence* of an `area` attribute, not that
    it is callable with the right signature or returns a float.
    """
    return isinstance(obj, SupportsArea)
