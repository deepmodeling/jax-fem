from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Union

@dataclass
class A:
    a: Any = 1.
    b: Any = 2.
    def __post_init__(self):
        print(self.a)
        print(self.b)

class B(A):
    def f(self):
        print(self.b)


b = B(4, 5)
b.f()