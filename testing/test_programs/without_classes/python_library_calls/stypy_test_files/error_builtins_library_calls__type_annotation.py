
# math: module
import math

# Foo: Foo

class Foo:
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

# x: int; r: int
r = sum(((x * x) for x in range(10)))
# err: TypeError
err = r.nothing()
# r2: int
r2 = abs(3)
# r3: TypeError
r3 = abs('3')
# r4: TypeError
r4 = abs(2, 3)
r4.nothing()
# r5: int
r5 = abs(3)
# err2: TypeError
err2 = r5.nothing()
# r6: TypeError
r6 = all(3)
# r7: bool
r7 = all([3])
# err3: TypeError
err3 = r7.nothing()
# r8: TypeError
r8 = bytearray(Foo())
# err4: TypeError
err4 = r8.nothing()
# words: list[str]
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
# x: int; S: list[int]
S = [(x ** 2) for x in range(10)]
# err5: TypeError
err5 = math.fsum(words)
# r9: float
r9 = math.fsum(S)
# err6: TypeError
err6 = r9.nothing()