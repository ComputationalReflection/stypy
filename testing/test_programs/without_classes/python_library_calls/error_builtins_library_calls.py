import math


class Foo:
    def __init__(self):
        pass


r = sum(x * x for x in range(10))

err = r.nothing()  # Unreported

r2 = abs(3)
r3 = abs("3")  # Unreported: Parameter types are not checked
r4 = abs(2, 3)  # Reported: Arities are checked

r4.nothing()  # Unreported

r5 = abs(3)
err2 = r5.nothing()  # Unreported

r6 = all(3)  # Reported
r7 = all([3])
err3 = r7.nothing()  # Reported

r8 = bytearray(Foo())  # Unreported
err4 = r8.nothing()  # Reported

words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
S = [x ** 2 for x in range(10)]

err5 = math.fsum(words)  # Reported
r9 = math.fsum(S)
err6 = r9.nothing()  # Reported
