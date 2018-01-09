
# C: C

class C:
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

    # method() -> None 

    def method(self):
        # self: instance; r: str
        self.r = 'str'

# c: C instance
c = C()
c.method()
# x: bool
x = (c.r == 5)
# Counter: Counter

class Counter:
    count = 0
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

    # inc(value: int) -> int /\ inc(value: float) -> float /\ inc(value: int) -> float 

    def inc(self, value):
        # count: int \/ float; self: instance
        self.count += value
        return self.count

# obj: Counter instance
obj = Counter()
# sum: float; value: int \/ float
sum = (obj.inc(1) + obj.inc(0.2))

if obj:
    # resul: float; value: int
    resul = obj.inc(1)
else:
    # resul: float; value: float
    resul = obj.inc(0.5)
