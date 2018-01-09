
# Foo: Foo

class Foo:
    # self: instance
    # static(x: int, y: int) -> int 

    @staticmethod
    def static(x, y):
        return (x + y)

    # instance(x: str, y: str) -> str 

    def instance(self, x, y):
        # self: instance
        return (x + y)

# f: Foo instance
f = Foo()
# y: int; x: int; r1: int
r1 = Foo.static(3, 4)
# y: str; x: str; r2: str
r2 = Foo().instance('a', 'b')