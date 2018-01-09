
# Foo: Foo

class Foo:
    # self: instance
    # static(x: str, y: str) -> str 

    @staticmethod
    def static(x, y):
        return (x + y)

    # <Dead code detected>

    def instance(self, x, y):
        return (x + y)

# f: Foo instance
f = Foo()
# r1: TypeError
r1 = Foo.instance(3, 4)
# y: str; x: str; r2: str
r2 = Foo().static('a', 'b')