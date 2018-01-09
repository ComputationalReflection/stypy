
# Foo: Foo

class Foo:
    # self: instance
    # <Dead code detected>

    def __abs__(self):
        return 4000

    # <Dead code detected>

    def __add__(self, other):
        return 4

# Foo2: Foo2

class Foo2:
    # self: instance
    # __getitem__(x1: slice[]) -> slice[] 

    def __getitem__(self, x1):
        # self: instance
        return x1

# r1: TypeError
r1 = Foo()[1:2]
# x: int
x = 3
# r2: TypeError
r2 = x[1:3]
# x1: slice[]; r3: slice[]
r3 = Foo2()[1:2]