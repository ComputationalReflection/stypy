
# comparations: function
# comparations() -> None 

def comparations():
    # a: int
    a = 3
    # b: int
    b = 4
    # c: int
    c = 8
    # Foo: Foo

    class Foo:
        # <Dead code detected>

        def __cmp__(self, other):
            return range(5)

    # c0: TypeError
    c0 = (a < Foo())
    # c1: TypeError
    c1 = (a < b < Foo())

comparations()