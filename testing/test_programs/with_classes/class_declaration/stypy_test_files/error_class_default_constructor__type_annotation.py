
# Foo: Foo

class Foo:
    # self: instance
    # <Dead code detected>

    def method(self, x):
        return x

# f: TypeError
f = Foo(3, 4, 5)