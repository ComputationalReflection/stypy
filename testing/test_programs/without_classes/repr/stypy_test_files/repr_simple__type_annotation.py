
# Foo: Foo

class Foo:
    # self: instance
    # __repr__() -> str 

    def __repr__(self):
        # self: instance
        return 'This is a test'

# x: Foo instance
x = Foo()
# y: str
y = repr(x)
# z: str
z = repr(((1 + 6) + 7))
print y
print z