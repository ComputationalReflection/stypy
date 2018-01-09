

class Foo:
    def __repr__(self):
        return "This is a test"


x = Foo()

y = repr(x)

z = repr(1+6+7)
print y
print z