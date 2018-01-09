class FooParent:
    def method(self):
        if True:
            return 3
        else:
            return list()


class FooChild(FooParent):
    def method(self):
        if True:
            return "a"
        else:
            return True


if True:
    o = FooParent()
else:
    o = FooChild()

x = o.method()
r1 = x.nothing()  # Detected (x is int | str)
l = len(x)  # Unreported (optimistic)

if True:
    o2 = FooParent()
    r2 = o2.method()
else:
    o2 = FooChild()
    r2 = o2.method()

r3 = r2.nothing()  # Detected (x is int | str)
l2 = len(x)  # Unreported (optimistic)
