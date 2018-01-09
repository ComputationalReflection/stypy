
# FooParent: FooParent

class FooParent:
    # self: instance
    # method() -> int \/ list[] \/ str \/ bool 

    def method(self):
        # self: instance

        if True:
            return 3
        else:
            return list()


# FooChild: FooChild

class FooChild(FooParent, ):
    # self: instance
    # method() -> str \/ bool 

    def method(self):
        # self: instance

        if True:
            return 'a'
        else:
            return True



if True:
    # o: FooParent instance
    o = FooParent()
else:
    # o: FooChild instance
    o = FooChild()

# x: int \/ list[] \/ str \/ bool
x = o.method()
# r1: TypeError
r1 = x.nothing()
# l: int
l = len(x)

if True:
    # o2: FooParent instance
    o2 = FooParent()
    # r2: int \/ list[]
    r2 = o2.method()
else:
    # o2: FooChild instance
    o2 = FooChild()
    # r2: str \/ bool
    r2 = o2.method()

# r3: TypeError
r3 = r2.nothing()
# l2: int
l2 = len(x)