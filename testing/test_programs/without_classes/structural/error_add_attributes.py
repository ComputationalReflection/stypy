class Foo:
    def __init__(self):
        pass


def func(x):
    return x


def class_func(cls, x):
    return x


f = Foo()
f2 = Foo()

f.a = 3
r1 = f.a
r2 = f2.a  # Reported

if f.a > 0:
    f.att1 = 4
else:
    f.att2 = "hi"

r3 = f.att1
r4 = f.att2  # Not reported

Foo.class_a = "hi"

r5 = f.class_a  # Incorrectly reported. Prints "hi"

