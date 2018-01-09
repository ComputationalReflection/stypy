
# Foo: Foo

class Foo:
    # __init__() -> None 

    def __init__(self):
        # self: instance
        pass

# func: function
# <Dead code detected>

def func(x):
    return x

# class_func: function
# <Dead code detected>

def class_func(cls, x):
    return x

# f: Foo instance
f = Foo()
# f2: Foo instance
f2 = Foo()
# a: int
f.a = 3
# r1: int
r1 = f.a
# r2: TypeError
r2 = f2.a

if (f.a > 0):
    # att1: int
    f.att1 = 4
else:
    # att2: str
    f.att2 = 'hi'

# r3: int \/ Undefined
r3 = f.att1
# r4: str \/ Undefined
r4 = f.att2
# class_a: str
Foo.class_a = 'hi'
# r5: str
r5 = f.class_a