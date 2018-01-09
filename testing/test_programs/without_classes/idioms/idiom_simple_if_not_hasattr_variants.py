theInt = 3
theStr = "hi"
if True:
    union = 3
else:
    union = "hi"


def simple_if_hasattr_variant1(a):
    b = "hi"
    if not '__div__' in a.__class__.__dict__:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_hasattr_variant2(a):
    b = "hi"
    if not type(a).__dict__.has_key('__div__'):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_hasattr_variant3(a):
    b = "hi"
    if not '__div__' in type(a).__dict__:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_hasattr_variant4(a):
    b = "hi"
    if not '__div__' in dir(type(a)):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


simple_if_hasattr_variant1(union)
simple_if_hasattr_variant2(union)
simple_if_hasattr_variant3(union)
simple_if_hasattr_variant4(union)