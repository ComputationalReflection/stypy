theInt = 3
theStr = "hi"
if True:
    union = 3
else:
    union = "hi"


def simple_if_variant1(a):
    b = "hi"

    if type(a) == int:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant2(a):
    b = "hi"

    if a.__class__ is int:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant3(a):
    b = "hi"
    if a.__class__ == int:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant4(a):
    b = "hi"
    if int is type(a):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant5(a):
    b = "hi"
    if int == type(a):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant6(a):
    b = "hi"
    if int is a.__class__:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_variant7(a):
    b = "hi"
    if int == a.__class__:
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


simple_if_variant1(union)
simple_if_variant2(union)
simple_if_variant3(union)
simple_if_variant4(union)
simple_if_variant5(union)
simple_if_variant6(union)
simple_if_variant7(union)
