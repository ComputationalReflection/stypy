theInt = 3
theStr = "hi"
if True:
    union = 3
else:
    union = "hi"


def simple_if_isinstance_base1(a):
    b = "hi"
    if isinstance(a, int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_isinstance_base2(a):
    b = "hi"
    if isinstance(a, int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_isinstance_base3(a):
    b = "hi"
    if isinstance(a, int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def sum(a, b):
    return a + b


def concat(a, b):
    return str(a) + str(b)


def simple_if_isinstance_call_int(a):
    b = "hi"
    if isinstance(sum(a, a), int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


def simple_if_isinstance_call_str(a):
    b = "hi"
    if not isinstance(concat(a, a), int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


simple_if_isinstance_base1(theInt)
simple_if_isinstance_base2(theStr)
simple_if_isinstance_base3(union)

simple_if_isinstance_call_int(theInt)
simple_if_isinstance_call_str(union)


def simple_if_not_isinstance_idiom(a):
    b = "hi"
    if not isinstance(a, int):
        r = a / 3
        r2 = a[0]
        b = 3
    r3 = a / 3
    r4 = b / 3


simple_if_not_isinstance_idiom(union)


#
class Foo:
    def __init__(self):
        self.attr = 4
        self.strattr = "bar"


def simple_if_isinstance_idiom_attr(a):
    b = "hi"
    if isinstance(a.attr, int):
        r = a.attr / 3
        r2 = a.attr[0]
        b = 3
    r3 = a.attr / 3
    r4 = b / 3


def simple_if_isinstance_idiom_attr_b(a):
    b = "hi"
    if not isinstance(a.strattr, str):
        r = a.attr / 3
        r2 = a.strattr[0]
        b = 3
    r3 = a.strattr / 3
    r4 = b / 3


simple_if_isinstance_idiom_attr(Foo())
simple_if_isinstance_idiom_attr_b(Foo())
