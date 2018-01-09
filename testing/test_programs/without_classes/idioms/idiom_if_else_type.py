theInt = 3
theStr = "hi"
if True:
    union = 3
else:
    union = "hi"

def if_else_base1(a):
    b = "hi"
    if type(a) is int:
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"
    r5 = a / 3
    r6 = b / 3

def if_else_base2(a):
    b = "hi"
    if type(a) is int:
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"
    r5 = a / 3
    r6 = b / 3

def if_else_base3(a):
    b = "hi"
    if type(a) is int:
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"
    r5 = a / 3
    r6 = b / 3

def if_else_base4(a):
    b = "hi"
    if type(a) is int:
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"
    r5 = a / 3
    r6 = b / 3


bigUnion = int() if True else str() if False else False

if_else_base1(theInt)
if_else_base2(theStr)
if_else_base3(union)
if_else_base4(bigUnion)

def simple_if_else_idiom_variant(a):
    b = "hi"
    if type(a) is type(3):
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"

    r5 = a / 3
    r6 = b / 3

simple_if_else_idiom_variant(union)

def simple_if_else_not_idiom(a):
    b = "hi"
    if type(a) is 3:
        r = a / 3
        r2 = a[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"

    r5 = a / 3
    r6 = b / 3

simple_if_else_not_idiom(union)

class Foo:
    def __init__(self):
        self.attr = 4
        self.strattr = "bar"

def simple_if_else_idiom_attr(a):
    b = "hi"
    if type(a.attr) is int:
        r = a.attr / 3
        r2 = a.attr[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"

    r5 = a.attr / 3
    r6 = b / 3

def simple_if_else_idiom_attr_b(a):
    b = "hi"
    if type(a.strattr) is int:
        r = a.attr / 3
        r2 = a.strattr[0]
        b = 3
    else:
        r3 = a[0]
        r4 = a / 3
        b = "bye"

    r3 = a.strattr / 3
    r4 = b / 3

simple_if_else_idiom_attr(Foo())
simple_if_else_idiom_attr_b(Foo())