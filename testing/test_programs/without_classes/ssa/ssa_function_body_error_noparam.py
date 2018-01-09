def f():
    arr = [None] * 4
    return arr[1]


def g():
    arr = [None] * 4
    return arr[1] / 2


if True:
    f()
else:
    g()

if True:
    r = f()
else:
    r2 = g()
