def f(param):
    return '*' + param


def g(param):
    return 3 + param


if True:
    f(None)
else:
    g(None)


if True:
    r = f(None)
else:
    r2 = g(None)