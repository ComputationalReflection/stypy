def f(param):
    return '*' + param


def g(param):
    return f(param)

if True:
    g(None)

if True:
    r = g(None)

