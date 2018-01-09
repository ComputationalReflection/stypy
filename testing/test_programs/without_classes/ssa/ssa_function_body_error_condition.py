def f(param):
    if len('*' + param) > 3:
        return 3
    return None


def g(param):
    accum = 0
    for i in range(len(3 + param)):
        accum += i
    return accum

if True:
    f(None)
else:
    g(None)

if True:
    r = f(None)
else:
    r2 = g(None)