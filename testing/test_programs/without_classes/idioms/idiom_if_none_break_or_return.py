
if True:
    x = None
else:
    x = 3


def f():
    if x is None:
        raise Exception("x is None")
    else:
        pass

    return x / 3

r = f()