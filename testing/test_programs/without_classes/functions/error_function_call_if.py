def functionb(x):
    a = 0
    if a > 0:
        x = x / 2
    return x


r1 = functionb("a")  # Nothing is reported on call site
r2 = functionb(range(5))  # Nothing is reported on call site
