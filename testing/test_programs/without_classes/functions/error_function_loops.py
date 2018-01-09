def functionb(x):
    for i in range(5):
        x /= 2

    return x
r1 = functionb("a")  # Not Reported on call site
r2 = functionb(range(5))  # Not Reported on call site

