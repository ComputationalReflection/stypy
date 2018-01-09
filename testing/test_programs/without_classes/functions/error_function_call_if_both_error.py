def function_1(x):
    a = 0
    if a > 0:
        x /= 2
    else:
        x -= 2

    return 3

r1 = function_1("a")  # Not Reported on call site
r2 = function_1(range(5))  # Not Reported on call site

def function_2(x):
    a = 0
    if a > 0:
        x /= 2
        return x
    else:
        x -= 2
        return x

r3 = function_2("a")  # Not Reported on call site
r4 = function_2(range(5))  # Not Reported on call site