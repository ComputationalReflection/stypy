str_l = ["1", "2", "3", "4"]
l = range(5)

other_l = map(lambda x: x, str_l)
r1 = other_l[0] + 6  # Unreported


def f(x):
    return str(x)


other_l2 = map(lambda x: f(x), l)
r2 = other_l2[0] + 6  # Reported


def f2(x):
    if True:
        return "3"
    else:
        return 3


other_l3 = map(lambda x: f2(x), l)
x = other_l3[0] + 6  # Not reported

other_l4 = map(lambda x, y: f(x), l)  # Runtime crash
