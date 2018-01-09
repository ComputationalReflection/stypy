l2 = [False, 1, "string"]


def f2(x):
    return str(x)


other_l = filter(lambda x: f2(x), l2)

r1 = other_l[2] + 6  # Not reported

l3 = ["False", "1", "string"]
other_l2 = filter(lambda x: f2(x), l3)
r2 = other_l2[2] + 6  # Reported

other_l3 = filter(lambda x, y: f2(x), l2)  # Not reported
