l = [1, 2, 3, 4]
l3 = ["False", "1", "string"]

other_l = reduce(lambda x, y: x + str(y), l, 0)  # Unreported. Runtime Crash
r1 = other_l.nothing()  # PyCharm Intellisense shows int type methods, but no error is reported here.

other_l2 = reduce(lambda x, y: x / y, l3, "")
r2 = other_l.nothing()  # Same error.

other_l3 = reduce(lambda x, y: x + y, l, 0)
r3 = other_l[5]  # Nothing is reported. Unchecked reduce return type. Intellisense shows int methods
r4 = other_l.capitalize()  # Nothing is reported

other_l4 = reduce(lambda x, y, z: x + y, l, 0)  # No error report
