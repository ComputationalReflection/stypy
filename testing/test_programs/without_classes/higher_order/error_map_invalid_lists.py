
l = range(5)
l2 = [False, 1, "string"]
other_l2 = map(lambda elem: elem / 2, l2)  # Unreported
r1 = other_l2[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)

l3 = [[], {}, "string"]
other_l3 = map(lambda elem: elem / 2, l3)  # Unreported
r2 = other_l3[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)

l4 = ["False", "1", "string"]
other_l4 = map(lambda elem: elem / 2, l3)  # Unreported
r3 = other_l4[0].capitalize()  # PyCharm assumes that the return type is list[string] (?)


