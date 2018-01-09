
# l: list[int]
l = range(5)
# l2: list[bool \/ int \/ str]
l2 = [False, 1, 'string']
# other_l2: TypeError; elem: bool \/ int \/ str
other_l2 = map((lambda elem: (elem / 2)), l2)
# r1: TypeError
r1 = other_l2[0].capitalize()
# l3: list[list[] \/ dict{} \/ str]
l3 = [[], {}, 'string']
# other_l3: TypeError; elem: list[] \/ dict{} \/ str
other_l3 = map((lambda elem: (elem / 2)), l3)
# r2: TypeError
r2 = other_l3[0].capitalize()
# l4: list[str]
l4 = ['False', '1', 'string']
# elem: list[] \/ dict{} \/ str; other_l4: TypeError
other_l4 = map((lambda elem: (elem / 2)), l3)
# r3: TypeError
r3 = other_l4[0].capitalize()