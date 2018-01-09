
# l: list[int]
l = [1, 2, 3, 4]
# l3: list[str]
l3 = ['False', '1', 'string']
# y: int; x: int; other_l: TypeError
other_l = reduce((lambda x, y: (x + str(y))), l, 0)
# r1: TypeError
r1 = other_l.nothing()
# y: str; x: str; other_l2: TypeError
other_l2 = reduce((lambda x, y: (x / y)), l3, '')
# r2: TypeError
r2 = other_l.nothing()
# y: int; x: int; other_l3: int
other_l3 = reduce((lambda x, y: (x + y)), l, 0)
# r3: TypeError
r3 = other_l[5]
# r4: TypeError
r4 = other_l.capitalize()
# y: int; x: int; z: TypeError; other_l4: TypeError
other_l4 = reduce((lambda x, y, z: (x + y)), l, 0)