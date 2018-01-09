
# l: list[int]
l = [1, 2, 3, 4]
# x: int; other_l: list[str]
other_l = map((lambda x: str(x)), l)
# l2: list[bool \/ int \/ str]
l2 = [False, 1, 'string']
# x: int; other_l2: list[str]
other_l2 = map((lambda x: str(x)), l)