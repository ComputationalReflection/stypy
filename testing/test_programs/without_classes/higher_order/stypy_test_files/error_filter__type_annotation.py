
# l2: list[bool \/ int \/ str]
l2 = [False, 1, 'string']
# f2: function
# f2(x: bool \/ int \/ str) -> str /\ f2(x: str) -> str 

def f2(x):
    return str(x)

# x: bool \/ int \/ str; other_l: list[bool \/ int \/ str \/ NoneType]
other_l = filter((lambda x: f2(x)), l2)
# r1: int
r1 = (other_l[2] + 6)
# l3: list[str]
l3 = ['False', '1', 'string']
# x: str; other_l2: list[str \/ NoneType]
other_l2 = filter((lambda x: f2(x)), l3)
# r2: TypeError
r2 = (other_l2[2] + 6)
# y: TypeError; x: bool \/ int \/ str; other_l3: TypeError
other_l3 = filter((lambda x, y: f2(x)), l2)