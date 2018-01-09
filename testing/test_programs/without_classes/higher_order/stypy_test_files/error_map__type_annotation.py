
# str_l: list[str]
str_l = ['1', '2', '3', '4']
# l: list[int]
l = range(5)
# x: str; other_l: list[str]
other_l = map((lambda x: x), str_l)
# r1: TypeError
r1 = (other_l[0] + 6)
# f: function
# f(x: int) -> str 

def f(x):
    return str(x)

# x: int; other_l2: list[str]
other_l2 = map((lambda x: f(x)), l)
# r2: TypeError
r2 = (other_l2[0] + 6)
# f2: function
# f2(x: int) -> str \/ int 

def f2(x):

    if True:
        return '3'
    else:
        return 3


# x: int; other_l3: list[str \/ int]
other_l3 = map((lambda x: f2(x)), l)
# x: int
x = (other_l3[0] + 6)
# y: TypeError; x: int; other_l4: TypeError
other_l4 = map((lambda x, y: f(x)), l)