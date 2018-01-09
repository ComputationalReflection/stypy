
# fun1: function
# fun1(l: list[]) -> TypeError /\ fun1(l: tuple[]) -> TypeError 

def fun1(l):
    return ('aaa' + l[0])

# l: list[]; r1: TypeError
r1 = fun1(list)
# l: tuple[]; r2: TypeError
r2 = fun1(tuple)