
# fun1: function
# fun1(l: int) -> TypeError /\ fun1(l: list[int]) -> TypeError /\ fun1(l: tuple[int]) -> TypeError 

def fun1(l):
    return ('aaa' + l[0])

# fun2: function
# fun2(l: list[str]) -> TypeError 

def fun2(l):
    return (3 / l[0])

# x: int; S: list[int]
S = [(x ** 2) for x in range(10)]
# i: int; V: list[str]
V = [str(i) for i in range(13)]
# normal_list: list[int]
normal_list = [1, 2, 3]
# tuple_: tuple[int]
tuple_ = (1, 2, 3)
# l: int; r1: TypeError
r1 = fun1(S[0])
# l: list[int]; r2: TypeError
r2 = fun1(S)
# l: list[int]; r3: TypeError
r3 = fun1(normal_list)
# r4: TypeError; l: tuple[int]
r4 = fun1(tuple_)
# r5: TypeError; l: list[str]
r5 = fun2(V)