def fun1(l):
    return "aaa" + l[0]


def fun2(l):
    return 3 / l[0]


S = [x ** 2 for x in range(10)]
V = [str(i) for i in range(13)]

normal_list = [1, 2, 3]
tuple_ = (1, 2, 3)

r1 = fun1(S[0])  # The error is reported on callsite instead on the actual code
r2 = fun1(S)  # No error reported. Runtime crash
r3 = fun1(normal_list)  # No error reported
r4 = fun1(tuple_)  # No error reported
r5 = fun2(V)  # No error reported. Runtime crash
