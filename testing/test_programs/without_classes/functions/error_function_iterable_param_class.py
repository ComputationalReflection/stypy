def fun1(l):
    return "aaa" + l[0]


r1 = fun1(list) # No error reported
r2 = fun1(tuple) # No error reported

