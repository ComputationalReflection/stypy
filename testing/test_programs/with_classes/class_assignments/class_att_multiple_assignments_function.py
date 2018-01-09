
mapper = [(1,2,3), ('a', 'b', 'c'), (1.1, 2.2, 3.3)]

class Test:
    def my_zip(self, list_):
        return zip(list_)

class C:
    (m, n, o) = zip(*mapper)
    (p, q, r) = zip(Test().my_zip(mapper))

c = C()

r = c.m
r2 = c.n
r3 = c.o

r4 = c.p
r5 = c.q
r6 = c.r



