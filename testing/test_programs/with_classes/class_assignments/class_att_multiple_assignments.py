class C:
    r = "hi"
    a, b = 5, 3
    c = 4, 5
    (m, n, o) = (4, 5, 6)
    (m, n, o) = [4, 5, 6]
    [x, y, z, r] = [1, 2, 3, 4]
    (x, y, z, r) = [1, 2, 3, 4]

    x1=x2=x3=5

    (r1,r2)=(r3,r4)=(8,9)
    [lr1,lr2]=[lr3,lr4]=(13,14)

    (lr1,lr2)=[lr3,lr4]=(113,114)

    def method(self):
        r = "hi"
        a, b = 5, 3
        c = 4, 5
        (m, n, o) = (4, 5, 6)
        (m, n, o) = [4, 5, 6]
        [x, y, z, r] = [1, 2, 3, 4]
        (x, y, z, r) = [1, 2, 3, 4]

        x1=x2=x3=5

        (r1,r2)=(r3,r4)=(8,9)
        [lr1,lr2]=[lr3,lr4]=(13,14)

        (lr1,lr2)=[lr3,lr4]=(113,114)

ca = C.a
cb = C.b
cc = C.c
cr = C.r
cm = C.m
cn = C.n
co = C.o
cx = C.x
cy = C.y
cz = C.z
cx1 = C.x1
cx2 = C.x2
cx3 = C.x3
cr1 = C.r1
cr2 = C.r2
cr3 = C.r3
cr4 = C.r4
clr1 = C.lr1
clr2 = C.lr2
clr3 = C.lr3
clr4 = C.lr4

c = C()
c.method()