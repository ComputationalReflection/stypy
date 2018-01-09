class Foo:
    def __abs__(self):
        return 4000

    def __add__(self, other):
        return 4

class Foo2:
    def __getitem__(self, x1):
        return x1

r1 = Foo()[1:2]  # missing __getslice__ not reported

x = 3
r2 = x[1:3] # Not reported

r3 = Foo2()[1:2]
