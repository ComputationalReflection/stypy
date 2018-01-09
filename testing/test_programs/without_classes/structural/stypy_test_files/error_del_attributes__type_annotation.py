
# Foo: Foo

class Foo:
    # self: instance
    att = 3
    # <Dead code detected>

    def met(self):
        self.my_att = 3
        return 3

# f: Foo instance
f = Foo()
del f.my_att
# a: int
a = 0

if (a > 0):
    # xx: int
    f.xx = 3
else:
    # yy: int
    f.yy = 5

del f.yy
del f.xx
del list.__doc__