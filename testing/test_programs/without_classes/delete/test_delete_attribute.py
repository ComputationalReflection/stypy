
class Nested:
    def __init__(self):
        self.a = 3

class Foo:
    def __init__(self):
        self.att = Nested()

    def met(self):
        return self.att



f = Foo()

x1 = f.att.a
del f.att.a
x2 = f.att.a

y1 = f.att
del f.att
y2 = f.att
