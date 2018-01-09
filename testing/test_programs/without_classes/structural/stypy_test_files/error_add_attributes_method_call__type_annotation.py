
# Foo: Foo

class Foo:
    # self: instance
    att = 3
    # <Dead code detected>

    def met(self):
        self.my_att = 3
        return 3

    # met_class() -> int 

    def met_class(self):
        # my_att: int; self: instance
        self.my_att = 3
        # class_att: bool
        Foo.class_att = True
        return 3

# f1: Foo instance
f1 = Foo()
# r1: TypeError
r1 = f1.my_att
# f2: Foo instance
f2 = Foo()
f2.met_class()
# r2: bool
r2 = Foo.class_att