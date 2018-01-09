class Foo:
    att = 3

    def met(self):
        self.my_att = 3

        return 3

    def met_class(self):
        self.my_att = 3
        Foo.class_att = True
        return 3


f1 = Foo()
# f1.met() # Not called!
r1 = f1.my_att  # No error reported, but it is an error because met was not called

f2 = Foo()

f2.met_class()
r2 = Foo.class_att  # This is reported as an error even calling met()!
