
class Foo:
    def __getitem__(self, *args):
        print args
        return args

x = Foo()
r1 = x[1]

r2 = x[1:]

r3 = x[1:, :]

r4 = x[1:, 20:10:-2, ...]
