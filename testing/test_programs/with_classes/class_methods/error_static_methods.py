class Foo:
    @staticmethod
    def static(x, y):
        return x + y

    def instance(self, x, y):
        return x + y


f = Foo()

r1 = Foo.instance(3, 4)
r2 = Foo().static("a", "b")
