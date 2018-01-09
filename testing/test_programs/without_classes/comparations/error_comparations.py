def comparations():
    a = 3
    b = 4
    c = 8

    class Foo:
        # def __cmp__(self): #Do not detect the wrong definition of predefined methods (missing param)
        def __cmp__(self, other):
            return range(5)  # Do not detect "comparison did not return an int"

    c0 = a < Foo()  # Not reported
    c1 = a < b < Foo()  # Not reported


comparations()
