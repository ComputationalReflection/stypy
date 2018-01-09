class Foo:
    att = 3

    def met(self):
        self.my_att = 3

        return 3


f = Foo()

del f.my_att  # Not reported and met was not called

a = 0
if a > 0:
    f.xx = 3
else:
    f.yy = 5

del f.yy  # Not detected
del f.xx

del list.__doc__  # Failure not detected
