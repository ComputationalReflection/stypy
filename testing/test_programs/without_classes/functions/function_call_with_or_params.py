
class Foo:
    def __init__(self, txt, params=None):
        self.txt = txt
        self._extras = params or {}

    def pars(self, x=3):
        return x

f = Foo("test")

r = f.pars()

r2 = f._extras