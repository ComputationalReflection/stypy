
class Argsort:
    __doc__ = "argsort doc"

class ndarray:
    argsort = Argsort()

class C:
    argsort = Argsort()

    def __init__(self):
        pass

    argsort.__doc__ = ndarray.argsort.__doc__
    argsort.__doc__.__doc__ = ndarray.argsort.__doc__

c = C()

x = c.argsort.__doc__
y = c.argsort.__doc__.capitalize()