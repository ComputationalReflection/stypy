class MaskedArray:
    def var(self):
        pass

    var.__doc__ = "hi"


m = MaskedArray()
y = m.var.__doc__
