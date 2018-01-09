
_mapper = [(1, 4, 'a'), (2, 5, 'b'), (3, 6, 'c')]


class Foo:
    (_defaulttype, _defaultfunc, _defaultfill) = zip(*_mapper)

f = Foo()

r1 = f._defaulttype
r2 = f._defaultfunc
r3 = f._defaultfill
