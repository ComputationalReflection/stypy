def asbytes(st):
    return 0


class StringConverter(object):
    _mapper = [(bool, str, False)]
    if True:
        _mapper.append((int, int, -1))

    _mapper.extend([(float, float, None), (str, bytes, asbytes('???'))])

    a, b, c = zip(*_mapper)

st = StringConverter()
r = st._mapper
r2 = st.a
r3 = st.b
r4 = st.c

print r
