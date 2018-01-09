class Inner:
    attInner = 3


class LessInner:
    attLessInner = Inner()


class Outer:
    attOuter = LessInner()


i1 = Inner()
r1 = i1.attInner

i2 = LessInner()
r2 = i2.attLessInner.attInner

i2.attLessInner.attInner = "3"
i2.attLessInner.attInner += "3"
r3 = i2.attLessInner.attInner