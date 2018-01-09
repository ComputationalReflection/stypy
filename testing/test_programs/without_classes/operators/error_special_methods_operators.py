# coding=utf-8
class Eq5:
    def __gt__(self, other):
        return str(other) + other


r1 = Eq5() > 3


class Eq6:
    def __gt__(self, other):
        return other[other]

r2 = Eq6() > 3