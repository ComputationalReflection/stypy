class Record:
    def __init__(self, PtrComp=None, Discr=0, EnumComp=0,
                 IntComp=0, StringComp=0):
        self.PtrComp = PtrComp
        self.Discr = Discr
        self.EnumComp = EnumComp
        self.IntComp = IntComp
        self.StringComp = StringComp

    def copy(self):
        return Record(self.PtrComp, self.Discr, self.EnumComp,
                      self.IntComp, self.StringComp)


r = Record()

x1 = r.PtrComp
x2 = r.Discr
x3 = r.EnumComp
x4 = r.IntComp
x5 = r.StringComp

r2 = r.copy()

y1 = r2.PtrComp
y2 = r2.Discr
y3 = r2.EnumComp
y4 = r2.IntComp
y5 = r2.StringComp