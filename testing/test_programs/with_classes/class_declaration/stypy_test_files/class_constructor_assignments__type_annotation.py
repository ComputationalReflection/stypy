
# Record: Record

class Record:
    # __init__(PtrComp: NoneType, Discr: int, EnumComp: int, IntComp: int, StringComp: int) -> None 

    def __init__(self, PtrComp=None, Discr=0, EnumComp=0, IntComp=0, StringComp=0):
        # PtrComp: NoneType
        self.PtrComp = PtrComp
        # Discr: int
        self.Discr = Discr
        # EnumComp: int
        self.EnumComp = EnumComp
        # IntComp: int
        self.IntComp = IntComp
        # StringComp: int
        self.StringComp = StringComp

    # copy() -> Record instance 

    def copy(self):
        # StringComp: int; self: instance; Discr: int; PtrComp: NoneType; IntComp: int; EnumComp: int
        return Record(self.PtrComp, self.Discr, self.EnumComp, self.IntComp, self.StringComp)

# StringComp: int; Discr: int; r: Record instance; PtrComp: NoneType; IntComp: int; EnumComp: int
r = Record()
# x1: NoneType
x1 = r.PtrComp
# x2: int
x2 = r.Discr
# x3: int
x3 = r.EnumComp
# x4: int
x4 = r.IntComp
# x5: int
x5 = r.StringComp
# r2: Record instance
r2 = r.copy()
# y1: NoneType
y1 = r2.PtrComp
# y2: int
y2 = r2.Discr
# y3: int
y3 = r2.EnumComp
# y4: int
y4 = r2.IntComp
# y5: int
y5 = r2.StringComp