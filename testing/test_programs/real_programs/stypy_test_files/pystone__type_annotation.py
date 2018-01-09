
# LOOPS: int
LOOPS = 50000
# clock: builtin_function_or_method
from time import clock

# __version__: str
__version__ = '1.1'
# Ident4: int; Ident5: int; Ident1: int; Ident2: int; Ident3: int
[Ident1, Ident2, Ident3, Ident4, Ident5] = range(1, 6)
# Record: Record

class Record:
    # __init__(PtrComp: NoneType, Discr: int, EnumComp: int, IntComp: int, StringComp: int) -> None /\ __init__(PtrComp: Record instance, Discr: int, EnumComp: int, IntComp: int, StringComp: str) -> None 

    def __init__(self, PtrComp=None, Discr=0, EnumComp=0, IntComp=0, StringComp=0):
        # PtrComp: NoneType \/ Record instance \/ Record instance
        self.PtrComp = PtrComp
        # Discr: int
        self.Discr = Discr
        # EnumComp: int
        self.EnumComp = EnumComp
        # IntComp: int
        self.IntComp = IntComp
        # StringComp: int \/ str
        self.StringComp = StringComp

    # copy() -> Record instance 

    def copy(self):
        # StringComp: str; self: instance; Discr: int; PtrComp: Record instance \/ Record instance; IntComp: int; EnumComp: int
        return Record(self.PtrComp, self.Discr, self.EnumComp, self.IntComp, self.StringComp)

# TRUE: int
TRUE = 1
# FALSE: int
FALSE = 0
# main: function
# main(loops: int) -> None 

def main(loops=LOOPS):
    # loops: int; stones: float; benchtime: float
    (benchtime, stones) = pystones(loops)
    print ('Pystone(%s) time for %d passes = %g' % (__version__, loops, benchtime))
    print ('This machine benchmarks at %g pystones/second' % stones)

# pystones: function
# pystones(loops: int) -> tuple[float] 

def pystones(loops=LOOPS):
    # loops: int
    return Proc0(loops)

# IntGlob: int
IntGlob = 0
# BoolGlob: int
BoolGlob = FALSE
# Char1Glob: str
Char1Glob = '\x00'
# Char2Glob: str
Char2Glob = '\x00'
# Array1Glob: list[int]
Array1Glob = ([0] * 51)
# x: list[int]; Array2Glob: list[list[int]]
Array2Glob = map((lambda x: x[:]), ([Array1Glob] * 51))
# PtrGlb: NoneType
PtrGlb = None
# PtrGlbNext: NoneType
PtrGlbNext = None
# Proc0: function
# Proc0(loops: int) -> tuple[float] 

def Proc0(loops=LOOPS):
    global IntGlob
    global BoolGlob
    global Char1Glob
    global Char2Glob
    global Array1Glob
    global Array2Glob
    global PtrGlb
    global PtrGlbNext
    # starttime: float
    starttime = clock()
    # i: int
    for i in range(loops):
        pass
    # nulltime: float
    nulltime = (clock() - starttime)
    # StringComp: int; Discr: int; PtrComp: NoneType; IntComp: int; PtrGlbNext: Record instance; EnumComp: int
    PtrGlbNext = Record()
    # StringComp: int; Discr: int; PtrComp: NoneType; IntComp: int; EnumComp: int; PtrGlb: Record instance
    PtrGlb = Record()
    # PtrComp: Record instance
    PtrGlb.PtrComp = PtrGlbNext
    # Discr: int
    PtrGlb.Discr = Ident1
    # EnumComp: int
    PtrGlb.EnumComp = Ident3
    # IntComp: int
    PtrGlb.IntComp = 40
    # StringComp: str
    PtrGlb.StringComp = 'DHRYSTONE PROGRAM, SOME STRING'
    # String1Loc: str
    String1Loc = "DHRYSTONE PROGRAM, 1'ST STRING"
    Array2Glob[8][7] = 10
    # starttime: float
    starttime = clock()
    # i: int
    for i in range(loops):
        Proc5()
        Proc4()
        # IntLoc1: int
        IntLoc1 = 2
        # IntLoc2: int
        IntLoc2 = 3
        # String2Loc: str
        String2Loc = "DHRYSTONE PROGRAM, 2'ND STRING"
        # EnumLoc: int
        EnumLoc = Ident2
        # StrParI1: str; StrParI2: str; BoolGlob: bool
        BoolGlob = (not Func2(String1Loc, String2Loc))
        while (IntLoc1 < IntLoc2):
            # IntLoc3: int
            IntLoc3 = ((5 * IntLoc1) - IntLoc2)
            # IntParI2: int; IntParI1: int; IntLoc3: int
            IntLoc3 = Proc7(IntLoc1, IntLoc2)
            # IntLoc1: int
            IntLoc1 = (IntLoc1 + 1)
        # Array2Par: list[list[int]]; Array1Par: list[int]; IntParI2: int; IntParI1: int
        Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)
        # PtrParIn: Record instance; PtrGlb: Record instance
        PtrGlb = Proc1(PtrGlb)
        # CharIndex: str
        CharIndex = 'A'
        while (CharIndex <= Char2Glob):
            # CharPar2: str; CharPar1: str

            if (EnumLoc == Func1(CharIndex, 'C')):
                # EnumLoc: int; EnumParIn: int
                EnumLoc = Proc6(Ident1)

            # CharIndex: str
            CharIndex = chr((ord(CharIndex) + 1))
        # IntLoc3: int
        IntLoc3 = (IntLoc2 * IntLoc1)
        # IntLoc2: int
        IntLoc2 = (IntLoc3 / IntLoc1)
        # IntLoc2: int
        IntLoc2 = ((7 * (IntLoc3 - IntLoc2)) - IntLoc1)
        # IntParIO: int; IntLoc1: int
        IntLoc1 = Proc2(IntLoc1)
    # benchtime: float
    benchtime = ((clock() - starttime) - nulltime)

    if (benchtime == 0.0):
        # loopsPerBenchtime: float
        loopsPerBenchtime = 0.0
    else:
        # loopsPerBenchtime: float
        loopsPerBenchtime = (loops / benchtime)

    return (benchtime, loopsPerBenchtime)

# Proc1: function
# Proc1(PtrParIn: Record instance) -> Record instance 

def Proc1(PtrParIn):
    # NextRecord: Record instance; PtrComp: Record instance
    PtrParIn.PtrComp = NextRecord = PtrGlb.copy()
    # IntComp: int
    PtrParIn.IntComp = 5
    # IntComp: int
    NextRecord.IntComp = PtrParIn.IntComp
    # PtrComp: Record instance
    NextRecord.PtrComp = PtrParIn.PtrComp
    # PtrComp: Record instance; PtrParOut: Record instance
    NextRecord.PtrComp = Proc3(NextRecord.PtrComp)

    if (NextRecord.Discr == Ident1):
        # IntComp: int
        NextRecord.IntComp = 6
        # EnumParIn: int; EnumComp: int
        NextRecord.EnumComp = Proc6(PtrParIn.EnumComp)
        # PtrComp: Record instance
        NextRecord.PtrComp = PtrGlb.PtrComp
        # IntParI2: int; IntParI1: int; IntComp: int
        NextRecord.IntComp = Proc7(NextRecord.IntComp, 10)
    else:
        # PtrParIn: Record instance
        PtrParIn = NextRecord.copy()

    # PtrComp: NoneType
    NextRecord.PtrComp = None
    return PtrParIn

# Proc2: function
# Proc2(IntParIO: int) -> int 

def Proc2(IntParIO):
    # IntLoc: int
    IntLoc = (IntParIO + 10)
    while 1:

        if (Char1Glob == 'A'):
            # IntLoc: int
            IntLoc = (IntLoc - 1)
            # IntParIO: int
            IntParIO = (IntLoc - IntGlob)
            # EnumLoc: int
            EnumLoc = Ident1


        if (EnumLoc == Ident1):
            break

    return IntParIO

# Proc3: function
# Proc3(PtrParOut: Record instance) -> Record instance 

def Proc3(PtrParOut):
    global IntGlob

    if (PtrGlb is not None):
        # PtrParOut: Record instance
        PtrParOut = PtrGlb.PtrComp
    else:
        # IntGlob: int
        IntGlob = 100

    # IntParI2: int; IntParI1: int; IntComp: int
    PtrGlb.IntComp = Proc7(10, IntGlob)
    return PtrParOut

# Proc4: function
# Proc4() -> None 

def Proc4():
    global Char2Glob
    # BoolLoc: bool
    BoolLoc = (Char1Glob == 'A')
    # BoolLoc: int
    BoolLoc = (BoolLoc or BoolGlob)
    # Char2Glob: str
    Char2Glob = 'B'

# Proc5: function
# Proc5() -> None 

def Proc5():
    global Char1Glob
    global BoolGlob
    # Char1Glob: str
    Char1Glob = 'A'
    # BoolGlob: int
    BoolGlob = FALSE

# Proc6: function
# Proc6(EnumParIn: int) -> int 

def Proc6(EnumParIn):
    # EnumParOut: int
    EnumParOut = EnumParIn
    # EnumParIn: int

    if (not Func3(EnumParIn)):
        # EnumParOut: int
        EnumParOut = Ident4


    if (EnumParIn == Ident1):
        # EnumParOut: int
        EnumParOut = Ident1
    elif (EnumParIn == Ident2):

        if (IntGlob > 100):
            # EnumParOut: int
            EnumParOut = Ident1
        else:
            # EnumParOut: int
            EnumParOut = Ident4

    elif (EnumParIn == Ident3):
        # EnumParOut: int
        EnumParOut = Ident2
    elif (EnumParIn == Ident4):
        pass
    elif (EnumParIn == Ident5):
        # EnumParOut: int
        EnumParOut = Ident3

    return EnumParOut

# Proc7: function
# Proc7(IntParI1: int, IntParI2: int) -> int 

def Proc7(IntParI1, IntParI2):
    # IntLoc: int
    IntLoc = (IntParI1 + 2)
    # IntParOut: int
    IntParOut = (IntParI2 + IntLoc)
    return IntParOut

# Proc8: function
# Proc8(Array1Par: list[int], Array2Par: list[list[int]], IntParI1: int, IntParI2: int) -> None 

def Proc8(Array1Par, Array2Par, IntParI1, IntParI2):
    global IntGlob
    # IntLoc: int
    IntLoc = (IntParI1 + 5)
    # <container elements type>: int
    Array1Par[IntLoc] = IntParI2
    # <container elements type>: int
    Array1Par[(IntLoc + 1)] = Array1Par[IntLoc]
    # <container elements type>: int
    Array1Par[(IntLoc + 30)] = IntLoc
    # IntIndex: int
    for IntIndex in range(IntLoc, (IntLoc + 2)):
        Array2Par[IntLoc][IntIndex] = IntLoc
    Array2Par[IntLoc][(IntLoc - 1)] = (Array2Par[IntLoc][(IntLoc - 1)] + 1)
    Array2Par[(IntLoc + 20)][IntLoc] = Array1Par[IntLoc]
    # IntGlob: int
    IntGlob = 5

# Func1: function
# Func1(CharPar1: str, CharPar2: str) -> int 

def Func1(CharPar1, CharPar2):
    # CharLoc1: str
    CharLoc1 = CharPar1
    # CharLoc2: str
    CharLoc2 = CharLoc1

    if (CharLoc2 != CharPar2):
        return Ident1
    else:
        return Ident2


# Func2: function
# Func2(StrParI1: str, StrParI2: str) -> int 

def Func2(StrParI1, StrParI2):
    # IntLoc: int
    IntLoc = 1
    while (IntLoc <= 1):
        # CharPar2: str; CharPar1: str

        if (Func1(StrParI1[IntLoc], StrParI2[(IntLoc + 1)]) == Ident1):
            # CharLoc: str
            CharLoc = 'A'
            # IntLoc: int
            IntLoc = (IntLoc + 1)


    if ((CharLoc >= 'W') and (CharLoc <= 'Z')):
        # IntLoc: int
        IntLoc = 7


    if (CharLoc == 'X'):
        return TRUE
    elif (StrParI1 > StrParI2):
        # IntLoc: int
        IntLoc = (IntLoc + 7)
        return TRUE
    else:
        return FALSE


# Func3: function
# Func3(EnumParIn: int) -> int 

def Func3(EnumParIn):
    # EnumLoc: int
    EnumLoc = EnumParIn

    if (EnumLoc == Ident3):
        return TRUE

    return FALSE


if (__name__ == '__main__'):
    # loops: int
    main(LOOPS)
