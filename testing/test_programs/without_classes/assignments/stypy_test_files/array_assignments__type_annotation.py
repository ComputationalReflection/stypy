
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

# Array1Glob: list[int]
Array1Glob = ([0] * 51)
# x: list[int]; Array2Glob: list[list[int]]
Array2Glob = map((lambda x: x[:]), ([Array1Glob] * 51))
# IntLoc1: int
IntLoc1 = 1
# IntLoc3: int
IntLoc3 = 3
# Array2Par: list[list[int]]; Array1Par: list[int]; IntParI2: int; IntParI1: int
Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)