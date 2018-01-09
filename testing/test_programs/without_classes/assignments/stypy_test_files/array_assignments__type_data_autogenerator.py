
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def Proc8(Array1Par, Array2Par, IntParI1, IntParI2):
    global IntGlob
    IntLoc = (IntParI1 + 5)
    Array1Par[IntLoc] = IntParI2
    Array1Par[(IntLoc + 1)] = Array1Par[IntLoc]
    Array1Par[(IntLoc + 30)] = IntLoc
    for IntIndex in range(IntLoc, (IntLoc + 2)):
        Array2Par[IntLoc][IntIndex] = IntLoc
    Array2Par[IntLoc][(IntLoc - 1)] = (Array2Par[IntLoc][(IntLoc - 1)] + 1)
    Array2Par[(IntLoc + 20)][IntLoc] = Array1Par[IntLoc]
    IntGlob = 5
    type_test.add_type_dict_for_context(locals())

Array1Glob = ([0] * 51)
Array2Glob = map((lambda x: x[:]), ([Array1Glob] * 51))
IntLoc1 = 1
IntLoc3 = 3
Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()