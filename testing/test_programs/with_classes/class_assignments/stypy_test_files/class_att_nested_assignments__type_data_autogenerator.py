
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

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
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()