
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def identity(x):
    return (x, type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

y = identity(3)
z = identity('3')
w = identity(3.4)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()