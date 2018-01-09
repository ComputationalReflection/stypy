
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def function(x):
    return (x, type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

y = function(3)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()