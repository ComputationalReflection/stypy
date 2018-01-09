
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def function(x):

    def another_function(z):
        return (str(z), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

    return (another_function(x), type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

ret = function(3)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()