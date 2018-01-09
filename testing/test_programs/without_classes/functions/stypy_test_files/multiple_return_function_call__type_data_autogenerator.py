
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def function(a):

    if (a > 0):
        return ('Positive', type_test.add_type_dict_for_context(locals()))[0]


    if (a < 0):
        return (a, type_test.add_type_dict_for_context(locals()))[0]


    if (a == 0):
        return (False, type_test.add_type_dict_for_context(locals()))[0]

    return (list, type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

x = function(3)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()