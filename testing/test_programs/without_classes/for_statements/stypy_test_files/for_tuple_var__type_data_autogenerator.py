
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
arguments = {'a': 1, 'b': 2}
ret_str = ''
for (key, arg) in arguments.items():
    ret_str += ((str(key) + ': ') + str(arg))
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()