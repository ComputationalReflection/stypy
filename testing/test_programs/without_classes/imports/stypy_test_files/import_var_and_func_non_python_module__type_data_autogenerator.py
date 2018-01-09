
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from module_to_import import global_a, f_parent

x = global_a
f_parent()
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()