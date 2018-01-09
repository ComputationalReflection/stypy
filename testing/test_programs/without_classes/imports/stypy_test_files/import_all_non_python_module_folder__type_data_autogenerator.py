
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from modules.module_to_import import *

x = global_a2
y = f_parent2()
z = submodule
w = submodule.submodule_var
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()