
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from modules.other_module_to_import import *

x = global_a2
y = f_parent2()
z = submodule_func()
w = submodule_var
a = var1
b = var2
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()