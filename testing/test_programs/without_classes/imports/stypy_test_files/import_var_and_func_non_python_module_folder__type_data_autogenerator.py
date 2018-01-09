
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from modules import module_to_import

x = module_to_import.global_a2
y = module_to_import.f_parent2()
z = module_to_import.submodule
w = module_to_import.submodule.submodule_var
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()