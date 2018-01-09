
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
import module_to_import


def f():
    return (3, type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

z = f()
x = module_to_import.global_a
y = module_to_import.f_parent()
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()