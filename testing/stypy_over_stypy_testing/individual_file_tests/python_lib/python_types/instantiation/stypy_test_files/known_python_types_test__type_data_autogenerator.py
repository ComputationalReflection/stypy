
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *

r1 = simple_python_types
r2 = known_python_type_typename_samplevalues
r3 = ExtraTypeDefinitions
r4 = ExtraTypeDefinitions.dict_values
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()