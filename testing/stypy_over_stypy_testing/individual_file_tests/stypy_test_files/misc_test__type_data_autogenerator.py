
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

x = union_type_copy.UnionType
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()