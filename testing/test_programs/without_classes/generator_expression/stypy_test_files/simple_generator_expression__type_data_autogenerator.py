
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
r = sum(((x * x) for x in range(10)))
print r
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()