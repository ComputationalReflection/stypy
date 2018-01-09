
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
a = 3
b = 4
c = 8
c1 = (a < b)
c2 = (c < b)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()