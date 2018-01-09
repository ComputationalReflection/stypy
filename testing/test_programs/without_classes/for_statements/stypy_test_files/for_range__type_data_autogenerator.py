
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
a = 3
b = False
for i in range(1000):
    b = 'hi'
    a = (a - 1)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()