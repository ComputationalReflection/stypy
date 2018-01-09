
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
string = 'this is an string'
s = ''
a = 0
for c in string:
    s = (s + c)
    a = (a + 1)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()