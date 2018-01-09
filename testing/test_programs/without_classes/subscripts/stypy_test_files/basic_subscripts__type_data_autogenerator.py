
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
l = [1, 2, 4, 5]
a = 2
x = l[1]
y = l[a]
z = l[(a + 1)]
s = 'abcd'
c = s[2]
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()