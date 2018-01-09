
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
(a, b) = (5, 3)
c = (4, 5)
[x, y, z, r] = [1, 2, 3, 4]
(m, n, o) = (4, 5, 6)
print a, b
print c
print x, y, z, r
print m, n, o
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()