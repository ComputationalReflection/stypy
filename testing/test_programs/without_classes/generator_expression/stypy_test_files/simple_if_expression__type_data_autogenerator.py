
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
a = 5
b = 6
x = (1 if (a > b) else (-1))
y = (1 if (a > b) else ((-1) if (a < b) else 0))
z = (1 if (a > b) else 'foo')
print x
print y
print z
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()