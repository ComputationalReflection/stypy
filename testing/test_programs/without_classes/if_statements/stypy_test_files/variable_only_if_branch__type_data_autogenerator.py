
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
a = 3
condition = (a > 0)

if condition:
    f = 3
else:
    x = 4

f = (f + 2)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()