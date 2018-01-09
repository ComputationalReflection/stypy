
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
l = [1, 3, False, 'aa']
x = l[0]
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()