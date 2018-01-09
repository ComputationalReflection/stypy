
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
x = 'str'
exec 'x=4+7/2'
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()