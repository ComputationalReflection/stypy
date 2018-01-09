
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
try:
    a = 3
    raise Exception()
except KeyError as k:
    a = '3'
except Exception as k2:
    a = k2
z = None
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()