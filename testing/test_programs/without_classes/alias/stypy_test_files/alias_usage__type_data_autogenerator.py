
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from math import sin

from math import acos as my_acos

y = my_acos(0.5)
z = my_acos
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()