
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
__temp_call_assignment1 = range(1, 6).__getitem__(0)
Ident1 = __temp_call_assignment1
__temp_call_assignment2 = range(1, 6).__getitem__(1)
Ident2 = __temp_call_assignment2
__temp_call_assignment3 = range(1, 6).__getitem__(2)
Ident3 = __temp_call_assignment3
__temp_call_assignment4 = range(1, 6).__getitem__(3)
Ident4 = __temp_call_assignment4
__temp_call_assignment5 = range(1, 6).__getitem__(4)
Ident5 = __temp_call_assignment5
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()