
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
l = [1, 2, 3, 4]
other_l = map((lambda x: str(x)), l)
l2 = [False, 1, 'string']
other_l2 = map((lambda x: str(x)), l)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()