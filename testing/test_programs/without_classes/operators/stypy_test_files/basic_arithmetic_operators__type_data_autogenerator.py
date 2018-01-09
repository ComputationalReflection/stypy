
import stypy.util.type_inference_code_testing.type_data_file_writer
type_test = stypy.util.type_inference_code_testing.type_data_file_writer.TypeDataFileWriter(__file__)
a = (4 + 5)
b = 4
c = (a * b)
d = (c / a)
e = ((a + b) + c)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()