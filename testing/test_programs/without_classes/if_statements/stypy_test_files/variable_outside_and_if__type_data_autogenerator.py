
import stypy.util.type_inference_code_testing.type_data_file_writer
type_test = stypy.util.type_inference_code_testing.type_data_file_writer.TypeDataFileWriter(__file__)
c = False
out_and_if = 1
if (not c):
    out_and_if = '1'
result = (out_and_if * 3)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()