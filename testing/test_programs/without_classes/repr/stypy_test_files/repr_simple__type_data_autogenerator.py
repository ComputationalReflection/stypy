
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Test:

    def __repr__(self):
        return ('This is a test', type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

x = Test()
y = repr(x)
z = repr(1)
print y
print z
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()