
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Simple:
    sample_att = 3
    (a, b) = (6, 7)

    def sample_method(self):
        self.att = 'sample'
        type_test.add_type_dict_for_context(locals())

s = Simple()
s.sample_method()
result = s.att
result2 = s.b
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()