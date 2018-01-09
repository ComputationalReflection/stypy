
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class C:

    def __init__(self):
        pass
        type_test.add_type_dict_for_context(locals())


    def method(self):
        self.r = 'str'
        type_test.add_type_dict_for_context(locals())

c = C()
c.method()
x = (c.r == 5)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()