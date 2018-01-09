
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Foo:

    @staticmethod
    def static(x, y):
        return ((x + y), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def instance(self, x, y):
        return ((x + y), type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

f = Foo()
r1 = Foo.static(3, 4)
r2 = Foo().instance('a', 'b')
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()