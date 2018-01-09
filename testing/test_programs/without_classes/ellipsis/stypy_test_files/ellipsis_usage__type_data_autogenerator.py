
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Foo:

    def __getitem__(self, *args):
        print args
        return (args, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

x = Foo()
r1 = x[1]
r2 = x[1:]
r3 = x[1:, :]
r4 = x[1:, 20:10:(-2), ...]
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()