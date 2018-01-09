
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class controlled_execution:

    def __enter__(self):
        print 'enter the with class'
        return (0, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


    def __exit__(self, type, value, traceback):
        print 'exit the with class'
        type_test.add_type_dict_for_context(locals())

a = 3
with controlled_execution() as thing:
    a = (a + 1)
    print thing
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()