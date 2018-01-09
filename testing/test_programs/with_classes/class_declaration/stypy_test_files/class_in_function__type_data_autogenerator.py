
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def function():

    class Simple:
        sample_att = 3

        def sample_method(self):
            self.att = 'sample'
            type_test.add_type_dict_for_context(locals())

    return (Simple(), type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

ret = function()
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()