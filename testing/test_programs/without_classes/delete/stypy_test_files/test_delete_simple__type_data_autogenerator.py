
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Foo:
    att = 'sample'

    def met(self):
        return (self.att, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())

f = Foo()
att_predelete = f.att
met_predelete = f.met
met_result_predelete = f.met()
func_predelete = f.met
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()