
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def createGenerator():
    mylist = range(3)
    for i in mylist:
        (yield (i * i))
    type_test.add_type_dict_for_context(locals())

x = createGenerator()
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()