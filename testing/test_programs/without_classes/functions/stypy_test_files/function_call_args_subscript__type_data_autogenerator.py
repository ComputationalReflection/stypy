
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
dic = {'a': 1, 'b': 2}
tup = ([3], dic)

def func(*args, **kwargs):
    print args
    print kwargs
    type_test.add_type_dict_for_context(locals())

func(*tup[0], **tup[1])
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()