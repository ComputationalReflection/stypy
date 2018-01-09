
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

def f(x, y, z, *arguments, **kwarguments):
    pass
    type_test.add_type_dict_for_context(locals())


def f2(x, y, z, *arguments, **kwarguments):
    pass
    type_test.add_type_dict_for_context(locals())


def f3(x=5, y=6, z=4, *args, **kwargs):
    pass
    type_test.add_type_dict_for_context(locals())

f(2, 3, 4, 5, 6, 7)
f2(1, 2, 8, 6, 4, r=23)
f3(z='1', x=4, y=True, r=11, s='12')
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()