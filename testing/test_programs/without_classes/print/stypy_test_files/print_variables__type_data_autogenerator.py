
from stypy.code_generation.type_inference.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
__version__ = '2'
loops = 100
benchtime = 1000
stones = 10
print ('Pystone(%s) time for %d passes = %g' % (__version__, loops, benchtime))
print ('This machine benchmarks at %g pystones/second' % stones)
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()