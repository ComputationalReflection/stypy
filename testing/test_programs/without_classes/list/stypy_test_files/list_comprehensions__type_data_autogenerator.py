
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
S = [(x ** 2) for x in range(10)]
V = [str(i) for i in range(13)]
M = [x for x in S if ((x % 2) == 0)]
noprimes = [j for i in range(2, 8) for j in range((i * 2), 50, i)]
primes = [x for x in range(2, 50) if (x not in noprimes)]
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()