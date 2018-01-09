
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
d = {n: (n ** 2) for n in range(5)}
print d
d2 = {n: True for n in range(5)}
print d2
d3 = {k: k for k in range(10)}
print d3
old_dict = {'a': 1, 'c': 3, 'b': 2}
print old_dict
new_dict = {key: 'your value here' for key in old_dict.keys()}
print new_dict
S = [(x ** 2) for x in range(10)]
d4 = {k: k for k in S}
print d4
noprimes = [j for i in range(2, 8) for j in range((i * 2), 50, i)]
primes = [x for x in range(2, 50) if (x not in noprimes)]
d5 = {k: k for k in primes}
print d5
words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
d6 = {k: len(k) for k in words}
print d6
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()