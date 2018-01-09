
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
import random


class Counter:
    count = 0

    def inc(self, value):
        self.count += value
        return (self.count, type_test.add_type_dict_for_context(locals()))[0]
        type_test.add_type_dict_for_context(locals())


def bitwise_or(counter, n):
    x = counter.count
    return ((counter.count | n), type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())


def flow_sensitive(obj, condition):

    if condition:
        obj.inc(1)
    else:
        obj.inc(0.5)

    return (bitwise_or(obj, 3), type_test.add_type_dict_for_context(locals()))[0]
    type_test.add_type_dict_for_context(locals())

obj = Counter()
flow_sensitive(obj, (random.randint(0, 1) == 0))
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()