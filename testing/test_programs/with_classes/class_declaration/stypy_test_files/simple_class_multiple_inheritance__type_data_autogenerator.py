
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)

class Parent1:
    pass

class Child(Parent1, ):
    pass

class Parent2:
    pass

class HybridChild(Parent1, Parent2, ):
    pass
instance = HybridChild()
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()