
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from stypy_copy.errors_copy.type_warning_copy import TypeWarning

from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

warn1 = TypeWarning(Localization('foo_file.py', 1, 1), 'foo')
warn2 = TypeWarning.instance(Localization('foo_file.py', 1, 1), 'foo')
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()