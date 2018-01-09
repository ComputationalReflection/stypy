from stypy_copy.errors_copy.type_warning_copy import TypeWarning
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

warn1 = TypeWarning(Localization("foo_file.py", 1, 1), "foo")
warn2 = TypeWarning.instance(Localization("foo_file.py", 1, 1), "foo")
# #warn3 = TypeWarning(None, "foo")
#
# r1 = warn1.print_warning_msgs()
# r2 = warn1.reset_warning_msgs()
# r3 = warn1.get_warning_msgs()