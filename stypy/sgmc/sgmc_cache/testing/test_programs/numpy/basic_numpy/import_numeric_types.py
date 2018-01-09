
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy.core.numerictypes
2: 
3: x = numpy.core.numerictypes.bool
4: x2 = numpy.core.numerictypes.bool_

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy.core.numerictypes' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.numerictypes')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.numerictypes', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy.core.numerictypes

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.numerictypes', numpy.core.numerictypes, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.numerictypes', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'numpy' (line 3)
numpy_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'numpy')
# Obtaining the member 'core' of a type (line 3)
core_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), numpy_3, 'core')
# Obtaining the member 'numerictypes' of a type (line 3)
numerictypes_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), core_4, 'numerictypes')
# Obtaining the member 'bool' of a type (line 3)
bool_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), numerictypes_5, 'bool')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', bool_6)

# Assigning a Attribute to a Name (line 4):
# Getting the type of 'numpy' (line 4)
numpy_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'numpy')
# Obtaining the member 'core' of a type (line 4)
core_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), numpy_7, 'core')
# Obtaining the member 'numerictypes' of a type (line 4)
numerictypes_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), core_8, 'numerictypes')
# Obtaining the member 'bool_' of a type (line 4)
bool__10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), numerictypes_9, 'bool_')
# Assigning a type to the variable 'x2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x2', bool__10)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
