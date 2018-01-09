
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: import numpy.core.numerictypes
4: 
5: r = numpy.core.numerictypes.float
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy.core.numerictypes' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209073 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.numerictypes')

if (type(import_209073) is not StypyTypeError):

    if (import_209073 != 'pyd_module'):
        __import__(import_209073)
        sys_modules_209074 = sys.modules[import_209073]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.numerictypes', sys_modules_209074.module_type_store, module_type_store)
    else:
        import numpy.core.numerictypes

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.numerictypes', numpy.core.numerictypes, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.numerictypes', import_209073)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 5):
# Getting the type of 'numpy' (line 5)
numpy_209075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy')
# Obtaining the member 'core' of a type (line 5)
core_209076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), numpy_209075, 'core')
# Obtaining the member 'numerictypes' of a type (line 5)
numerictypes_209077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), core_209076, 'numerictypes')
# Obtaining the member 'float' of a type (line 5)
float_209078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), numerictypes_209077, 'float')
# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', float_209078)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
