
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy.core
2: 
3: 
4: x = numpy.core.fabs(-5)
5: 
6: print x

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy.core' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core')

if (type(import_209064) is not StypyTypeError):

    if (import_209064 != 'pyd_module'):
        __import__(import_209064)
        sys_modules_209065 = sys.modules[import_209064]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', sys_modules_209065.module_type_store, module_type_store)
    else:
        import numpy.core

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', numpy.core, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', import_209064)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Call to a Name (line 4):

# Call to fabs(...): (line 4)
# Processing the call arguments (line 4)
int_209069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'int')
# Processing the call keyword arguments (line 4)
kwargs_209070 = {}
# Getting the type of 'numpy' (line 4)
numpy_209066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy', False)
# Obtaining the member 'core' of a type (line 4)
core_209067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), numpy_209066, 'core')
# Obtaining the member 'fabs' of a type (line 4)
fabs_209068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), core_209067, 'fabs')
# Calling fabs(args, kwargs) (line 4)
fabs_call_result_209071 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), fabs_209068, *[int_209069], **kwargs_209070)

# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', fabs_call_result_209071)
# Getting the type of 'x' (line 6)
x_209072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'x')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
