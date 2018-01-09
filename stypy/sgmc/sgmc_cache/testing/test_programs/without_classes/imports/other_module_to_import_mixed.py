
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import secondary_module_mixed
2: import math
3: 
4: global_a = 1
5: 
6: 
7: def f_parent():
8:     local_a = 2
9: 
10:     return local_a
11: 
12: my_func = secondary_module_mixed.secondary_function
13: my_func2 = math.tan
14: my_func3 = secondary_module_mixed.time.time

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import secondary_module_mixed' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5153 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module_mixed')

if (type(import_5153) is not StypyTypeError):

    if (import_5153 != 'pyd_module'):
        __import__(import_5153)
        sys_modules_5154 = sys.modules[import_5153]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module_mixed', sys_modules_5154.module_type_store, module_type_store)
    else:
        import secondary_module_mixed

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module_mixed', secondary_module_mixed, module_type_store)

else:
    # Assigning a type to the variable 'secondary_module_mixed' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module_mixed', import_5153)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import math' statement (line 2)
import math

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', math, module_type_store)


# Assigning a Num to a Name (line 4):
int_5155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
# Assigning a type to the variable 'global_a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'global_a', int_5155)

@norecursion
def f_parent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f_parent'
    module_type_store = module_type_store.open_function_context('f_parent', 7, 0, False)
    
    # Passed parameters checking function
    f_parent.stypy_localization = localization
    f_parent.stypy_type_of_self = None
    f_parent.stypy_type_store = module_type_store
    f_parent.stypy_function_name = 'f_parent'
    f_parent.stypy_param_names_list = []
    f_parent.stypy_varargs_param_name = None
    f_parent.stypy_kwargs_param_name = None
    f_parent.stypy_call_defaults = defaults
    f_parent.stypy_call_varargs = varargs
    f_parent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f_parent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f_parent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f_parent(...)' code ##################

    
    # Assigning a Num to a Name (line 8):
    int_5156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
    # Assigning a type to the variable 'local_a' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'local_a', int_5156)
    # Getting the type of 'local_a' (line 10)
    local_a_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'local_a')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', local_a_5157)
    
    # ################# End of 'f_parent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f_parent' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_5158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5158)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f_parent'
    return stypy_return_type_5158

# Assigning a type to the variable 'f_parent' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'f_parent', f_parent)

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'secondary_module_mixed' (line 12)
secondary_module_mixed_5159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'secondary_module_mixed')
# Obtaining the member 'secondary_function' of a type (line 12)
secondary_function_5160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), secondary_module_mixed_5159, 'secondary_function')
# Assigning a type to the variable 'my_func' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'my_func', secondary_function_5160)

# Assigning a Attribute to a Name (line 13):
# Getting the type of 'math' (line 13)
math_5161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'math')
# Obtaining the member 'tan' of a type (line 13)
tan_5162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), math_5161, 'tan')
# Assigning a type to the variable 'my_func2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'my_func2', tan_5162)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'secondary_module_mixed' (line 14)
secondary_module_mixed_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'secondary_module_mixed')
# Obtaining the member 'time' of a type (line 14)
time_5164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), secondary_module_mixed_5163, 'time')
# Obtaining the member 'time' of a type (line 14)
time_5165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), time_5164, 'time')
# Assigning a type to the variable 'my_func3' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'my_func3', time_5165)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
