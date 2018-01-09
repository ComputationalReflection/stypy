
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import secondary_module
2: 
3: global_a = 1
4: 
5: 
6: def f_parent():
7:     local_a = 2
8: 
9:     return local_a
10: 
11: my_func = secondary_module.secondary_function
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import secondary_module' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5145 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module')

if (type(import_5145) is not StypyTypeError):

    if (import_5145 != 'pyd_module'):
        __import__(import_5145)
        sys_modules_5146 = sys.modules[import_5145]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module', sys_modules_5146.module_type_store, module_type_store)
    else:
        import secondary_module

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module', secondary_module, module_type_store)

else:
    # Assigning a type to the variable 'secondary_module' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'secondary_module', import_5145)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Num to a Name (line 3):
int_5147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
# Assigning a type to the variable 'global_a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'global_a', int_5147)

@norecursion
def f_parent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f_parent'
    module_type_store = module_type_store.open_function_context('f_parent', 6, 0, False)
    
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

    
    # Assigning a Num to a Name (line 7):
    int_5148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
    # Assigning a type to the variable 'local_a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'local_a', int_5148)
    # Getting the type of 'local_a' (line 9)
    local_a_5149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'local_a')
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', local_a_5149)
    
    # ################# End of 'f_parent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f_parent' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_5150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5150)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f_parent'
    return stypy_return_type_5150

# Assigning a type to the variable 'f_parent' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'f_parent', f_parent)

# Assigning a Attribute to a Name (line 11):
# Getting the type of 'secondary_module' (line 11)
secondary_module_5151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'secondary_module')
# Obtaining the member 'secondary_function' of a type (line 11)
secondary_function_5152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 10), secondary_module_5151, 'secondary_function')
# Assigning a type to the variable 'my_func' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'my_func', secondary_function_5152)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
