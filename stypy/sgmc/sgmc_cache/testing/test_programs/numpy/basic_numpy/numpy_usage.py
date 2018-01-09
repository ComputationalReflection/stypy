
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: from numpy import add
3: 
4: class C:
5:     pass
6: 
7: print add
8: 
9: r = add(C, C)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from numpy import add' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', sys_modules_2.module_type_store, module_type_store, ['add'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from numpy import add

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', None, module_type_store, ['add'], [add])

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')

# Declaration of the 'C' class

class C:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'C' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'C', C)
# Getting the type of 'add' (line 7)
add_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 6), 'add')

# Assigning a Call to a Name (line 9):

# Call to add(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'C' (line 9)
C_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'C', False)
# Getting the type of 'C' (line 9)
C_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'C', False)
# Processing the call keyword arguments (line 9)
kwargs_7 = {}
# Getting the type of 'add' (line 9)
add_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'add', False)
# Calling add(args, kwargs) (line 9)
add_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_4, *[C_5, C_6], **kwargs_7)

# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', add_call_result_8)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
