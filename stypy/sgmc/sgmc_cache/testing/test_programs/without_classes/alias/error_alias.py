
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from math import cos as aliased
2: 
3: aliased = []  # Wrong error
4: 
5: 
6: def alias():
7:     r = aliased(0.5)  # Not detected
8: 
9: 
10: alias()
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from math import aliased' statement (line 1)
from math import cos as aliased

import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', None, module_type_store, ['cos'], [aliased])
# Adding an alias
module_type_store.add_alias('aliased', 'cos')


# Assigning a List to a Name (line 3):

# Obtaining an instance of the builtin type 'list' (line 3)
list_6916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)

# Assigning a type to the variable 'aliased' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'aliased', list_6916)

@norecursion
def alias(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'alias'
    module_type_store = module_type_store.open_function_context('alias', 6, 0, False)
    
    # Passed parameters checking function
    alias.stypy_localization = localization
    alias.stypy_type_of_self = None
    alias.stypy_type_store = module_type_store
    alias.stypy_function_name = 'alias'
    alias.stypy_param_names_list = []
    alias.stypy_varargs_param_name = None
    alias.stypy_kwargs_param_name = None
    alias.stypy_call_defaults = defaults
    alias.stypy_call_varargs = varargs
    alias.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alias', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alias', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alias(...)' code ##################

    
    # Assigning a Call to a Name (line 7):
    
    # Call to aliased(...): (line 7)
    # Processing the call arguments (line 7)
    float_6918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'float')
    # Processing the call keyword arguments (line 7)
    kwargs_6919 = {}
    # Getting the type of 'aliased' (line 7)
    aliased_6917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'aliased', False)
    # Calling aliased(args, kwargs) (line 7)
    aliased_call_result_6920 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), aliased_6917, *[float_6918], **kwargs_6919)
    
    # Assigning a type to the variable 'r' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'r', aliased_call_result_6920)
    
    # ################# End of 'alias(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alias' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_6921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alias'
    return stypy_return_type_6921

# Assigning a type to the variable 'alias' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'alias', alias)

# Call to alias(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_6923 = {}
# Getting the type of 'alias' (line 10)
alias_6922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'alias', False)
# Calling alias(args, kwargs) (line 10)
alias_call_result_6924 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), alias_6922, *[], **kwargs_6923)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
