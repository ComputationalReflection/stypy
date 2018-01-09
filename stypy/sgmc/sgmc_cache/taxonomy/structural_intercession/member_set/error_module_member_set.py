
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a module"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7: 
8:     def new_func():
9:         return "new function"
10: 
11: 
12:     setattr(math, 'cos', new_func)
13:     setattr(math, 'pi', "str")
14: 
15:     print math.cos()
16:     print len(math.pi)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a module')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    

    @norecursion
    def new_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_func'
        module_type_store = module_type_store.open_function_context('new_func', 8, 4, False)
        
        # Passed parameters checking function
        new_func.stypy_localization = localization
        new_func.stypy_type_of_self = None
        new_func.stypy_type_store = module_type_store
        new_func.stypy_function_name = 'new_func'
        new_func.stypy_param_names_list = []
        new_func.stypy_varargs_param_name = None
        new_func.stypy_kwargs_param_name = None
        new_func.stypy_call_defaults = defaults
        new_func.stypy_call_varargs = varargs
        new_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_func(...)' code ##################

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'new function')
        # Assigning a type to the variable 'stypy_return_type' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_2)
        
        # ################# End of 'new_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_func' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_func'
        return stypy_return_type_3

    # Assigning a type to the variable 'new_func' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'new_func', new_func)
    
    # Call to setattr(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'math' (line 12)
    math_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'math', False)
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', 'cos')
    # Getting the type of 'new_func' (line 12)
    new_func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'new_func', False)
    # Processing the call keyword arguments (line 12)
    kwargs_8 = {}
    # Getting the type of 'setattr' (line 12)
    setattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 12)
    setattr_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), setattr_4, *[math_5, str_6, new_func_7], **kwargs_8)
    
    
    # Call to setattr(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'math' (line 13)
    math_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'math', False)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'str', 'pi')
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'str')
    # Processing the call keyword arguments (line 13)
    kwargs_14 = {}
    # Getting the type of 'setattr' (line 13)
    setattr_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 13)
    setattr_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), setattr_10, *[math_11, str_12, str_13], **kwargs_14)
    
    
    # Call to cos(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_18 = {}
    # Getting the type of 'math' (line 15)
    math_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 15)
    cos_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), math_16, 'cos')
    # Calling cos(args, kwargs) (line 15)
    cos_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), cos_17, *[], **kwargs_18)
    
    
    # Call to len(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'math' (line 16)
    math_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'math', False)
    # Obtaining the member 'pi' of a type (line 16)
    pi_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), math_21, 'pi')
    # Processing the call keyword arguments (line 16)
    kwargs_23 = {}
    # Getting the type of 'len' (line 16)
    len_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'len', False)
    # Calling len(args, kwargs) (line 16)
    len_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), len_20, *[pi_22], **kwargs_23)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
