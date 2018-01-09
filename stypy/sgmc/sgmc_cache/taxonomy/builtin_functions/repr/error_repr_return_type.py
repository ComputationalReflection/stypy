
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "repr builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__repr__) -> <type 'str'>
7:     class Sample:
8:         def __repr__(self):
9:             return "str"
10: 
11: 
12:     # Call the builtin
13:     ret = repr(Sample())
14: 
15:     # Type error
16:     ret.unexisting_method()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'repr builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__repr__'
            module_type_store = module_type_store.open_function_context('__repr__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Sample.__repr__')
            Sample.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__repr__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__repr__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__repr__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__repr__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__repr__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__repr__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Sample' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to repr(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to Sample(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 13)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 13)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), Sample_5, *[], **kwargs_6)
    
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'repr' (line 13)
    repr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'repr', False)
    # Calling repr(args, kwargs) (line 13)
    repr_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), repr_4, *[Sample_call_result_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', repr_call_result_9)
    
    # Call to unexisting_method(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_12 = {}
    # Getting the type of 'ret' (line 16)
    ret_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 16)
    unexisting_method_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), ret_10, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 16)
    unexisting_method_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), unexisting_method_11, *[], **kwargs_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
