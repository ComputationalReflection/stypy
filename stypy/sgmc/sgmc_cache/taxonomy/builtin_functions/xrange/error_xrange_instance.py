
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "range builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <built-in function range>
7:     # (Overloads__trunc__) -> <built-in function range>
8:     # (Integer, Integer) -> <built-in function range>
9:     # (Overloads__trunc__, Integer) -> <built-in function range>
10:     # (Integer, Overloads__trunc__) -> <built-in function range>
11:     # (Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
12:     # (Integer, Integer, Integer) -> <built-in function range>
13:     # (Overloads__trunc__, Integer, Integer) -> <built-in function range>
14:     # (Integer, Overloads__trunc__, Integer) -> <built-in function range>
15:     # (Integer, Integer, Overloads__trunc__) -> <built-in function range>
16:     # (Integer, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
17:     # (Overloads__trunc__, Overloads__trunc__, Integer) -> <built-in function range>
18:     # (Overloads__trunc__, Integer, Overloads__trunc__) -> <built-in function range>
19:     # (Overloads__trunc__, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
20: 
21: 
22: 
23:     class Sample:
24:         def __trunc__(self):
25:             return 4
26: 
27: 
28:     # Type error
29:     ret = xrange(int, int)
30:     # Type error
31:     ret = xrange(Sample, Sample)
32:     # Type error
33:     ret = xrange(Sample, Sample, 4)
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'range builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 24, 8, False)
            # Assigning a type to the variable 'self' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__trunc__.__dict__.__setitem__('stypy_function_name', 'Sample.__trunc__')
            Sample.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__trunc__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 24)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
            # Assigning a type to the variable 'self' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to xrange(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'int' (line 29)
    int_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'int', False)
    # Getting the type of 'int' (line 29)
    int_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'int', False)
    # Processing the call keyword arguments (line 29)
    kwargs_7 = {}
    # Getting the type of 'xrange' (line 29)
    xrange_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 29)
    xrange_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), xrange_4, *[int_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', xrange_call_result_8)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to xrange(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'Sample' (line 31)
    Sample_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'Sample', False)
    # Getting the type of 'Sample' (line 31)
    Sample_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'Sample', False)
    # Processing the call keyword arguments (line 31)
    kwargs_12 = {}
    # Getting the type of 'xrange' (line 31)
    xrange_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 31)
    xrange_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), xrange_9, *[Sample_10, Sample_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ret', xrange_call_result_13)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to xrange(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'Sample' (line 33)
    Sample_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'Sample', False)
    # Getting the type of 'Sample' (line 33)
    Sample_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'Sample', False)
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_18 = {}
    # Getting the type of 'xrange' (line 33)
    xrange_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 33)
    xrange_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), xrange_14, *[Sample_15, Sample_16, int_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ret', xrange_call_result_19)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
