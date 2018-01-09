
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "abs builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type bool>) -> <type 'int'>
7:     # (<type complex>) -> <type 'float'>
8:     # (Number) -> TypeOfParam(1)
9:     # (Overloads__abs__) -> <type 'int'>
10:     class Sample:
11:         def __abs__(self):
12:             return 4
13: 
14: 
15:     # Type error
16:     ret = abs(Sample)
17: 
18:     # Type error
19:     ret2 = abs(bool)
20:     # Type error
21:     ret3 = abs(int)
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'abs builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __abs__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__abs__'
            module_type_store = module_type_store.open_function_context('__abs__', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__abs__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__abs__.__dict__.__setitem__('stypy_function_name', 'Sample.__abs__')
            Sample.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__abs__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__abs__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__abs__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__abs__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__abs__' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__abs__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to abs(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'Sample' (line 16)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'Sample', False)
    # Processing the call keyword arguments (line 16)
    kwargs_6 = {}
    # Getting the type of 'abs' (line 16)
    abs_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 16)
    abs_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), abs_4, *[Sample_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', abs_call_result_7)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to abs(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'bool' (line 19)
    bool_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'bool', False)
    # Processing the call keyword arguments (line 19)
    kwargs_10 = {}
    # Getting the type of 'abs' (line 19)
    abs_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 19)
    abs_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), abs_8, *[bool_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret2' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret2', abs_call_result_11)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to abs(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'int' (line 21)
    int_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'int', False)
    # Processing the call keyword arguments (line 21)
    kwargs_14 = {}
    # Getting the type of 'abs' (line 21)
    abs_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'abs', False)
    # Calling abs(args, kwargs) (line 21)
    abs_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), abs_12, *[int_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret3' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret3', abs_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
