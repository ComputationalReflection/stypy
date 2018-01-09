
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive class modification"
3: 
4: if __name__ == '__main__':
5: 
6:     class Dummy:
7:         att = 3
8: 
9:         def method(self):
10:             return 0
11: 
12: 
13:     c = Dummy()
14: 
15:     if True:
16:         class Dummy:
17:             attIf = 3
18: 
19:             def methodIf(self):
20:                 return 0
21: 
22: 
23:         c = Dummy()
24: 
25:     # Type warning
26:     print c.att
27:     # Type warning
28:     print c.att
29:     # Type warning
30:     print c.methodIf()
31:     # Type error
32:     print c.methodElse()
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive class modification')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Dummy.method.__dict__.__setitem__('stypy_localization', localization)
            Dummy.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Dummy.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            Dummy.method.__dict__.__setitem__('stypy_function_name', 'Dummy.method')
            Dummy.method.__dict__.__setitem__('stypy_param_names_list', [])
            Dummy.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            Dummy.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Dummy.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            Dummy.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            Dummy.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Dummy.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.method', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'method', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'method(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type', int_2)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 9)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Dummy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 7):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
    # Getting the type of 'Dummy'
    Dummy_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'att' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_5, 'att', int_4)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to Dummy(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'Dummy' (line 13)
    Dummy_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 13)
    Dummy_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), Dummy_6, *[], **kwargs_7)
    
    # Assigning a type to the variable 'c' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'c', Dummy_call_result_8)
    
    # Getting the type of 'True' (line 15)
    True_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'True')
    # Testing the type of an if condition (line 15)
    if_condition_10 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), True_9)
    # Assigning a type to the variable 'if_condition_10' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_10', if_condition_10)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def methodIf(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'methodIf'
            module_type_store = module_type_store.open_function_context('methodIf', 19, 12, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            Dummy.methodIf.__dict__.__setitem__('stypy_localization', localization)
            Dummy.methodIf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Dummy.methodIf.__dict__.__setitem__('stypy_type_store', module_type_store)
            Dummy.methodIf.__dict__.__setitem__('stypy_function_name', 'Dummy.methodIf')
            Dummy.methodIf.__dict__.__setitem__('stypy_param_names_list', [])
            Dummy.methodIf.__dict__.__setitem__('stypy_varargs_param_name', None)
            Dummy.methodIf.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Dummy.methodIf.__dict__.__setitem__('stypy_call_defaults', defaults)
            Dummy.methodIf.__dict__.__setitem__('stypy_call_varargs', varargs)
            Dummy.methodIf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Dummy.methodIf.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.methodIf', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'methodIf', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'methodIf(...)' code ##################

            int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'stypy_return_type', int_11)
            
            # ################# End of 'methodIf(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'methodIf' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_12)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'methodIf'
            return stypy_return_type_12


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Dummy' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 17):
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    # Getting the type of 'Dummy'
    Dummy_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'attIf' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_14, 'attIf', int_13)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to Dummy(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_16 = {}
    # Getting the type of 'Dummy' (line 23)
    Dummy_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 23)
    Dummy_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), Dummy_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'c' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'c', Dummy_call_result_17)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c' (line 26)
    c_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'c')
    # Obtaining the member 'att' of a type (line 26)
    att_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), c_18, 'att')
    # Getting the type of 'c' (line 28)
    c_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'c')
    # Obtaining the member 'att' of a type (line 28)
    att_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 10), c_20, 'att')
    
    # Call to methodIf(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_24 = {}
    # Getting the type of 'c' (line 30)
    c_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'c', False)
    # Obtaining the member 'methodIf' of a type (line 30)
    methodIf_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 10), c_22, 'methodIf')
    # Calling methodIf(args, kwargs) (line 30)
    methodIf_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), methodIf_23, *[], **kwargs_24)
    
    
    # Call to methodElse(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_28 = {}
    # Getting the type of 'c' (line 32)
    c_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'c', False)
    # Obtaining the member 'methodElse' of a type (line 32)
    methodElse_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 10), c_26, 'methodElse')
    # Calling methodElse(args, kwargs) (line 32)
    methodElse_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), methodElse_27, *[], **kwargs_28)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
