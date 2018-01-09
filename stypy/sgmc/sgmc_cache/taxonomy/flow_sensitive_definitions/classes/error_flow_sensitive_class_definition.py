
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive class definition"
3: 
4: if __name__ == '__main__':
5:     if True:
6:         class DummyIf:
7:             attIf = 3
8: 
9:             def methodIf(self):
10:                 return 0
11: 
12: 
13:         c = DummyIf()
14:     else:
15:         class DummyElse:
16:             attElse = 3
17: 
18:             def methodElse(self):
19:                 return 0
20: 
21: 
22:         c = DummyElse()
23: 
24:     # Type warning
25:     print c.attIf
26:     # Type warning
27:     print c.attElse
28:     # Type warning
29:     print c.methodIf()
30:     # Type warning
31:     print c.methodElse()
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive class definition')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'True' (line 5)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'True')
    # Testing the type of an if condition (line 5)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 5)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Declaration of the 'DummyIf' class

    class DummyIf:

        @norecursion
        def methodIf(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'methodIf'
            module_type_store = module_type_store.open_function_context('methodIf', 9, 12, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyIf.methodIf.__dict__.__setitem__('stypy_localization', localization)
            DummyIf.methodIf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyIf.methodIf.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyIf.methodIf.__dict__.__setitem__('stypy_function_name', 'DummyIf.methodIf')
            DummyIf.methodIf.__dict__.__setitem__('stypy_param_names_list', [])
            DummyIf.methodIf.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyIf.methodIf.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyIf.methodIf.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyIf.methodIf.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyIf.methodIf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyIf.methodIf.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyIf.methodIf', [], None, None, defaults, varargs, kwargs)

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

            int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'stypy_return_type', int_4)
            
            # ################# End of 'methodIf(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'methodIf' in the type store
            # Getting the type of 'stypy_return_type' (line 9)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'methodIf'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyIf.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DummyIf' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'DummyIf', DummyIf)
    
    # Assigning a Num to a Name (line 7):
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
    # Getting the type of 'DummyIf'
    DummyIf_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyIf')
    # Setting the type of the member 'attIf' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyIf_7, 'attIf', int_6)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to DummyIf(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_9 = {}
    # Getting the type of 'DummyIf' (line 13)
    DummyIf_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'DummyIf', False)
    # Calling DummyIf(args, kwargs) (line 13)
    DummyIf_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), DummyIf_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 'c' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'c', DummyIf_call_result_10)
    # SSA branch for the else part of an if statement (line 5)
    module_type_store.open_ssa_branch('else')
    # Declaration of the 'DummyElse' class

    class DummyElse:

        @norecursion
        def methodElse(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'methodElse'
            module_type_store = module_type_store.open_function_context('methodElse', 18, 12, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyElse.methodElse.__dict__.__setitem__('stypy_localization', localization)
            DummyElse.methodElse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyElse.methodElse.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyElse.methodElse.__dict__.__setitem__('stypy_function_name', 'DummyElse.methodElse')
            DummyElse.methodElse.__dict__.__setitem__('stypy_param_names_list', [])
            DummyElse.methodElse.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyElse.methodElse.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyElse.methodElse.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyElse.methodElse.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyElse.methodElse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyElse.methodElse.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyElse.methodElse', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'methodElse', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'methodElse(...)' code ##################

            int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'stypy_return_type', int_11)
            
            # ################# End of 'methodElse(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'methodElse' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_12)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'methodElse'
            return stypy_return_type_12


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 15, 8, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyElse.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DummyElse' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'DummyElse', DummyElse)
    
    # Assigning a Num to a Name (line 16):
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'int')
    # Getting the type of 'DummyElse'
    DummyElse_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyElse')
    # Setting the type of the member 'attElse' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyElse_14, 'attElse', int_13)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to DummyElse(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_16 = {}
    # Getting the type of 'DummyElse' (line 22)
    DummyElse_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'DummyElse', False)
    # Calling DummyElse(args, kwargs) (line 22)
    DummyElse_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), DummyElse_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'c' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'c', DummyElse_call_result_17)
    # SSA join for if statement (line 5)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c' (line 25)
    c_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'c')
    # Obtaining the member 'attIf' of a type (line 25)
    attIf_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), c_18, 'attIf')
    # Getting the type of 'c' (line 27)
    c_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'c')
    # Obtaining the member 'attElse' of a type (line 27)
    attElse_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), c_20, 'attElse')
    
    # Call to methodIf(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_24 = {}
    # Getting the type of 'c' (line 29)
    c_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'c', False)
    # Obtaining the member 'methodIf' of a type (line 29)
    methodIf_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 10), c_22, 'methodIf')
    # Calling methodIf(args, kwargs) (line 29)
    methodIf_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), methodIf_23, *[], **kwargs_24)
    
    
    # Call to methodElse(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_28 = {}
    # Getting the type of 'c' (line 31)
    c_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'c', False)
    # Obtaining the member 'methodElse' of a type (line 31)
    methodElse_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), c_26, 'methodElse')
    # Calling methodElse(args, kwargs) (line 31)
    methodElse_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), methodElse_27, *[], **kwargs_28)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
