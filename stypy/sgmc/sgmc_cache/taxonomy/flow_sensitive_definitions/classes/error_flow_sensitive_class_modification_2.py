
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive class modification"
3: 
4: if __name__ == '__main__':
5: 
6:     if True:
7:         class Dummy:
8:             attIf = 3
9: 
10:             def methodIf(self):
11:                 return 0
12:     else:
13:         class Dummy:
14:             attElse = 3
15: 
16:             def methodElse(self):
17:                 return 0
18: 
19:     c = Dummy()
20: 
21:     # Type warning
22:     print c.attIf
23:     # Type warning
24:     print c.attElse
25:     # Type warning
26:     print c.methodIf()
27:     # Type warning
28:     print c.methodElse()
29: 

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
    
    # Getting the type of 'True' (line 6)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'True')
    # Testing the type of an if condition (line 6)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def methodIf(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'methodIf'
            module_type_store = module_type_store.open_function_context('methodIf', 10, 12, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'self', type_of_self)
            
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

            int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'stypy_return_type', int_4)
            
            # ################# End of 'methodIf(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'methodIf' in the type store
            # Getting the type of 'stypy_return_type' (line 10)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 7, 8, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Dummy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 8):
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
    # Getting the type of 'Dummy'
    Dummy_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'attIf' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_7, 'attIf', int_6)
    # SSA branch for the else part of an if statement (line 6)
    module_type_store.open_ssa_branch('else')
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def methodElse(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'methodElse'
            module_type_store = module_type_store.open_function_context('methodElse', 16, 12, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'self', type_of_self)
            
            # Passed parameters checking function
            Dummy.methodElse.__dict__.__setitem__('stypy_localization', localization)
            Dummy.methodElse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Dummy.methodElse.__dict__.__setitem__('stypy_type_store', module_type_store)
            Dummy.methodElse.__dict__.__setitem__('stypy_function_name', 'Dummy.methodElse')
            Dummy.methodElse.__dict__.__setitem__('stypy_param_names_list', [])
            Dummy.methodElse.__dict__.__setitem__('stypy_varargs_param_name', None)
            Dummy.methodElse.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Dummy.methodElse.__dict__.__setitem__('stypy_call_defaults', defaults)
            Dummy.methodElse.__dict__.__setitem__('stypy_call_varargs', varargs)
            Dummy.methodElse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Dummy.methodElse.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.methodElse', [], None, None, defaults, varargs, kwargs)

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

            int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'stypy_return_type', int_8)
            
            # ################# End of 'methodElse(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'methodElse' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_9)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'methodElse'
            return stypy_return_type_9


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 13, 8, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Dummy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 14):
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
    # Getting the type of 'Dummy'
    Dummy_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'attElse' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_11, 'attElse', int_10)
    # SSA join for if statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 19):
    
    # Call to Dummy(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_13 = {}
    # Getting the type of 'Dummy' (line 19)
    Dummy_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 19)
    Dummy_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), Dummy_12, *[], **kwargs_13)
    
    # Assigning a type to the variable 'c' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'c', Dummy_call_result_14)
    # Getting the type of 'c' (line 22)
    c_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'c')
    # Obtaining the member 'attIf' of a type (line 22)
    attIf_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), c_15, 'attIf')
    # Getting the type of 'c' (line 24)
    c_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'c')
    # Obtaining the member 'attElse' of a type (line 24)
    attElse_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 10), c_17, 'attElse')
    
    # Call to methodIf(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_21 = {}
    # Getting the type of 'c' (line 26)
    c_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'c', False)
    # Obtaining the member 'methodIf' of a type (line 26)
    methodIf_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), c_19, 'methodIf')
    # Calling methodIf(args, kwargs) (line 26)
    methodIf_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), methodIf_20, *[], **kwargs_21)
    
    
    # Call to methodElse(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_25 = {}
    # Getting the type of 'c' (line 28)
    c_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'c', False)
    # Obtaining the member 'methodElse' of a type (line 28)
    methodElse_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 10), c_23, 'methodElse')
    # Calling methodElse(args, kwargs) (line 28)
    methodElse_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), methodElse_24, *[], **kwargs_25)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
