
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive type change"
3: 
4: if __name__ == '__main__':
5: 
6:     class DummyOrigin:
7:         class_attribute = 0
8: 
9:         def __init__(self):
10:             self.instance_attribute = "str"
11: 
12:         def method(self):
13:             return self.instance_attribute
14: 
15: 
16:     class DummyDestination:
17:         class_attribute2 = "str"
18: 
19:         def __init__(self):
20:             self.instance_attribute2 = 0
21: 
22:         def method(self):
23:             return self.instance_attribute2
24: 
25: 
26:     class DummyDestination2:
27:         class_attribute = float
28: 
29:         def __init__(self):
30:             self.instance_attribute = float
31: 
32:         def method(self):
33:             return self.instance_attribute2
34: 
35: 
36:     d = DummyOrigin()
37: 
38:     print d.class_attribute / 2
39:     print d.instance_attribute + "str"
40: 
41:     if True:
42:         d.__class__ = DummyDestination
43:     else:
44:         d.__class__ = DummyDestination2
45: 
46:     # Type warning
47:     print d.class_attribute / 2
48: 
49:     print d.instance_attribute + "str"
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive type change')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DummyOrigin' class

    class DummyOrigin:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyOrigin.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Str to a Attribute (line 10):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'str', 'str')
            # Getting the type of 'self' (line 10)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 10)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), self_3, 'instance_attribute', str_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyOrigin.method.__dict__.__setitem__('stypy_localization', localization)
            DummyOrigin.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyOrigin.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyOrigin.method.__dict__.__setitem__('stypy_function_name', 'DummyOrigin.method')
            DummyOrigin.method.__dict__.__setitem__('stypy_param_names_list', [])
            DummyOrigin.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyOrigin.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyOrigin.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyOrigin.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyOrigin.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyOrigin.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyOrigin.method', [], None, None, defaults, varargs, kwargs)

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

            # Getting the type of 'self' (line 13)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 13)
            instance_attribute_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), self_4, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', instance_attribute_5)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'DummyOrigin' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'DummyOrigin', DummyOrigin)
    
    # Assigning a Num to a Name (line 7):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
    # Getting the type of 'DummyOrigin'
    DummyOrigin_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyOrigin')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyOrigin_8, 'class_attribute', int_7)
    # Declaration of the 'DummyDestination' class

    class DummyDestination:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 19, 8, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Num to a Attribute (line 20):
            int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 39), 'int')
            # Getting the type of 'self' (line 20)
            self_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'self')
            # Setting the type of the member 'instance_attribute2' of a type (line 20)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), self_10, 'instance_attribute2', int_9)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 22, 8, False)
            # Assigning a type to the variable 'self' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination.method.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination.method.__dict__.__setitem__('stypy_function_name', 'DummyDestination.method')
            DummyDestination.method.__dict__.__setitem__('stypy_param_names_list', [])
            DummyDestination.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.method', [], None, None, defaults, varargs, kwargs)

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

            # Getting the type of 'self' (line 23)
            self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'self')
            # Obtaining the member 'instance_attribute2' of a type (line 23)
            instance_attribute2_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), self_11, 'instance_attribute2')
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'stypy_return_type', instance_attribute2_12)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 22)
            stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_13

    
    # Assigning a type to the variable 'DummyDestination' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'DummyDestination', DummyDestination)
    
    # Assigning a Str to a Name (line 17):
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'str', 'str')
    # Getting the type of 'DummyDestination'
    DummyDestination_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination')
    # Setting the type of the member 'class_attribute2' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination_15, 'class_attribute2', str_14)
    # Declaration of the 'DummyDestination2' class

    class DummyDestination2:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 29, 8, False)
            # Assigning a type to the variable 'self' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination2.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Name to a Attribute (line 30):
            # Getting the type of 'float' (line 30)
            float_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 38), 'float')
            # Getting the type of 'self' (line 30)
            self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 30)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 12), self_17, 'instance_attribute', float_16)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 32, 8, False)
            # Assigning a type to the variable 'self' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination2.method.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination2.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination2.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination2.method.__dict__.__setitem__('stypy_function_name', 'DummyDestination2.method')
            DummyDestination2.method.__dict__.__setitem__('stypy_param_names_list', [])
            DummyDestination2.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination2.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination2.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination2.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination2.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination2.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination2.method', [], None, None, defaults, varargs, kwargs)

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

            # Getting the type of 'self' (line 33)
            self_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'self')
            # Obtaining the member 'instance_attribute2' of a type (line 33)
            instance_attribute2_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), self_18, 'instance_attribute2')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', instance_attribute2_19)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 32)
            stypy_return_type_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_20

    
    # Assigning a type to the variable 'DummyDestination2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'DummyDestination2', DummyDestination2)
    
    # Assigning a Name to a Name (line 27):
    # Getting the type of 'float' (line 27)
    float_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'float')
    # Getting the type of 'DummyDestination2'
    DummyDestination2_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination2')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination2_22, 'class_attribute', float_21)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to DummyOrigin(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_24 = {}
    # Getting the type of 'DummyOrigin' (line 36)
    DummyOrigin_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'DummyOrigin', False)
    # Calling DummyOrigin(args, kwargs) (line 36)
    DummyOrigin_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), DummyOrigin_23, *[], **kwargs_24)
    
    # Assigning a type to the variable 'd' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'd', DummyOrigin_call_result_25)
    # Getting the type of 'd' (line 38)
    d_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 38)
    class_attribute_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), d_26, 'class_attribute')
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
    # Applying the binary operator 'div' (line 38)
    result_div_29 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 10), 'div', class_attribute_27, int_28)
    
    # Getting the type of 'd' (line 39)
    d_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd')
    # Obtaining the member 'instance_attribute' of a type (line 39)
    instance_attribute_31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d_30, 'instance_attribute')
    str_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'str', 'str')
    # Applying the binary operator '+' (line 39)
    result_add_33 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), '+', instance_attribute_31, str_32)
    
    
    # Getting the type of 'True' (line 41)
    True_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'True')
    # Testing the type of an if condition (line 41)
    if_condition_35 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), True_34)
    # Assigning a type to the variable 'if_condition_35' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_35', if_condition_35)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 42):
    # Getting the type of 'DummyDestination' (line 42)
    DummyDestination_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'DummyDestination')
    # Getting the type of 'd' (line 42)
    d_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'd')
    # Setting the type of the member '__class__' of a type (line 42)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), d_37, '__class__', DummyDestination_36)
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Attribute (line 44):
    # Getting the type of 'DummyDestination2' (line 44)
    DummyDestination2_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'DummyDestination2')
    # Getting the type of 'd' (line 44)
    d_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'd')
    # Setting the type of the member '__class__' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), d_39, '__class__', DummyDestination2_38)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 47)
    d_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 47)
    class_attribute_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 10), d_40, 'class_attribute')
    int_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'int')
    # Applying the binary operator 'div' (line 47)
    result_div_43 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 10), 'div', class_attribute_41, int_42)
    
    # Getting the type of 'd' (line 49)
    d_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'd')
    # Obtaining the member 'instance_attribute' of a type (line 49)
    instance_attribute_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 10), d_44, 'instance_attribute')
    str_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 33), 'str', 'str')
    # Applying the binary operator '+' (line 49)
    result_add_47 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), '+', instance_attribute_45, str_46)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
