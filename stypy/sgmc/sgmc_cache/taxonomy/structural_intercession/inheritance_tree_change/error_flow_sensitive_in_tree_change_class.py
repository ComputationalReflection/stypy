
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive inheritance tree change"
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
17:         parent_class_attribute = "str"
18: 
19:         def __init__(self):
20:             self.instance_attribute = 0
21: 
22:         def parent_method(self):
23:             return self.instance_attribute
24: 
25: 
26:     class DummyDestination2:
27:         parent_class_attribute = 3.4
28:         one_parent_only = list()
29: 
30:         def __init__(self):
31:             self.instance_attribute = float
32: 
33:         def parent_method(self, param):
34:             return param
35: 
36: 
37:     d = DummyOrigin()
38: 
39:     print d.class_attribute / 2
40:     print d.instance_attribute + "str"
41: 
42:     if True:
43:         DummyOrigin.__bases__ = (DummyDestination,)
44:     else:
45:         DummyOrigin.__bases__ = (DummyDestination2,)
46: 
47:     # Type warning
48:     print d.parent_class_attribute / 2
49:     # Type warning
50:     print len(d.one_parent_only)
51: 
52:     # Type warning
53:     print d.parent_method()
54:     # Type warning
55:     print d.parent_method(3)
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive inheritance tree change')
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
            int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'int')
            # Getting the type of 'self' (line 20)
            self_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 20)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), self_10, 'instance_attribute', int_9)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def parent_method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'parent_method'
            module_type_store = module_type_store.open_function_context('parent_method', 22, 8, False)
            # Assigning a type to the variable 'self' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination.parent_method.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_function_name', 'DummyDestination.parent_method')
            DummyDestination.parent_method.__dict__.__setitem__('stypy_param_names_list', [])
            DummyDestination.parent_method.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination.parent_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.parent_method', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'parent_method', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'parent_method(...)' code ##################

            # Getting the type of 'self' (line 23)
            self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 23)
            instance_attribute_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), self_11, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'stypy_return_type', instance_attribute_12)
            
            # ################# End of 'parent_method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'parent_method' in the type store
            # Getting the type of 'stypy_return_type' (line 22)
            stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'parent_method'
            return stypy_return_type_13

    
    # Assigning a type to the variable 'DummyDestination' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'DummyDestination', DummyDestination)
    
    # Assigning a Str to a Name (line 17):
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'str', 'str')
    # Getting the type of 'DummyDestination'
    DummyDestination_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination')
    # Setting the type of the member 'parent_class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination_15, 'parent_class_attribute', str_14)
    # Declaration of the 'DummyDestination2' class

    class DummyDestination2:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 30, 8, False)
            # Assigning a type to the variable 'self' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self', type_of_self)
            
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

            
            # Assigning a Name to a Attribute (line 31):
            # Getting the type of 'float' (line 31)
            float_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'float')
            # Getting the type of 'self' (line 31)
            self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), self_17, 'instance_attribute', float_16)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def parent_method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'parent_method'
            module_type_store = module_type_store.open_function_context('parent_method', 33, 8, False)
            # Assigning a type to the variable 'self' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_function_name', 'DummyDestination2.parent_method')
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_param_names_list', ['param'])
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination2.parent_method.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination2.parent_method', ['param'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'parent_method', localization, ['param'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'parent_method(...)' code ##################

            # Getting the type of 'param' (line 34)
            param_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'param')
            # Assigning a type to the variable 'stypy_return_type' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', param_18)
            
            # ################# End of 'parent_method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'parent_method' in the type store
            # Getting the type of 'stypy_return_type' (line 33)
            stypy_return_type_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'parent_method'
            return stypy_return_type_19

    
    # Assigning a type to the variable 'DummyDestination2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'DummyDestination2', DummyDestination2)
    
    # Assigning a Num to a Name (line 27):
    float_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 33), 'float')
    # Getting the type of 'DummyDestination2'
    DummyDestination2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination2')
    # Setting the type of the member 'parent_class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination2_21, 'parent_class_attribute', float_20)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to list(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_23 = {}
    # Getting the type of 'list' (line 28)
    list_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'list', False)
    # Calling list(args, kwargs) (line 28)
    list_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 28, 26), list_22, *[], **kwargs_23)
    
    # Getting the type of 'DummyDestination2'
    DummyDestination2_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination2')
    # Setting the type of the member 'one_parent_only' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination2_25, 'one_parent_only', list_call_result_24)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to DummyOrigin(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_27 = {}
    # Getting the type of 'DummyOrigin' (line 37)
    DummyOrigin_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'DummyOrigin', False)
    # Calling DummyOrigin(args, kwargs) (line 37)
    DummyOrigin_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), DummyOrigin_26, *[], **kwargs_27)
    
    # Assigning a type to the variable 'd' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'd', DummyOrigin_call_result_28)
    # Getting the type of 'd' (line 39)
    d_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 39)
    class_attribute_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d_29, 'class_attribute')
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
    # Applying the binary operator 'div' (line 39)
    result_div_32 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), 'div', class_attribute_30, int_31)
    
    # Getting the type of 'd' (line 40)
    d_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'd')
    # Obtaining the member 'instance_attribute' of a type (line 40)
    instance_attribute_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 10), d_33, 'instance_attribute')
    str_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'str', 'str')
    # Applying the binary operator '+' (line 40)
    result_add_36 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 10), '+', instance_attribute_34, str_35)
    
    
    # Getting the type of 'True' (line 42)
    True_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'True')
    # Testing the type of an if condition (line 42)
    if_condition_38 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), True_37)
    # Assigning a type to the variable 'if_condition_38' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_38', if_condition_38)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Attribute (line 43):
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    # Getting the type of 'DummyDestination' (line 43)
    DummyDestination_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'DummyDestination')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 33), tuple_39, DummyDestination_40)
    
    # Getting the type of 'DummyOrigin' (line 43)
    DummyOrigin_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'DummyOrigin')
    # Setting the type of the member '__bases__' of a type (line 43)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), DummyOrigin_41, '__bases__', tuple_39)
    # SSA branch for the else part of an if statement (line 42)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Attribute (line 45):
    
    # Obtaining an instance of the builtin type 'tuple' (line 45)
    tuple_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 45)
    # Adding element type (line 45)
    # Getting the type of 'DummyDestination2' (line 45)
    DummyDestination2_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'DummyDestination2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 33), tuple_42, DummyDestination2_43)
    
    # Getting the type of 'DummyOrigin' (line 45)
    DummyOrigin_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'DummyOrigin')
    # Setting the type of the member '__bases__' of a type (line 45)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), DummyOrigin_44, '__bases__', tuple_42)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 48)
    d_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'd')
    # Obtaining the member 'parent_class_attribute' of a type (line 48)
    parent_class_attribute_46 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 10), d_45, 'parent_class_attribute')
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 37), 'int')
    # Applying the binary operator 'div' (line 48)
    result_div_48 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 10), 'div', parent_class_attribute_46, int_47)
    
    
    # Call to len(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'd' (line 50)
    d_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'd', False)
    # Obtaining the member 'one_parent_only' of a type (line 50)
    one_parent_only_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 14), d_50, 'one_parent_only')
    # Processing the call keyword arguments (line 50)
    kwargs_52 = {}
    # Getting the type of 'len' (line 50)
    len_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 10), 'len', False)
    # Calling len(args, kwargs) (line 50)
    len_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 50, 10), len_49, *[one_parent_only_51], **kwargs_52)
    
    
    # Call to parent_method(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_56 = {}
    # Getting the type of 'd' (line 53)
    d_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 10), 'd', False)
    # Obtaining the member 'parent_method' of a type (line 53)
    parent_method_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 10), d_54, 'parent_method')
    # Calling parent_method(args, kwargs) (line 53)
    parent_method_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 53, 10), parent_method_55, *[], **kwargs_56)
    
    
    # Call to parent_method(...): (line 55)
    # Processing the call arguments (line 55)
    int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_61 = {}
    # Getting the type of 'd' (line 55)
    d_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 10), 'd', False)
    # Obtaining the member 'parent_method' of a type (line 55)
    parent_method_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 10), d_58, 'parent_method')
    # Calling parent_method(args, kwargs) (line 55)
    parent_method_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 55, 10), parent_method_59, *[int_60], **kwargs_61)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
