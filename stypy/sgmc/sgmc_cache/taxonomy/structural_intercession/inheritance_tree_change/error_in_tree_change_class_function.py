
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Change the inheritance tree of a user class inside a function"
3: 
4: if __name__ == '__main__':
5:     class DummyOrigin:
6:         class_attribute = 0
7: 
8:         def __init__(self):
9:             self.instance_attribute = "str"
10: 
11:         def method(self):
12:             return self.instance_attribute
13: 
14: 
15:     class DummyDestination:
16:         class_attribute2 = "str"
17: 
18:         def __init__(self):
19:             self.instance_attribute2 = 0
20: 
21:         def method2(self):
22:             return self.class_attribute2
23: 
24: 
25:     d = DummyOrigin()
26: 
27:     print d.class_attribute / 2
28:     print d.instance_attribute + "str"
29: 
30: 
31:     def func():
32:         DummyOrigin.__bases__ = (DummyDestination,)
33: 
34: 
35:     func()
36:     print d.class_attribute / 2
37:     print d.instance_attribute + "str"
38: 
39:     print d.class_attribute2 + "str"
40:     # Type error
41:     print d.instance_attribute2 / 2
42: 
43:     print d.method2()
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Change the inheritance tree of a user class inside a function')
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
            module_type_store = module_type_store.open_function_context('__init__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
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

            
            # Assigning a Str to a Attribute (line 9):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 38), 'str', 'str')
            # Getting the type of 'self' (line 9)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 9)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), self_3, 'instance_attribute', str_2)
            
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
            module_type_store = module_type_store.open_function_context('method', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
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

            # Getting the type of 'self' (line 12)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 12)
            instance_attribute_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), self_4, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', instance_attribute_5)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'DummyOrigin' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'DummyOrigin', DummyOrigin)
    
    # Assigning a Num to a Name (line 6):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
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
            module_type_store = module_type_store.open_function_context('__init__', 18, 8, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', type_of_self)
            
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

            
            # Assigning a Num to a Attribute (line 19):
            int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 39), 'int')
            # Getting the type of 'self' (line 19)
            self_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'self')
            # Setting the type of the member 'instance_attribute2' of a type (line 19)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), self_10, 'instance_attribute2', int_9)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method2(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method2'
            module_type_store = module_type_store.open_function_context('method2', 21, 8, False)
            # Assigning a type to the variable 'self' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination.method2.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination.method2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination.method2.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination.method2.__dict__.__setitem__('stypy_function_name', 'DummyDestination.method2')
            DummyDestination.method2.__dict__.__setitem__('stypy_param_names_list', [])
            DummyDestination.method2.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination.method2.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination.method2.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.method2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'method2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'method2(...)' code ##################

            # Getting the type of 'self' (line 22)
            self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'self')
            # Obtaining the member 'class_attribute2' of a type (line 22)
            class_attribute2_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 19), self_11, 'class_attribute2')
            # Assigning a type to the variable 'stypy_return_type' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', class_attribute2_12)
            
            # ################# End of 'method2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method2' in the type store
            # Getting the type of 'stypy_return_type' (line 21)
            stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method2'
            return stypy_return_type_13

    
    # Assigning a type to the variable 'DummyDestination' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'DummyDestination', DummyDestination)
    
    # Assigning a Str to a Name (line 16):
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'str', 'str')
    # Getting the type of 'DummyDestination'
    DummyDestination_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination')
    # Setting the type of the member 'class_attribute2' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination_15, 'class_attribute2', str_14)
    
    # Assigning a Call to a Name (line 25):
    
    # Call to DummyOrigin(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_17 = {}
    # Getting the type of 'DummyOrigin' (line 25)
    DummyOrigin_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'DummyOrigin', False)
    # Calling DummyOrigin(args, kwargs) (line 25)
    DummyOrigin_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), DummyOrigin_16, *[], **kwargs_17)
    
    # Assigning a type to the variable 'd' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'd', DummyOrigin_call_result_18)
    # Getting the type of 'd' (line 27)
    d_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 27)
    class_attribute_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), d_19, 'class_attribute')
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_22 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 10), 'div', class_attribute_20, int_21)
    
    # Getting the type of 'd' (line 28)
    d_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'd')
    # Obtaining the member 'instance_attribute' of a type (line 28)
    instance_attribute_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 10), d_23, 'instance_attribute')
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 33), 'str', 'str')
    # Applying the binary operator '+' (line 28)
    result_add_26 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 10), '+', instance_attribute_24, str_25)
    

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 31, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Assigning a Tuple to a Attribute (line 32):
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        # Getting the type of 'DummyDestination' (line 32)
        DummyDestination_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'DummyDestination')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 33), tuple_27, DummyDestination_28)
        
        # Getting the type of 'DummyOrigin' (line 32)
        DummyOrigin_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'DummyOrigin')
        # Setting the type of the member '__bases__' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), DummyOrigin_29, '__bases__', tuple_27)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_30

    # Assigning a type to the variable 'func' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'func', func)
    
    # Call to func(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_32 = {}
    # Getting the type of 'func' (line 35)
    func_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'func', False)
    # Calling func(args, kwargs) (line 35)
    func_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), func_31, *[], **kwargs_32)
    
    # Getting the type of 'd' (line 36)
    d_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 36)
    class_attribute_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), d_34, 'class_attribute')
    int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_37 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 10), 'div', class_attribute_35, int_36)
    
    # Getting the type of 'd' (line 37)
    d_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'd')
    # Obtaining the member 'instance_attribute' of a type (line 37)
    instance_attribute_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), d_38, 'instance_attribute')
    str_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'str', 'str')
    # Applying the binary operator '+' (line 37)
    result_add_41 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 10), '+', instance_attribute_39, str_40)
    
    # Getting the type of 'd' (line 39)
    d_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd')
    # Obtaining the member 'class_attribute2' of a type (line 39)
    class_attribute2_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d_42, 'class_attribute2')
    str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'str', 'str')
    # Applying the binary operator '+' (line 39)
    result_add_45 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), '+', class_attribute2_43, str_44)
    
    # Getting the type of 'd' (line 41)
    d_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'd')
    # Obtaining the member 'instance_attribute2' of a type (line 41)
    instance_attribute2_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 10), d_46, 'instance_attribute2')
    int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'int')
    # Applying the binary operator 'div' (line 41)
    result_div_49 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 10), 'div', instance_attribute2_47, int_48)
    
    
    # Call to method2(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_52 = {}
    # Getting the type of 'd' (line 43)
    d_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'd', False)
    # Obtaining the member 'method2' of a type (line 43)
    method2_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), d_50, 'method2')
    # Calling method2(args, kwargs) (line 43)
    method2_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), method2_51, *[], **kwargs_52)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
