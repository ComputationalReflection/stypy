
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Collect types of the members added to a user object"
3: 
4: if __name__ == '__main__':
5:     import types
6: 
7: 
8:     class Dummy:
9:         class_attribute = 0
10: 
11:         def __init__(self):
12:             self.instance_attribute = "str"
13: 
14:         def method(self):
15:             return self.instance_attribute
16: 
17: 
18:     d = Dummy()
19:     d2 = Dummy()
20: 
21: 
22:     def new_class_method(self):
23:         return "new method"
24: 
25: 
26:     def new_class_method2(self):
27:         return "new method 2"
28: 
29: 
30:     d.new_class_method = types.MethodType(new_class_method, d)
31:     d.new_class_attribute = 0.0
32: 
33:     print d.new_class_attribute
34:     print d.new_class_method()
35: 
36:     # Type error
37:     print d2.new_class_attribute
38:     # Type error
39:     print d2.new_class_method()
40: 
41:     # Type error
42:     print Dummy.new_class_attribute
43:     # Type error
44:     print Dummy.new_class_method()
45: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Collect types of the members added to a user object')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import types' statement (line 5)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'types', types, module_type_store)
    
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
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

            
            # Assigning a Str to a Attribute (line 12):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 38), 'str', 'str')
            # Getting the type of 'self' (line 12)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 12)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), self_3, 'instance_attribute', str_2)
            
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
            module_type_store = module_type_store.open_function_context('method', 14, 8, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', type_of_self)
            
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

            # Getting the type of 'self' (line 15)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 15)
            instance_attribute_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 19), self_4, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', instance_attribute_5)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 14)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'Dummy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 9):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'int')
    # Getting the type of 'Dummy'
    Dummy_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_8, 'class_attribute', int_7)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to Dummy(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_10 = {}
    # Getting the type of 'Dummy' (line 18)
    Dummy_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 18)
    Dummy_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), Dummy_9, *[], **kwargs_10)
    
    # Assigning a type to the variable 'd' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'd', Dummy_call_result_11)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to Dummy(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_13 = {}
    # Getting the type of 'Dummy' (line 19)
    Dummy_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 19)
    Dummy_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), Dummy_12, *[], **kwargs_13)
    
    # Assigning a type to the variable 'd2' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'd2', Dummy_call_result_14)

    @norecursion
    def new_class_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method'
        module_type_store = module_type_store.open_function_context('new_class_method', 22, 4, False)
        
        # Passed parameters checking function
        new_class_method.stypy_localization = localization
        new_class_method.stypy_type_of_self = None
        new_class_method.stypy_type_store = module_type_store
        new_class_method.stypy_function_name = 'new_class_method'
        new_class_method.stypy_param_names_list = ['self']
        new_class_method.stypy_varargs_param_name = None
        new_class_method.stypy_kwargs_param_name = None
        new_class_method.stypy_call_defaults = defaults
        new_class_method.stypy_call_varargs = varargs
        new_class_method.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method(...)' code ##################

        str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', 'new method')
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', str_15)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_16

    # Assigning a type to the variable 'new_class_method' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'new_class_method', new_class_method)

    @norecursion
    def new_class_method2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method2'
        module_type_store = module_type_store.open_function_context('new_class_method2', 26, 4, False)
        
        # Passed parameters checking function
        new_class_method2.stypy_localization = localization
        new_class_method2.stypy_type_of_self = None
        new_class_method2.stypy_type_store = module_type_store
        new_class_method2.stypy_function_name = 'new_class_method2'
        new_class_method2.stypy_param_names_list = ['self']
        new_class_method2.stypy_varargs_param_name = None
        new_class_method2.stypy_kwargs_param_name = None
        new_class_method2.stypy_call_defaults = defaults
        new_class_method2.stypy_call_varargs = varargs
        new_class_method2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method2', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method2', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method2(...)' code ##################

        str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', 'new method 2')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', str_17)
        
        # ################# End of 'new_class_method2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method2' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method2'
        return stypy_return_type_18

    # Assigning a type to the variable 'new_class_method2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'new_class_method2', new_class_method2)
    
    # Assigning a Call to a Attribute (line 30):
    
    # Call to MethodType(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'new_class_method' (line 30)
    new_class_method_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 42), 'new_class_method', False)
    # Getting the type of 'd' (line 30)
    d_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 60), 'd', False)
    # Processing the call keyword arguments (line 30)
    kwargs_23 = {}
    # Getting the type of 'types' (line 30)
    types_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 30)
    MethodType_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 25), types_19, 'MethodType')
    # Calling MethodType(args, kwargs) (line 30)
    MethodType_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), MethodType_20, *[new_class_method_21, d_22], **kwargs_23)
    
    # Getting the type of 'd' (line 30)
    d_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'd')
    # Setting the type of the member 'new_class_method' of a type (line 30)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), d_25, 'new_class_method', MethodType_call_result_24)
    
    # Assigning a Num to a Attribute (line 31):
    float_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'float')
    # Getting the type of 'd' (line 31)
    d_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'd')
    # Setting the type of the member 'new_class_attribute' of a type (line 31)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), d_27, 'new_class_attribute', float_26)
    # Getting the type of 'd' (line 33)
    d_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'd')
    # Obtaining the member 'new_class_attribute' of a type (line 33)
    new_class_attribute_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 10), d_28, 'new_class_attribute')
    
    # Call to new_class_method(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_32 = {}
    # Getting the type of 'd' (line 34)
    d_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'd', False)
    # Obtaining the member 'new_class_method' of a type (line 34)
    new_class_method_31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 10), d_30, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 34)
    new_class_method_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 34, 10), new_class_method_31, *[], **kwargs_32)
    
    # Getting the type of 'd2' (line 37)
    d2_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'd2')
    # Obtaining the member 'new_class_attribute' of a type (line 37)
    new_class_attribute_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), d2_34, 'new_class_attribute')
    
    # Call to new_class_method(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_38 = {}
    # Getting the type of 'd2' (line 39)
    d2_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd2', False)
    # Obtaining the member 'new_class_method' of a type (line 39)
    new_class_method_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d2_36, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 39)
    new_class_method_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), new_class_method_37, *[], **kwargs_38)
    
    # Getting the type of 'Dummy' (line 42)
    Dummy_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'Dummy')
    # Obtaining the member 'new_class_attribute' of a type (line 42)
    new_class_attribute_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), Dummy_40, 'new_class_attribute')
    
    # Call to new_class_method(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_44 = {}
    # Getting the type of 'Dummy' (line 44)
    Dummy_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 10), 'Dummy', False)
    # Obtaining the member 'new_class_method' of a type (line 44)
    new_class_method_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 10), Dummy_42, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 44)
    new_class_method_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 44, 10), new_class_method_43, *[], **kwargs_44)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
