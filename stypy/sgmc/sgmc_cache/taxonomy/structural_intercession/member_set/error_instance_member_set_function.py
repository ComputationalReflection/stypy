
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a user object inside a function"
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
22:     def new_class_method(self, param):
23:         return "new method: " + str(param)
24: 
25: 
26:     met = types.MethodType(new_class_method, Dummy)
27: 
28: 
29:     def func():
30:         setattr(d, 'method', met)
31:         setattr(d, 'class_attribute', "str")
32: 
33: 
34:     func()
35:     # Type error
36:     print d.method()
37:     print d.method(3)
38: 
39:     print d.class_attribute + "str"
40:     # Type error
41:     print d.class_attribute / 2
42: 
43:     print d2.method()
44:     # Type error
45:     print d2.method(3)
46: 
47:     # Type error
48:     print d2.class_attribute + "str"
49:     print d2.class_attribute / 2
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a user object inside a function')
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
        new_class_method.stypy_param_names_list = ['self', 'param']
        new_class_method.stypy_varargs_param_name = None
        new_class_method.stypy_kwargs_param_name = None
        new_class_method.stypy_call_defaults = defaults
        new_class_method.stypy_call_varargs = varargs
        new_class_method.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method', ['self', 'param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method', localization, ['self', 'param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method(...)' code ##################

        str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', 'new method: ')
        
        # Call to str(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'param' (line 23)
        param_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 36), 'param', False)
        # Processing the call keyword arguments (line 23)
        kwargs_18 = {}
        # Getting the type of 'str' (line 23)
        str_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', False)
        # Calling str(args, kwargs) (line 23)
        str_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 23, 32), str_16, *[param_17], **kwargs_18)
        
        # Applying the binary operator '+' (line 23)
        result_add_20 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 15), '+', str_15, str_call_result_19)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', result_add_20)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_21

    # Assigning a type to the variable 'new_class_method' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'new_class_method', new_class_method)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to MethodType(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'new_class_method' (line 26)
    new_class_method_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'new_class_method', False)
    # Getting the type of 'Dummy' (line 26)
    Dummy_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 45), 'Dummy', False)
    # Processing the call keyword arguments (line 26)
    kwargs_26 = {}
    # Getting the type of 'types' (line 26)
    types_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 26)
    MethodType_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), types_22, 'MethodType')
    # Calling MethodType(args, kwargs) (line 26)
    MethodType_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), MethodType_23, *[new_class_method_24, Dummy_25], **kwargs_26)
    
    # Assigning a type to the variable 'met' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'met', MethodType_call_result_27)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 29, 4, False)
        
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

        
        # Call to setattr(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'd' (line 30)
        d_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'd', False)
        str_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'str', 'method')
        # Getting the type of 'met' (line 30)
        met_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'met', False)
        # Processing the call keyword arguments (line 30)
        kwargs_32 = {}
        # Getting the type of 'setattr' (line 30)
        setattr_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 30)
        setattr_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), setattr_28, *[d_29, str_30, met_31], **kwargs_32)
        
        
        # Call to setattr(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'd' (line 31)
        d_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'd', False)
        str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'str', 'class_attribute')
        str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'str', 'str')
        # Processing the call keyword arguments (line 31)
        kwargs_38 = {}
        # Getting the type of 'setattr' (line 31)
        setattr_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 31)
        setattr_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), setattr_34, *[d_35, str_36, str_37], **kwargs_38)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_40

    # Assigning a type to the variable 'func' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'func', func)
    
    # Call to func(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_42 = {}
    # Getting the type of 'func' (line 34)
    func_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'func', False)
    # Calling func(args, kwargs) (line 34)
    func_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), func_41, *[], **kwargs_42)
    
    
    # Call to method(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_46 = {}
    # Getting the type of 'd' (line 36)
    d_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 36)
    method_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), d_44, 'method')
    # Calling method(args, kwargs) (line 36)
    method_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), method_45, *[], **kwargs_46)
    
    
    # Call to method(...): (line 37)
    # Processing the call arguments (line 37)
    int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'int')
    # Processing the call keyword arguments (line 37)
    kwargs_51 = {}
    # Getting the type of 'd' (line 37)
    d_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 37)
    method_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), d_48, 'method')
    # Calling method(args, kwargs) (line 37)
    method_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), method_49, *[int_50], **kwargs_51)
    
    # Getting the type of 'd' (line 39)
    d_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 39)
    class_attribute_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d_53, 'class_attribute')
    str_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'str', 'str')
    # Applying the binary operator '+' (line 39)
    result_add_56 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 10), '+', class_attribute_54, str_55)
    
    # Getting the type of 'd' (line 41)
    d_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 41)
    class_attribute_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 10), d_57, 'class_attribute')
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'int')
    # Applying the binary operator 'div' (line 41)
    result_div_60 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 10), 'div', class_attribute_58, int_59)
    
    
    # Call to method(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_63 = {}
    # Getting the type of 'd2' (line 43)
    d2_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'd2', False)
    # Obtaining the member 'method' of a type (line 43)
    method_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), d2_61, 'method')
    # Calling method(args, kwargs) (line 43)
    method_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), method_62, *[], **kwargs_63)
    
    
    # Call to method(...): (line 45)
    # Processing the call arguments (line 45)
    int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'int')
    # Processing the call keyword arguments (line 45)
    kwargs_68 = {}
    # Getting the type of 'd2' (line 45)
    d2_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'd2', False)
    # Obtaining the member 'method' of a type (line 45)
    method_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 10), d2_65, 'method')
    # Calling method(args, kwargs) (line 45)
    method_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), method_66, *[int_67], **kwargs_68)
    
    # Getting the type of 'd2' (line 48)
    d2_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'd2')
    # Obtaining the member 'class_attribute' of a type (line 48)
    class_attribute_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 10), d2_70, 'class_attribute')
    str_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'str', 'str')
    # Applying the binary operator '+' (line 48)
    result_add_73 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 10), '+', class_attribute_71, str_72)
    
    # Getting the type of 'd2' (line 49)
    d2_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'd2')
    # Obtaining the member 'class_attribute' of a type (line 49)
    class_attribute_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 10), d2_74, 'class_attribute')
    int_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'int')
    # Applying the binary operator 'div' (line 49)
    result_div_77 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), 'div', class_attribute_75, int_76)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
