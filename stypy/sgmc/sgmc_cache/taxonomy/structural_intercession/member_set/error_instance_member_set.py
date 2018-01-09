
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a user object"
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
27:     setattr(d, 'method', met)
28:     setattr(d, 'class_attribute', "str")
29: 
30:     # Type error
31:     print d.method()
32:     print d.method(3)
33: 
34:     print d.class_attribute + "str"
35:     # Type error
36:     print d.class_attribute / 2
37: 
38:     print d2.method()
39:     # Type error
40:     print d2.method(3)
41: 
42:     # Type error
43:     print d2.class_attribute + "str"
44:     print d2.class_attribute / 2
45: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a user object')
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
    
    # Call to setattr(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'd' (line 27)
    d_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'd', False)
    str_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', 'method')
    # Getting the type of 'met' (line 27)
    met_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'met', False)
    # Processing the call keyword arguments (line 27)
    kwargs_32 = {}
    # Getting the type of 'setattr' (line 27)
    setattr_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 27)
    setattr_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), setattr_28, *[d_29, str_30, met_31], **kwargs_32)
    
    
    # Call to setattr(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'd' (line 28)
    d_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'd', False)
    str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'str', 'class_attribute')
    str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'str', 'str')
    # Processing the call keyword arguments (line 28)
    kwargs_38 = {}
    # Getting the type of 'setattr' (line 28)
    setattr_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 28)
    setattr_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), setattr_34, *[d_35, str_36, str_37], **kwargs_38)
    
    
    # Call to method(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_42 = {}
    # Getting the type of 'd' (line 31)
    d_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 31)
    method_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), d_40, 'method')
    # Calling method(args, kwargs) (line 31)
    method_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), method_41, *[], **kwargs_42)
    
    
    # Call to method(...): (line 32)
    # Processing the call arguments (line 32)
    int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 19), 'int')
    # Processing the call keyword arguments (line 32)
    kwargs_47 = {}
    # Getting the type of 'd' (line 32)
    d_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 32)
    method_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 10), d_44, 'method')
    # Calling method(args, kwargs) (line 32)
    method_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), method_45, *[int_46], **kwargs_47)
    
    # Getting the type of 'd' (line 34)
    d_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 34)
    class_attribute_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 10), d_49, 'class_attribute')
    str_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'str', 'str')
    # Applying the binary operator '+' (line 34)
    result_add_52 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 10), '+', class_attribute_50, str_51)
    
    # Getting the type of 'd' (line 36)
    d_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'd')
    # Obtaining the member 'class_attribute' of a type (line 36)
    class_attribute_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 10), d_53, 'class_attribute')
    int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_56 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 10), 'div', class_attribute_54, int_55)
    
    
    # Call to method(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_59 = {}
    # Getting the type of 'd2' (line 38)
    d2_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'd2', False)
    # Obtaining the member 'method' of a type (line 38)
    method_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), d2_57, 'method')
    # Calling method(args, kwargs) (line 38)
    method_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), method_58, *[], **kwargs_59)
    
    
    # Call to method(...): (line 40)
    # Processing the call arguments (line 40)
    int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_64 = {}
    # Getting the type of 'd2' (line 40)
    d2_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'd2', False)
    # Obtaining the member 'method' of a type (line 40)
    method_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 10), d2_61, 'method')
    # Calling method(args, kwargs) (line 40)
    method_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), method_62, *[int_63], **kwargs_64)
    
    # Getting the type of 'd2' (line 43)
    d2_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'd2')
    # Obtaining the member 'class_attribute' of a type (line 43)
    class_attribute_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), d2_66, 'class_attribute')
    str_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'str', 'str')
    # Applying the binary operator '+' (line 43)
    result_add_69 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 10), '+', class_attribute_67, str_68)
    
    # Getting the type of 'd2' (line 44)
    d2_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 10), 'd2')
    # Obtaining the member 'class_attribute' of a type (line 44)
    class_attribute_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 10), d2_70, 'class_attribute')
    int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_73 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 10), 'div', class_attribute_71, int_72)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
