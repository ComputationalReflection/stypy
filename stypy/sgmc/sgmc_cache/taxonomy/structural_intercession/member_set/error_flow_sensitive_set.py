
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive type member modification"
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
19: 
20: 
21:     def new_class_method(self, param):
22:         return "new method: " + str(param)
23: 
24: 
25:     def new_class_method2(self, param, param2):
26:         return "new method 2: " + str(param) + str(param2)
27: 
28: 
29:     met = types.MethodType(new_class_method, Dummy)
30:     met2 = types.MethodType(new_class_method2, Dummy)
31: 
32:     if True:
33:         setattr(Dummy, 'method', met)
34:         setattr(Dummy, 'class_attribute', "str")
35:     else:
36:         setattr(Dummy, 'method', met2)
37:         setattr(Dummy, 'class_attribute', 0.0)
38: 
39:     # Type error
40:     print d.method()
41:     # Type warning
42:     print d.method(3)
43:     # Type warning
44:     print d.method(3, 4)
45: 
46:     # Type warning
47:     print Dummy.class_attribute + "str"
48:     # Type warning
49:     print Dummy.class_attribute / 2
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive type member modification')
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

    @norecursion
    def new_class_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method'
        module_type_store = module_type_store.open_function_context('new_class_method', 21, 4, False)
        
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

        str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'new method: ')
        
        # Call to str(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'param' (line 22)
        param_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'param', False)
        # Processing the call keyword arguments (line 22)
        kwargs_15 = {}
        # Getting the type of 'str' (line 22)
        str_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', False)
        # Calling str(args, kwargs) (line 22)
        str_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 22, 32), str_13, *[param_14], **kwargs_15)
        
        # Applying the binary operator '+' (line 22)
        result_add_17 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 15), '+', str_12, str_call_result_16)
        
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', result_add_17)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_18

    # Assigning a type to the variable 'new_class_method' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'new_class_method', new_class_method)

    @norecursion
    def new_class_method2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method2'
        module_type_store = module_type_store.open_function_context('new_class_method2', 25, 4, False)
        
        # Passed parameters checking function
        new_class_method2.stypy_localization = localization
        new_class_method2.stypy_type_of_self = None
        new_class_method2.stypy_type_store = module_type_store
        new_class_method2.stypy_function_name = 'new_class_method2'
        new_class_method2.stypy_param_names_list = ['self', 'param', 'param2']
        new_class_method2.stypy_varargs_param_name = None
        new_class_method2.stypy_kwargs_param_name = None
        new_class_method2.stypy_call_defaults = defaults
        new_class_method2.stypy_call_varargs = varargs
        new_class_method2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method2', ['self', 'param', 'param2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method2', localization, ['self', 'param', 'param2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method2(...)' code ##################

        str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', 'new method 2: ')
        
        # Call to str(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'param' (line 26)
        param_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'param', False)
        # Processing the call keyword arguments (line 26)
        kwargs_22 = {}
        # Getting the type of 'str' (line 26)
        str_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 34), 'str', False)
        # Calling str(args, kwargs) (line 26)
        str_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 26, 34), str_20, *[param_21], **kwargs_22)
        
        # Applying the binary operator '+' (line 26)
        result_add_24 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '+', str_19, str_call_result_23)
        
        
        # Call to str(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'param2' (line 26)
        param2_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 51), 'param2', False)
        # Processing the call keyword arguments (line 26)
        kwargs_27 = {}
        # Getting the type of 'str' (line 26)
        str_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 47), 'str', False)
        # Calling str(args, kwargs) (line 26)
        str_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 26, 47), str_25, *[param2_26], **kwargs_27)
        
        # Applying the binary operator '+' (line 26)
        result_add_29 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 45), '+', result_add_24, str_call_result_28)
        
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', result_add_29)
        
        # ################# End of 'new_class_method2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method2' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method2'
        return stypy_return_type_30

    # Assigning a type to the variable 'new_class_method2' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'new_class_method2', new_class_method2)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to MethodType(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'new_class_method' (line 29)
    new_class_method_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'new_class_method', False)
    # Getting the type of 'Dummy' (line 29)
    Dummy_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 45), 'Dummy', False)
    # Processing the call keyword arguments (line 29)
    kwargs_35 = {}
    # Getting the type of 'types' (line 29)
    types_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 29)
    MethodType_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 10), types_31, 'MethodType')
    # Calling MethodType(args, kwargs) (line 29)
    MethodType_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), MethodType_32, *[new_class_method_33, Dummy_34], **kwargs_35)
    
    # Assigning a type to the variable 'met' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'met', MethodType_call_result_36)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to MethodType(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'new_class_method2' (line 30)
    new_class_method2_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'new_class_method2', False)
    # Getting the type of 'Dummy' (line 30)
    Dummy_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'Dummy', False)
    # Processing the call keyword arguments (line 30)
    kwargs_41 = {}
    # Getting the type of 'types' (line 30)
    types_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 30)
    MethodType_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), types_37, 'MethodType')
    # Calling MethodType(args, kwargs) (line 30)
    MethodType_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), MethodType_38, *[new_class_method2_39, Dummy_40], **kwargs_41)
    
    # Assigning a type to the variable 'met2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'met2', MethodType_call_result_42)
    
    # Getting the type of 'True' (line 32)
    True_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'True')
    # Testing the type of an if condition (line 32)
    if_condition_44 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), True_43)
    # Assigning a type to the variable 'if_condition_44' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_44', if_condition_44)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to setattr(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'Dummy' (line 33)
    Dummy_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'Dummy', False)
    str_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'str', 'method')
    # Getting the type of 'met' (line 33)
    met_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'met', False)
    # Processing the call keyword arguments (line 33)
    kwargs_49 = {}
    # Getting the type of 'setattr' (line 33)
    setattr_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'setattr', False)
    # Calling setattr(args, kwargs) (line 33)
    setattr_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), setattr_45, *[Dummy_46, str_47, met_48], **kwargs_49)
    
    
    # Call to setattr(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'Dummy' (line 34)
    Dummy_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'Dummy', False)
    str_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'str', 'class_attribute')
    str_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 42), 'str', 'str')
    # Processing the call keyword arguments (line 34)
    kwargs_55 = {}
    # Getting the type of 'setattr' (line 34)
    setattr_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'setattr', False)
    # Calling setattr(args, kwargs) (line 34)
    setattr_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), setattr_51, *[Dummy_52, str_53, str_54], **kwargs_55)
    
    # SSA branch for the else part of an if statement (line 32)
    module_type_store.open_ssa_branch('else')
    
    # Call to setattr(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'Dummy' (line 36)
    Dummy_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'Dummy', False)
    str_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'method')
    # Getting the type of 'met2' (line 36)
    met2_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'met2', False)
    # Processing the call keyword arguments (line 36)
    kwargs_61 = {}
    # Getting the type of 'setattr' (line 36)
    setattr_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'setattr', False)
    # Calling setattr(args, kwargs) (line 36)
    setattr_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), setattr_57, *[Dummy_58, str_59, met2_60], **kwargs_61)
    
    
    # Call to setattr(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'Dummy' (line 37)
    Dummy_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'Dummy', False)
    str_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'str', 'class_attribute')
    float_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 42), 'float')
    # Processing the call keyword arguments (line 37)
    kwargs_67 = {}
    # Getting the type of 'setattr' (line 37)
    setattr_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'setattr', False)
    # Calling setattr(args, kwargs) (line 37)
    setattr_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), setattr_63, *[Dummy_64, str_65, float_66], **kwargs_67)
    
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to method(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_71 = {}
    # Getting the type of 'd' (line 40)
    d_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 40)
    method_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 10), d_69, 'method')
    # Calling method(args, kwargs) (line 40)
    method_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), method_70, *[], **kwargs_71)
    
    
    # Call to method(...): (line 42)
    # Processing the call arguments (line 42)
    int_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_76 = {}
    # Getting the type of 'd' (line 42)
    d_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 42)
    method_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 10), d_73, 'method')
    # Calling method(args, kwargs) (line 42)
    method_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 42, 10), method_74, *[int_75], **kwargs_76)
    
    
    # Call to method(...): (line 44)
    # Processing the call arguments (line 44)
    int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'int')
    # Processing the call keyword arguments (line 44)
    kwargs_82 = {}
    # Getting the type of 'd' (line 44)
    d_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 44)
    method_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 10), d_78, 'method')
    # Calling method(args, kwargs) (line 44)
    method_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 44, 10), method_79, *[int_80, int_81], **kwargs_82)
    
    # Getting the type of 'Dummy' (line 47)
    Dummy_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 47)
    class_attribute_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 10), Dummy_84, 'class_attribute')
    str_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'str', 'str')
    # Applying the binary operator '+' (line 47)
    result_add_87 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 10), '+', class_attribute_85, str_86)
    
    # Getting the type of 'Dummy' (line 49)
    Dummy_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 49)
    class_attribute_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 10), Dummy_88, 'class_attribute')
    int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
    # Applying the binary operator 'div' (line 49)
    result_div_91 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 10), 'div', class_attribute_89, int_90)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
