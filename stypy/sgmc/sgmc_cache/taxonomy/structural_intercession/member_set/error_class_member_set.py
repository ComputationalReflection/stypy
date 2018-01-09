
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a user class"
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
25:     met = types.MethodType(new_class_method, Dummy)
26:     setattr(Dummy, 'method', met)
27:     setattr(Dummy, 'class_attribute', "str")
28: 
29:     # Type error
30:     print d.method()
31:     print d.method(3)
32: 
33:     print Dummy.class_attribute + "str"
34:     # Type error
35:     print Dummy.class_attribute / 2
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a user class')
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
    
    # Assigning a Call to a Name (line 25):
    
    # Call to MethodType(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'new_class_method' (line 25)
    new_class_method_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'new_class_method', False)
    # Getting the type of 'Dummy' (line 25)
    Dummy_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 45), 'Dummy', False)
    # Processing the call keyword arguments (line 25)
    kwargs_23 = {}
    # Getting the type of 'types' (line 25)
    types_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 25)
    MethodType_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), types_19, 'MethodType')
    # Calling MethodType(args, kwargs) (line 25)
    MethodType_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), MethodType_20, *[new_class_method_21, Dummy_22], **kwargs_23)
    
    # Assigning a type to the variable 'met' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'met', MethodType_call_result_24)
    
    # Call to setattr(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'Dummy' (line 26)
    Dummy_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'Dummy', False)
    str_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'method')
    # Getting the type of 'met' (line 26)
    met_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'met', False)
    # Processing the call keyword arguments (line 26)
    kwargs_29 = {}
    # Getting the type of 'setattr' (line 26)
    setattr_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 26)
    setattr_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), setattr_25, *[Dummy_26, str_27, met_28], **kwargs_29)
    
    
    # Call to setattr(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'Dummy' (line 27)
    Dummy_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'Dummy', False)
    str_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'class_attribute')
    str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 38), 'str', 'str')
    # Processing the call keyword arguments (line 27)
    kwargs_35 = {}
    # Getting the type of 'setattr' (line 27)
    setattr_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 27)
    setattr_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), setattr_31, *[Dummy_32, str_33, str_34], **kwargs_35)
    
    
    # Call to method(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_39 = {}
    # Getting the type of 'd' (line 30)
    d_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 30)
    method_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 10), d_37, 'method')
    # Calling method(args, kwargs) (line 30)
    method_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), method_38, *[], **kwargs_39)
    
    
    # Call to method(...): (line 31)
    # Processing the call arguments (line 31)
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_44 = {}
    # Getting the type of 'd' (line 31)
    d_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'd', False)
    # Obtaining the member 'method' of a type (line 31)
    method_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), d_41, 'method')
    # Calling method(args, kwargs) (line 31)
    method_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), method_42, *[int_43], **kwargs_44)
    
    # Getting the type of 'Dummy' (line 33)
    Dummy_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 33)
    class_attribute_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 10), Dummy_46, 'class_attribute')
    str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'str')
    # Applying the binary operator '+' (line 33)
    result_add_49 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 10), '+', class_attribute_47, str_48)
    
    # Getting the type of 'Dummy' (line 35)
    Dummy_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 35)
    class_attribute_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 10), Dummy_50, 'class_attribute')
    int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'int')
    # Applying the binary operator 'div' (line 35)
    result_div_53 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 10), 'div', class_attribute_51, int_52)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
