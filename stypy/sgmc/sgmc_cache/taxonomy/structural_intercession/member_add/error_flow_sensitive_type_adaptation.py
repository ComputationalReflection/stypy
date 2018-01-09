
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import types
3: 
4: __doc__ = "Inferring flow-sensitive type adaptation"
5: 
6: if __name__ == '__main__':
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
21:     def new_class_method(self):
22:         return "new method"
23: 
24: 
25:     def new_class_method2(self):
26:         return "new method 2"
27: 
28: 
29:     if True:
30:         Dummy.new_class_method = types.MethodType(new_class_method, Dummy)
31:         Dummy.new_class_attribute = 0.0
32:     else:
33:         Dummy.new_class_method2 = types.MethodType(new_class_method2, Dummy)
34:         Dummy.new_class_attribute2 = list()
35: 
36:     # Type warning
37:     print d.new_class_attribute / 2
38:     # Type warning
39:     print d.new_class_method()
40:     # Type warning
41:     print len(d.new_class_attribute2)
42:     # Type warning
43:     print d.new_class_method2()
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Inferring flow-sensitive type adaptation')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
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

        str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'new method')
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', str_12)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_13

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

        str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', 'new method 2')
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', str_14)
        
        # ################# End of 'new_class_method2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method2' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method2'
        return stypy_return_type_15

    # Assigning a type to the variable 'new_class_method2' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'new_class_method2', new_class_method2)
    
    # Getting the type of 'True' (line 29)
    True_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'True')
    # Testing the type of an if condition (line 29)
    if_condition_17 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), True_16)
    # Assigning a type to the variable 'if_condition_17' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_17', if_condition_17)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Attribute (line 30):
    
    # Call to MethodType(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'new_class_method' (line 30)
    new_class_method_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'new_class_method', False)
    # Getting the type of 'Dummy' (line 30)
    Dummy_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 68), 'Dummy', False)
    # Processing the call keyword arguments (line 30)
    kwargs_22 = {}
    # Getting the type of 'types' (line 30)
    types_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 30)
    MethodType_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 33), types_18, 'MethodType')
    # Calling MethodType(args, kwargs) (line 30)
    MethodType_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 30, 33), MethodType_19, *[new_class_method_20, Dummy_21], **kwargs_22)
    
    # Getting the type of 'Dummy' (line 30)
    Dummy_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'Dummy')
    # Setting the type of the member 'new_class_method' of a type (line 30)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), Dummy_24, 'new_class_method', MethodType_call_result_23)
    
    # Assigning a Num to a Attribute (line 31):
    float_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'float')
    # Getting the type of 'Dummy' (line 31)
    Dummy_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'Dummy')
    # Setting the type of the member 'new_class_attribute' of a type (line 31)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), Dummy_26, 'new_class_attribute', float_25)
    # SSA branch for the else part of an if statement (line 29)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Attribute (line 33):
    
    # Call to MethodType(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'new_class_method2' (line 33)
    new_class_method2_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 51), 'new_class_method2', False)
    # Getting the type of 'Dummy' (line 33)
    Dummy_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 70), 'Dummy', False)
    # Processing the call keyword arguments (line 33)
    kwargs_31 = {}
    # Getting the type of 'types' (line 33)
    types_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 34), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 33)
    MethodType_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 34), types_27, 'MethodType')
    # Calling MethodType(args, kwargs) (line 33)
    MethodType_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 33, 34), MethodType_28, *[new_class_method2_29, Dummy_30], **kwargs_31)
    
    # Getting the type of 'Dummy' (line 33)
    Dummy_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'Dummy')
    # Setting the type of the member 'new_class_method2' of a type (line 33)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), Dummy_33, 'new_class_method2', MethodType_call_result_32)
    
    # Assigning a Call to a Attribute (line 34):
    
    # Call to list(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_35 = {}
    # Getting the type of 'list' (line 34)
    list_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 37), 'list', False)
    # Calling list(args, kwargs) (line 34)
    list_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 34, 37), list_34, *[], **kwargs_35)
    
    # Getting the type of 'Dummy' (line 34)
    Dummy_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'Dummy')
    # Setting the type of the member 'new_class_attribute2' of a type (line 34)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), Dummy_37, 'new_class_attribute2', list_call_result_36)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 37)
    d_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'd')
    # Obtaining the member 'new_class_attribute' of a type (line 37)
    new_class_attribute_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 10), d_38, 'new_class_attribute')
    int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_41 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 10), 'div', new_class_attribute_39, int_40)
    
    
    # Call to new_class_method(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_44 = {}
    # Getting the type of 'd' (line 39)
    d_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'd', False)
    # Obtaining the member 'new_class_method' of a type (line 39)
    new_class_method_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), d_42, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 39)
    new_class_method_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), new_class_method_43, *[], **kwargs_44)
    
    
    # Call to len(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'd' (line 41)
    d_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'd', False)
    # Obtaining the member 'new_class_attribute2' of a type (line 41)
    new_class_attribute2_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 14), d_47, 'new_class_attribute2')
    # Processing the call keyword arguments (line 41)
    kwargs_49 = {}
    # Getting the type of 'len' (line 41)
    len_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'len', False)
    # Calling len(args, kwargs) (line 41)
    len_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 41, 10), len_46, *[new_class_attribute2_48], **kwargs_49)
    
    
    # Call to new_class_method2(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_53 = {}
    # Getting the type of 'd' (line 43)
    d_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'd', False)
    # Obtaining the member 'new_class_method2' of a type (line 43)
    new_class_method2_52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), d_51, 'new_class_method2')
    # Calling new_class_method2(args, kwargs) (line 43)
    new_class_method2_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 43, 10), new_class_method2_52, *[], **kwargs_53)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
