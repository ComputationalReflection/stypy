
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive member get"
3: 
4: if __name__ == '__main__':
5: 
6:     class Dummy:
7:         class_attribute = 0
8:         class_attribute2 = "str"
9: 
10:         def __init__(self):
11:             self.instance_attribute = "str"
12: 
13:         def method(self):
14:             return self.instance_attribute
15: 
16: 
17:     if True:
18:         r = getattr(Dummy, 'class_attribute')
19:     else:
20:         r = getattr(Dummy, 'class_attribute2')
21: 
22:     # Type warning
23:     print r + "str"
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive member get')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 10, 8, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self', type_of_self)
            
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

            
            # Assigning a Str to a Attribute (line 11):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 38), 'str', 'str')
            # Getting the type of 'self' (line 11)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 11)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), self_3, 'instance_attribute', str_2)
            
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
            module_type_store = module_type_store.open_function_context('method', 13, 8, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self', type_of_self)
            
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

            # Getting the type of 'self' (line 14)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 14)
            instance_attribute_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 19), self_4, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'stypy_return_type', instance_attribute_5)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 13)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'Dummy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 7):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
    # Getting the type of 'Dummy'
    Dummy_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_8, 'class_attribute', int_7)
    
    # Assigning a Str to a Name (line 8):
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'str')
    # Getting the type of 'Dummy'
    Dummy_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute2' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_10, 'class_attribute2', str_9)
    
    # Getting the type of 'True' (line 17)
    True_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'True')
    # Testing the type of an if condition (line 17)
    if_condition_12 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), True_11)
    # Assigning a type to the variable 'if_condition_12' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_12', if_condition_12)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 18):
    
    # Call to getattr(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'Dummy' (line 18)
    Dummy_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'Dummy', False)
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'str', 'class_attribute')
    # Processing the call keyword arguments (line 18)
    kwargs_16 = {}
    # Getting the type of 'getattr' (line 18)
    getattr_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 18)
    getattr_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), getattr_13, *[Dummy_14, str_15], **kwargs_16)
    
    # Assigning a type to the variable 'r' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'r', getattr_call_result_17)
    # SSA branch for the else part of an if statement (line 17)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 20):
    
    # Call to getattr(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'Dummy' (line 20)
    Dummy_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'Dummy', False)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'str', 'class_attribute2')
    # Processing the call keyword arguments (line 20)
    kwargs_21 = {}
    # Getting the type of 'getattr' (line 20)
    getattr_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 20)
    getattr_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), getattr_18, *[Dummy_19, str_20], **kwargs_21)
    
    # Assigning a type to the variable 'r' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'r', getattr_call_result_22)
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 23)
    r_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'r')
    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', 'str')
    # Applying the binary operator '+' (line 23)
    result_add_25 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 10), '+', r_23, str_24)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
