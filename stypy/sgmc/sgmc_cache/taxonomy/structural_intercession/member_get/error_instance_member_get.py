
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Get the type of a member of a user object"
3: 
4: if __name__ == '__main__':
5:     class Dummy:
6:         class_attribute = 0
7: 
8:         def __init__(self):
9:             self.instance_attribute = "str"
10: 
11:         def method(self):
12:             return self.instance_attribute
13: 
14: 
15:     d = Dummy()
16:     r = getattr(d, 'class_attribute')
17:     print r * 3
18:     # Type error
19:     print r + "str"
20: 
21:     r = getattr(d, 'instance_attribute')
22:     print len(r)
23:     # Type error
24:     print r / 3
25: 
26:     r = getattr(d, 'method')
27:     # Type error
28:     print r / 3
29: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Get the type of a member of a user object')
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
            module_type_store = module_type_store.open_function_context('__init__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Dummy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 6):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
    # Getting the type of 'Dummy'
    Dummy_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_8, 'class_attribute', int_7)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to Dummy(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_10 = {}
    # Getting the type of 'Dummy' (line 15)
    Dummy_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 15)
    Dummy_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), Dummy_9, *[], **kwargs_10)
    
    # Assigning a type to the variable 'd' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'd', Dummy_call_result_11)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to getattr(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'd' (line 16)
    d_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'd', False)
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'str', 'class_attribute')
    # Processing the call keyword arguments (line 16)
    kwargs_15 = {}
    # Getting the type of 'getattr' (line 16)
    getattr_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 16)
    getattr_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getattr_12, *[d_13, str_14], **kwargs_15)
    
    # Assigning a type to the variable 'r' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'r', getattr_call_result_16)
    # Getting the type of 'r' (line 17)
    r_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'r')
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    # Applying the binary operator '*' (line 17)
    result_mul_19 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 10), '*', r_17, int_18)
    
    # Getting the type of 'r' (line 19)
    r_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'r')
    str_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'str', 'str')
    # Applying the binary operator '+' (line 19)
    result_add_22 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', r_20, str_21)
    
    
    # Assigning a Call to a Name (line 21):
    
    # Call to getattr(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'd' (line 21)
    d_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'd', False)
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'str', 'instance_attribute')
    # Processing the call keyword arguments (line 21)
    kwargs_26 = {}
    # Getting the type of 'getattr' (line 21)
    getattr_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 21)
    getattr_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), getattr_23, *[d_24, str_25], **kwargs_26)
    
    # Assigning a type to the variable 'r' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'r', getattr_call_result_27)
    
    # Call to len(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'r' (line 22)
    r_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'r', False)
    # Processing the call keyword arguments (line 22)
    kwargs_30 = {}
    # Getting the type of 'len' (line 22)
    len_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'len', False)
    # Calling len(args, kwargs) (line 22)
    len_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), len_28, *[r_29], **kwargs_30)
    
    # Getting the type of 'r' (line 24)
    r_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'r')
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'int')
    # Applying the binary operator 'div' (line 24)
    result_div_34 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 10), 'div', r_32, int_33)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Call to getattr(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'd' (line 26)
    d_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'd', False)
    str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'method')
    # Processing the call keyword arguments (line 26)
    kwargs_38 = {}
    # Getting the type of 'getattr' (line 26)
    getattr_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 26)
    getattr_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), getattr_35, *[d_36, str_37], **kwargs_38)
    
    # Assigning a type to the variable 'r' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'r', getattr_call_result_39)
    # Getting the type of 'r' (line 28)
    r_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'r')
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_42 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 10), 'div', r_40, int_41)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
