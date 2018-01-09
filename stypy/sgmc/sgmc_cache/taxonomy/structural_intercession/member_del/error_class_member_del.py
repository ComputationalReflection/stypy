
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Del a member of a user class"
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
15:     delattr(Dummy, 'class_attribute')
16:     # Type error
17:     print Dummy.class_attribute / 2
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Del a member of a user class')
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
    
    # Call to delattr(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'Dummy' (line 15)
    Dummy_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'Dummy', False)
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'str', 'class_attribute')
    # Processing the call keyword arguments (line 15)
    kwargs_12 = {}
    # Getting the type of 'delattr' (line 15)
    delattr_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'delattr', False)
    # Calling delattr(args, kwargs) (line 15)
    delattr_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), delattr_9, *[Dummy_10, str_11], **kwargs_12)
    
    # Getting the type of 'Dummy' (line 17)
    Dummy_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 17)
    class_attribute_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 10), Dummy_14, 'class_attribute')
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'int')
    # Applying the binary operator 'div' (line 17)
    result_div_17 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 10), 'div', class_attribute_15, int_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
