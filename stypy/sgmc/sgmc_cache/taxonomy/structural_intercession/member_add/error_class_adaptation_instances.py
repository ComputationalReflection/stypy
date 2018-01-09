
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import types
3: 
4: __doc__ = "The modification of a class structure is reflected in its instances"
5: 
6: if __name__ == '__main__':
7:     class Dummy:
8:         class_attribute = 0
9: 
10:         def __init__(self):
11:             self.instance_attribute = "str"
12: 
13:         def method(self):
14:             return self.instance_attribute
15: 
16: 
17:     d = Dummy()
18: 
19: 
20:     def new_class_method(self):
21:         return "new method"
22: 
23: 
24:     Dummy.new_class_method = types.MethodType(new_class_method, Dummy)
25:     Dummy.new_class_attribute = 0.0
26: 
27:     print d.new_class_attribute
28:     print d.new_class_method()
29: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'The modification of a class structure is reflected in its instances')
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

    
    # Assigning a type to the variable 'Dummy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 8):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'int')
    # Getting the type of 'Dummy'
    Dummy_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_8, 'class_attribute', int_7)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to Dummy(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_10 = {}
    # Getting the type of 'Dummy' (line 17)
    Dummy_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'Dummy', False)
    # Calling Dummy(args, kwargs) (line 17)
    Dummy_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), Dummy_9, *[], **kwargs_10)
    
    # Assigning a type to the variable 'd' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'd', Dummy_call_result_11)

    @norecursion
    def new_class_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method'
        module_type_store = module_type_store.open_function_context('new_class_method', 20, 4, False)
        
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

        str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'new method')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', str_12)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_13

    # Assigning a type to the variable 'new_class_method' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'new_class_method', new_class_method)
    
    # Assigning a Call to a Attribute (line 24):
    
    # Call to MethodType(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'new_class_method' (line 24)
    new_class_method_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 46), 'new_class_method', False)
    # Getting the type of 'Dummy' (line 24)
    Dummy_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 64), 'Dummy', False)
    # Processing the call keyword arguments (line 24)
    kwargs_18 = {}
    # Getting the type of 'types' (line 24)
    types_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 24)
    MethodType_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), types_14, 'MethodType')
    # Calling MethodType(args, kwargs) (line 24)
    MethodType_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), MethodType_15, *[new_class_method_16, Dummy_17], **kwargs_18)
    
    # Getting the type of 'Dummy' (line 24)
    Dummy_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'Dummy')
    # Setting the type of the member 'new_class_method' of a type (line 24)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), Dummy_20, 'new_class_method', MethodType_call_result_19)
    
    # Assigning a Num to a Attribute (line 25):
    float_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'float')
    # Getting the type of 'Dummy' (line 25)
    Dummy_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'Dummy')
    # Setting the type of the member 'new_class_attribute' of a type (line 25)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), Dummy_22, 'new_class_attribute', float_21)
    # Getting the type of 'd' (line 27)
    d_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'd')
    # Obtaining the member 'new_class_attribute' of a type (line 27)
    new_class_attribute_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), d_23, 'new_class_attribute')
    
    # Call to new_class_method(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_27 = {}
    # Getting the type of 'd' (line 28)
    d_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'd', False)
    # Obtaining the member 'new_class_method' of a type (line 28)
    new_class_method_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 10), d_25, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 28)
    new_class_method_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), new_class_method_26, *[], **kwargs_27)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
