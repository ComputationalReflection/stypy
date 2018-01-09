
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Illegal inheritance tree change"
3: 
4: if __name__ == '__main__':
5:     class DummyDestination:
6:         class_attribute2 = "str"
7: 
8:         def __init__(self):
9:             self.instance_attribute2 = 0
10: 
11:         def method2(self):
12:             return self.class_attribute2
13: 
14: 
15:     # Type error
16:     list.__bases__ = (DummyDestination,)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Illegal inheritance tree change')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DummyDestination' class

    class DummyDestination:

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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Num to a Attribute (line 9):
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 39), 'int')
            # Getting the type of 'self' (line 9)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'self')
            # Setting the type of the member 'instance_attribute2' of a type (line 9)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), self_3, 'instance_attribute2', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method2(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method2'
            module_type_store = module_type_store.open_function_context('method2', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DummyDestination.method2.__dict__.__setitem__('stypy_localization', localization)
            DummyDestination.method2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DummyDestination.method2.__dict__.__setitem__('stypy_type_store', module_type_store)
            DummyDestination.method2.__dict__.__setitem__('stypy_function_name', 'DummyDestination.method2')
            DummyDestination.method2.__dict__.__setitem__('stypy_param_names_list', [])
            DummyDestination.method2.__dict__.__setitem__('stypy_varargs_param_name', None)
            DummyDestination.method2.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_defaults', defaults)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_varargs', varargs)
            DummyDestination.method2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DummyDestination.method2.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyDestination.method2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'method2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'method2(...)' code ##################

            # Getting the type of 'self' (line 12)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'self')
            # Obtaining the member 'class_attribute2' of a type (line 12)
            class_attribute2_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), self_4, 'class_attribute2')
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', class_attribute2_5)
            
            # ################# End of 'method2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method2' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method2'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'DummyDestination' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'DummyDestination', DummyDestination)
    
    # Assigning a Str to a Name (line 6):
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'str')
    # Getting the type of 'DummyDestination'
    DummyDestination_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'DummyDestination')
    # Setting the type of the member 'class_attribute2' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), DummyDestination_8, 'class_attribute2', str_7)
    
    # Assigning a Tuple to a Attribute (line 16):
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'DummyDestination' (line 16)
    DummyDestination_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'DummyDestination')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 22), tuple_9, DummyDestination_10)
    
    # Getting the type of 'list' (line 16)
    list_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'list')
    # Setting the type of the member '__bases__' of a type (line 16)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), list_11, '__bases__', tuple_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
