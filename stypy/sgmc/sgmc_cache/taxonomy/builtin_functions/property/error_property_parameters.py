
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "property method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'property'>
7:     # (Has__call__) -> <type 'property'>
8:     # (Has__call__, Has__call__) -> <type 'property'>
9:     # (Has__call__, Has__call__, Has__call__) -> <type 'property'>
10:     # (Has__call__, Has__call__, Has__call__, Str) -> <type 'property'>
11: 
12:     x = 5
13: 
14: 
15:     def fget(self):
16:         return x
17: 
18: 
19:     def fset(self, value):
20:         x = value
21: 
22: 
23:     def fdel(self):
24:         pass
25: 
26: 
27:     # Call the builtin with incorrect number of parameters
28:     # Type error
29:     ret = property(fget, fset, fdel, "test", 4)
30: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'property method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 12):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'int')
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', int_2)

    @norecursion
    def fget(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fget'
        module_type_store = module_type_store.open_function_context('fget', 15, 4, False)
        
        # Passed parameters checking function
        fget.stypy_localization = localization
        fget.stypy_type_of_self = None
        fget.stypy_type_store = module_type_store
        fget.stypy_function_name = 'fget'
        fget.stypy_param_names_list = ['self']
        fget.stypy_varargs_param_name = None
        fget.stypy_kwargs_param_name = None
        fget.stypy_call_defaults = defaults
        fget.stypy_call_varargs = varargs
        fget.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fget', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fget', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fget(...)' code ##################

        # Getting the type of 'x' (line 16)
        x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', x_3)
        
        # ################# End of 'fget(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fget' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fget'
        return stypy_return_type_4

    # Assigning a type to the variable 'fget' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'fget', fget)

    @norecursion
    def fset(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fset'
        module_type_store = module_type_store.open_function_context('fset', 19, 4, False)
        
        # Passed parameters checking function
        fset.stypy_localization = localization
        fset.stypy_type_of_self = None
        fset.stypy_type_store = module_type_store
        fset.stypy_function_name = 'fset'
        fset.stypy_param_names_list = ['self', 'value']
        fset.stypy_varargs_param_name = None
        fset.stypy_kwargs_param_name = None
        fset.stypy_call_defaults = defaults
        fset.stypy_call_varargs = varargs
        fset.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fset', ['self', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fset', localization, ['self', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fset(...)' code ##################

        
        # Assigning a Name to a Name (line 20):
        # Getting the type of 'value' (line 20)
        value_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'value')
        # Assigning a type to the variable 'x' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'x', value_5)
        
        # ################# End of 'fset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fset' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fset'
        return stypy_return_type_6

    # Assigning a type to the variable 'fset' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'fset', fset)

    @norecursion
    def fdel(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fdel'
        module_type_store = module_type_store.open_function_context('fdel', 23, 4, False)
        
        # Passed parameters checking function
        fdel.stypy_localization = localization
        fdel.stypy_type_of_self = None
        fdel.stypy_type_store = module_type_store
        fdel.stypy_function_name = 'fdel'
        fdel.stypy_param_names_list = ['self']
        fdel.stypy_varargs_param_name = None
        fdel.stypy_kwargs_param_name = None
        fdel.stypy_call_defaults = defaults
        fdel.stypy_call_varargs = varargs
        fdel.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fdel', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fdel', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fdel(...)' code ##################

        pass
        
        # ################# End of 'fdel(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fdel' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fdel'
        return stypy_return_type_7

    # Assigning a type to the variable 'fdel' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'fdel', fdel)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to property(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'fget' (line 29)
    fget_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'fget', False)
    # Getting the type of 'fset' (line 29)
    fset_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'fset', False)
    # Getting the type of 'fdel' (line 29)
    fdel_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'fdel', False)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'str', 'test')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_14 = {}
    # Getting the type of 'property' (line 29)
    property_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'property', False)
    # Calling property(args, kwargs) (line 29)
    property_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), property_8, *[fget_9, fset_10, fdel_11, str_12, int_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', property_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
