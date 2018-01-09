
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Foo:
2:     def method(self, x):
3:         return x
4: 
5: f = Foo(3, 4, 5)  # Unreported
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.method.__dict__.__setitem__('stypy_localization', localization)
        Foo.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.method.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.method.__dict__.__setitem__('stypy_function_name', 'Foo.method')
        Foo.method.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Foo.method.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.method.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.method.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.method.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.method', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        # Getting the type of 'x' (line 3)
        x_6767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', x_6767)
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_6768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6768)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_6768


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 5):

# Call to Foo(...): (line 5)
# Processing the call arguments (line 5)
int_6770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
int_6771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'int')
int_6772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_6773 = {}
# Getting the type of 'Foo' (line 5)
Foo_6769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 5)
Foo_call_result_6774 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), Foo_6769, *[int_6770, int_6771, int_6772], **kwargs_6773)

# Assigning a type to the variable 'f' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'f', Foo_call_result_6774)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
