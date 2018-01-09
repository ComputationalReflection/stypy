
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Foo:
2:     @staticmethod
3:     def static(x, y):
4:         return x + y
5: 
6:     def instance(self, x, y):
7:         return x + y
8: 
9: 
10: f = Foo()
11: 
12: r1 = Foo.instance(3, 4)
13: r2 = Foo().static("a", "b")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @staticmethod
    @norecursion
    def static(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'static'
        module_type_store = module_type_store.open_function_context('static', 2, 4, False)
        
        # Passed parameters checking function
        Foo.static.__dict__.__setitem__('stypy_localization', localization)
        Foo.static.__dict__.__setitem__('stypy_type_of_self', None)
        Foo.static.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.static.__dict__.__setitem__('stypy_function_name', 'static')
        Foo.static.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        Foo.static.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.static.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.static.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.static.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.static.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.static.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'static', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'static', localization, ['y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'static(...)' code ##################

        # Getting the type of 'x' (line 4)
        x_6832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 15), 'x')
        # Getting the type of 'y' (line 4)
        y_6833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'y')
        # Applying the binary operator '+' (line 4)
        result_add_6834 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 15), '+', x_6832, y_6833)
        
        # Assigning a type to the variable 'stypy_return_type' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'stypy_return_type', result_add_6834)
        
        # ################# End of 'static(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'static' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_6835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'static'
        return stypy_return_type_6835


    @norecursion
    def instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'instance'
        module_type_store = module_type_store.open_function_context('instance', 6, 4, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.instance.__dict__.__setitem__('stypy_localization', localization)
        Foo.instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.instance.__dict__.__setitem__('stypy_function_name', 'Foo.instance')
        Foo.instance.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        Foo.instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.instance.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.instance', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'instance', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'instance(...)' code ##################

        # Getting the type of 'x' (line 7)
        x_6836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'x')
        # Getting the type of 'y' (line 7)
        y_6837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'y')
        # Applying the binary operator '+' (line 7)
        result_add_6838 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 15), '+', x_6836, y_6837)
        
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', result_add_6838)
        
        # ################# End of 'instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'instance' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'instance'
        return stypy_return_type_6839


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

# Assigning a Call to a Name (line 10):

# Call to Foo(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_6841 = {}
# Getting the type of 'Foo' (line 10)
Foo_6840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 10)
Foo_call_result_6842 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), Foo_6840, *[], **kwargs_6841)

# Assigning a type to the variable 'f' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'f', Foo_call_result_6842)

# Assigning a Call to a Name (line 12):

# Call to instance(...): (line 12)
# Processing the call arguments (line 12)
int_6845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
int_6846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
# Processing the call keyword arguments (line 12)
kwargs_6847 = {}
# Getting the type of 'Foo' (line 12)
Foo_6843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'Foo', False)
# Obtaining the member 'instance' of a type (line 12)
instance_6844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), Foo_6843, 'instance')
# Calling instance(args, kwargs) (line 12)
instance_call_result_6848 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), instance_6844, *[int_6845, int_6846], **kwargs_6847)

# Assigning a type to the variable 'r1' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r1', instance_call_result_6848)

# Assigning a Call to a Name (line 13):

# Call to static(...): (line 13)
# Processing the call arguments (line 13)
str_6853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'str', 'a')
str_6854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'b')
# Processing the call keyword arguments (line 13)
kwargs_6855 = {}

# Call to Foo(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_6850 = {}
# Getting the type of 'Foo' (line 13)
Foo_6849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'Foo', False)
# Calling Foo(args, kwargs) (line 13)
Foo_call_result_6851 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), Foo_6849, *[], **kwargs_6850)

# Obtaining the member 'static' of a type (line 13)
static_6852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), Foo_call_result_6851, 'static')
# Calling static(args, kwargs) (line 13)
static_call_result_6856 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), static_6852, *[str_6853, str_6854], **kwargs_6855)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', static_call_result_6856)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
