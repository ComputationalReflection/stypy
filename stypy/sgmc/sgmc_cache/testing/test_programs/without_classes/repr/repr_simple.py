
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class Foo:
4:     def __repr__(self):
5:         return "This is a test"
6: 
7: 
8: x = Foo()
9: 
10: y = repr(x)
11: 
12: z = repr(1+6+7)
13: print y
14: print z

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 4, 4, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Foo.__repr__')
        Foo.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_6390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', 'This is a test')
        # Assigning a type to the variable 'stypy_return_type' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', str_6390)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 4)
        stypy_return_type_6391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6391)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_6391


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 3, 0, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'Foo' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 8):

# Call to Foo(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_6393 = {}
# Getting the type of 'Foo' (line 8)
Foo_6392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 8)
Foo_call_result_6394 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), Foo_6392, *[], **kwargs_6393)

# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', Foo_call_result_6394)

# Assigning a Call to a Name (line 10):

# Call to repr(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_6397 = {}
# Getting the type of 'repr' (line 10)
repr_6395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'repr', False)
# Calling repr(args, kwargs) (line 10)
repr_call_result_6398 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), repr_6395, *[x_6396], **kwargs_6397)

# Assigning a type to the variable 'y' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'y', repr_call_result_6398)

# Assigning a Call to a Name (line 12):

# Call to repr(...): (line 12)
# Processing the call arguments (line 12)
int_6400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 9), 'int')
int_6401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'int')
# Applying the binary operator '+' (line 12)
result_add_6402 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '+', int_6400, int_6401)

int_6403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'int')
# Applying the binary operator '+' (line 12)
result_add_6404 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '+', result_add_6402, int_6403)

# Processing the call keyword arguments (line 12)
kwargs_6405 = {}
# Getting the type of 'repr' (line 12)
repr_6399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'repr', False)
# Calling repr(args, kwargs) (line 12)
repr_call_result_6406 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), repr_6399, *[result_add_6404], **kwargs_6405)

# Assigning a type to the variable 'z' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'z', repr_call_result_6406)
# Getting the type of 'y' (line 13)
y_6407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'y')
# Getting the type of 'z' (line 14)
z_6408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'z')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
