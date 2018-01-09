
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: class Foo:
3:     def __getitem__(self, *args):
4:         print args
5:         return args
6: 
7: x = Foo()
8: r1 = x[1]
9: 
10: r2 = x[1:]
11: 
12: r3 = x[1:, :]
13: 
14: r4 = x[1:, 20:10:-2, ...]
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 3, 4, False)
        # Assigning a type to the variable 'self' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Foo.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.__getitem__.__dict__.__setitem__('stypy_function_name', 'Foo.__getitem__')
        Foo.__getitem__.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Foo.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__getitem__', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        # Getting the type of 'args' (line 4)
        args_6603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 14), 'args')
        # Getting the type of 'args' (line 5)
        args_6604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'args')
        # Assigning a type to the variable 'stypy_return_type' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', args_6604)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 3)
        stypy_return_type_6605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6605)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_6605


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 2, 0, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'Foo' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'Foo', Foo)

# Assigning a Call to a Name (line 7):

# Call to Foo(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_6607 = {}
# Getting the type of 'Foo' (line 7)
Foo_6606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Foo', False)
# Calling Foo(args, kwargs) (line 7)
Foo_call_result_6608 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), Foo_6606, *[], **kwargs_6607)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', Foo_call_result_6608)

# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
int_6609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 7), 'int')
# Getting the type of 'x' (line 8)
x_6610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___6611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), x_6610, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_6612 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), getitem___6611, int_6609)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', subscript_call_result_6612)

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
int_6613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'int')
slice_6614 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 5), int_6613, None, None)
# Getting the type of 'x' (line 10)
x_6615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___6616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), x_6615, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_6617 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), getitem___6616, slice_6614)

# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', subscript_call_result_6617)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_6618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 7), 'int')
slice_6619 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 5), int_6618, None, None)
slice_6620 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 5), None, None, None)
# Getting the type of 'x' (line 12)
x_6621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___6622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), x_6621, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_6623 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), getitem___6622, (slice_6619, slice_6620))

# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', subscript_call_result_6623)

# Assigning a Subscript to a Name (line 14):

# Obtaining the type of the subscript
int_6624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 7), 'int')
slice_6625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 14, 5), int_6624, None, None)
int_6626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'int')
int_6627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 14), 'int')
int_6628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'int')
slice_6629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 14, 5), int_6626, int_6627, int_6628)
Ellipsis_6630 = Ellipsis
# Getting the type of 'x' (line 14)
x_6631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___6632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), x_6631, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_6633 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), getitem___6632, (slice_6625, slice_6629, Ellipsis_6630))

# Assigning a type to the variable 'r4' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r4', subscript_call_result_6633)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
