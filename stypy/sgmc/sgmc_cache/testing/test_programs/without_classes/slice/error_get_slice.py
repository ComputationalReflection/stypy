
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class Foo:
2:     def __abs__(self):
3:         return 4000
4: 
5:     def __add__(self, other):
6:         return 4
7: 
8: class Foo2:
9:     def __getitem__(self, x1):
10:         return x1
11: 
12: r1 = Foo()[1:2]  # missing __getslice__ not reported
13: 
14: x = 3
15: r2 = x[1:3] # Not reported
16: 
17: r3 = Foo2()[1:2]
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.__abs__.__dict__.__setitem__('stypy_localization', localization)
        Foo.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.__abs__.__dict__.__setitem__('stypy_function_name', 'Foo.__abs__')
        Foo.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        int_7698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', int_7698)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_7699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_7699


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.__add__.__dict__.__setitem__('stypy_localization', localization)
        Foo.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.__add__.__dict__.__setitem__('stypy_function_name', 'Foo.__add__')
        Foo.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Foo.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        int_7700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', int_7700)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_7701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7701)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_7701


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
# Declaration of the 'Foo2' class

class Foo2:

    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo2.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Foo2.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo2.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo2.__getitem__.__dict__.__setitem__('stypy_function_name', 'Foo2.__getitem__')
        Foo2.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['x1'])
        Foo2.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo2.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo2.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo2.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo2.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo2.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo2.__getitem__', ['x1'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['x1'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        # Getting the type of 'x1' (line 10)
        x1_7702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'x1')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', x1_7702)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_7703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_7703


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo2.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Foo2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Foo2', Foo2)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_7704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'int')
int_7705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'int')
slice_7706 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 5), int_7704, int_7705, None)

# Call to Foo(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_7708 = {}
# Getting the type of 'Foo' (line 12)
Foo_7707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'Foo', False)
# Calling Foo(args, kwargs) (line 12)
Foo_call_result_7709 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), Foo_7707, *[], **kwargs_7708)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___7710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), Foo_call_result_7709, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_7711 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), getitem___7710, slice_7706)

# Assigning a type to the variable 'r1' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r1', subscript_call_result_7711)

# Assigning a Num to a Name (line 14):
int_7712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'int')
# Assigning a type to the variable 'x' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'x', int_7712)

# Assigning a Subscript to a Name (line 15):

# Obtaining the type of the subscript
int_7713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 7), 'int')
int_7714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 9), 'int')
slice_7715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 15, 5), int_7713, int_7714, None)
# Getting the type of 'x' (line 15)
x_7716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___7717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), x_7716, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_7718 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), getitem___7717, slice_7715)

# Assigning a type to the variable 'r2' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r2', subscript_call_result_7718)

# Assigning a Subscript to a Name (line 17):

# Obtaining the type of the subscript
int_7719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'int')
int_7720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
slice_7721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 17, 5), int_7719, int_7720, None)

# Call to Foo2(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_7723 = {}
# Getting the type of 'Foo2' (line 17)
Foo2_7722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'Foo2', False)
# Calling Foo2(args, kwargs) (line 17)
Foo2_call_result_7724 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), Foo2_7722, *[], **kwargs_7723)

# Obtaining the member '__getitem__' of a type (line 17)
getitem___7725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), Foo2_call_result_7724, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 17)
subscript_call_result_7726 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), getitem___7725, slice_7721)

# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', subscript_call_result_7726)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
