
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: class controlled_execution:
5:     def __enter__(self):
6:         print "enter the with class"
7:         return 0
8:     def __exit__(self, type, value, traceback):
9:         print "exit the with class"
10: a = 3
11: 
12: with controlled_execution():
13:     a = a + 1
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'controlled_execution' class

class controlled_execution:

    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        controlled_execution.__enter__.__dict__.__setitem__('stypy_localization', localization)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_function_name', 'controlled_execution.__enter__')
        controlled_execution.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        controlled_execution.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        controlled_execution.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        str_6476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'str', 'enter the with class')
        int_6477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', int_6477)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_6478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_6478


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        controlled_execution.__exit__.__dict__.__setitem__('stypy_localization', localization)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_function_name', 'controlled_execution.__exit__')
        controlled_execution.__exit__.__dict__.__setitem__('stypy_param_names_list', ['type', 'value', 'traceback'])
        controlled_execution.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        controlled_execution.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution.__exit__', ['type', 'value', 'traceback'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['type', 'value', 'traceback'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        str_6479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'str', 'exit the with class')
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_6480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_6480


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'controlled_execution.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'controlled_execution' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'controlled_execution', controlled_execution)

# Assigning a Num to a Name (line 10):
int_6481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'int')
# Assigning a type to the variable 'a' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'a', int_6481)

# Call to controlled_execution(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_6483 = {}
# Getting the type of 'controlled_execution' (line 12)
controlled_execution_6482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'controlled_execution', False)
# Calling controlled_execution(args, kwargs) (line 12)
controlled_execution_call_result_6484 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), controlled_execution_6482, *[], **kwargs_6483)

with_6485 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 12, 5), controlled_execution_call_result_6484, 'with parameter', '__enter__', '__exit__')

if with_6485:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 12)
    enter___6486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), controlled_execution_call_result_6484, '__enter__')
    with_enter_6487 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), enter___6486)
    
    # Assigning a BinOp to a Name (line 13):
    # Getting the type of 'a' (line 13)
    a_6488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'a')
    int_6489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
    # Applying the binary operator '+' (line 13)
    result_add_6490 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 8), '+', a_6488, int_6489)
    
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'a', result_add_6490)
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 12)
    exit___6491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), controlled_execution_call_result_6484, '__exit__')
    with_exit_6492 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), exit___6491, None, None, None)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
