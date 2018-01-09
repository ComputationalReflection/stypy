
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
8: 
9: a = 3
10: 
11: with controlled_execution() as thing:
12:     a = a + 1
13:     print thing
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

        str_6460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'str', 'enter the with class')
        int_6461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', int_6461)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_6462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_6462


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

# Assigning a Num to a Name (line 9):
int_6463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'int')
# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a', int_6463)

# Call to controlled_execution(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_6465 = {}
# Getting the type of 'controlled_execution' (line 11)
controlled_execution_6464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'controlled_execution', False)
# Calling controlled_execution(args, kwargs) (line 11)
controlled_execution_call_result_6466 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), controlled_execution_6464, *[], **kwargs_6465)

with_6467 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 11, 5), controlled_execution_call_result_6466, 'with parameter', '__enter__', '__exit__')

if with_6467:
    # Calling the __enter__ method to initiate a with section
    # Obtaining the member '__enter__' of a type (line 11)
    enter___6468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), controlled_execution_call_result_6466, '__enter__')
    with_enter_6469 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), enter___6468)
    # Assigning a type to the variable 'thing' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'thing', with_enter_6469)
    
    # Assigning a BinOp to a Name (line 12):
    # Getting the type of 'a' (line 12)
    a_6470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'a')
    int_6471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
    # Applying the binary operator '+' (line 12)
    result_add_6472 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 8), '+', a_6470, int_6471)
    
    # Assigning a type to the variable 'a' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a', result_add_6472)
    # Getting the type of 'thing' (line 13)
    thing_6473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'thing')
    # Calling the __exit__ method to finish a with section
    # Obtaining the member '__exit__' of a type (line 11)
    exit___6474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), controlled_execution_call_result_6466, '__exit__')
    with_exit_6475 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), exit___6474, None, None, None)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
