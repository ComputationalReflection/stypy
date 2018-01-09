
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def comparations():
2:     a = 3
3:     b = 4
4:     c = 8
5: 
6:     class Foo:
7:         # def __cmp__(self): #Do not detect the wrong definition of predefined methods (missing param)
8:         def __cmp__(self, other):
9:             return range(5)  # Do not detect "comparison did not return an int"
10: 
11:     c0 = a < Foo()  # Not reported
12:     c1 = a < b < Foo()  # Not reported
13: 
14: 
15: comparations()
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def comparations(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'comparations'
    module_type_store = module_type_store.open_function_context('comparations', 1, 0, False)
    
    # Passed parameters checking function
    comparations.stypy_localization = localization
    comparations.stypy_type_of_self = None
    comparations.stypy_type_store = module_type_store
    comparations.stypy_function_name = 'comparations'
    comparations.stypy_param_names_list = []
    comparations.stypy_varargs_param_name = None
    comparations.stypy_kwargs_param_name = None
    comparations.stypy_call_defaults = defaults
    comparations.stypy_call_varargs = varargs
    comparations.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'comparations', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'comparations', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'comparations(...)' code ##################

    
    # Assigning a Num to a Name (line 2):
    int_7161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
    # Assigning a type to the variable 'a' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_7161)
    
    # Assigning a Num to a Name (line 3):
    int_7162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
    # Assigning a type to the variable 'b' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'b', int_7162)
    
    # Assigning a Num to a Name (line 4):
    int_7163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
    # Assigning a type to the variable 'c' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'c', int_7163)
    # Declaration of the 'Foo' class

    class Foo:

        @norecursion
        def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__cmp__'
            module_type_store = module_type_store.open_function_context('__cmp__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'Foo.__cmp__')
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__cmp__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__cmp__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__cmp__(...)' code ##################

            
            # Call to range(...): (line 9)
            # Processing the call arguments (line 9)
            int_7165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 25), 'int')
            # Processing the call keyword arguments (line 9)
            kwargs_7166 = {}
            # Getting the type of 'range' (line 9)
            range_7164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'range', False)
            # Calling range(args, kwargs) (line 9)
            range_call_result_7167 = invoke(stypy.reporting.localization.Localization(__file__, 9, 19), range_7164, *[int_7165], **kwargs_7166)
            
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', range_call_result_7167)
            
            # ################# End of '__cmp__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__cmp__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7168)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__cmp__'
            return stypy_return_type_7168


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Foo' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'Foo', Foo)
    
    # Assigning a Compare to a Name (line 11):
    
    # Getting the type of 'a' (line 11)
    a_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'a')
    
    # Call to Foo(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_7171 = {}
    # Getting the type of 'Foo' (line 11)
    Foo_7170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'Foo', False)
    # Calling Foo(args, kwargs) (line 11)
    Foo_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), Foo_7170, *[], **kwargs_7171)
    
    # Applying the binary operator '<' (line 11)
    result_lt_7173 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 9), '<', a_7169, Foo_call_result_7172)
    
    # Assigning a type to the variable 'c0' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'c0', result_lt_7173)
    
    # Assigning a Compare to a Name (line 12):
    
    # Getting the type of 'a' (line 12)
    a_7174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'a')
    # Getting the type of 'b' (line 12)
    b_7175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'b')
    # Applying the binary operator '<' (line 12)
    result_lt_7176 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '<', a_7174, b_7175)
    
    # Call to Foo(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_7178 = {}
    # Getting the type of 'Foo' (line 12)
    Foo_7177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'Foo', False)
    # Calling Foo(args, kwargs) (line 12)
    Foo_call_result_7179 = invoke(stypy.reporting.localization.Localization(__file__, 12, 17), Foo_7177, *[], **kwargs_7178)
    
    # Applying the binary operator '<' (line 12)
    result_lt_7180 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '<', b_7175, Foo_call_result_7179)
    # Applying the binary operator '&' (line 12)
    result_and__7181 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 9), '&', result_lt_7176, result_lt_7180)
    
    # Assigning a type to the variable 'c1' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'c1', result_and__7181)
    
    # ################# End of 'comparations(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'comparations' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7182)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'comparations'
    return stypy_return_type_7182

# Assigning a type to the variable 'comparations' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'comparations', comparations)

# Call to comparations(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_7184 = {}
# Getting the type of 'comparations' (line 15)
comparations_7183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'comparations', False)
# Calling comparations(args, kwargs) (line 15)
comparations_call_result_7185 = invoke(stypy.reporting.localization.Localization(__file__, 15, 0), comparations_7183, *[], **kwargs_7184)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
