
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: def function():
3:     class Simple:
4:         sample_att = 3
5: 
6:         def sample_method(self):
7:             self.att = "sample"
8: 
9:     return Simple()
10: 
11: ret = function()
12: 
13: 
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'function'
    module_type_store = module_type_store.open_function_context('function', 2, 0, False)
    
    # Passed parameters checking function
    function.stypy_localization = localization
    function.stypy_type_of_self = None
    function.stypy_type_store = module_type_store
    function.stypy_function_name = 'function'
    function.stypy_param_names_list = []
    function.stypy_varargs_param_name = None
    function.stypy_kwargs_param_name = None
    function.stypy_call_defaults = defaults
    function.stypy_call_varargs = varargs
    function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'function', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'function', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'function(...)' code ##################

    # Declaration of the 'Simple' class

    class Simple:

        @norecursion
        def sample_method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'sample_method'
            module_type_store = module_type_store.open_function_context('sample_method', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Simple.sample_method.__dict__.__setitem__('stypy_localization', localization)
            Simple.sample_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Simple.sample_method.__dict__.__setitem__('stypy_type_store', module_type_store)
            Simple.sample_method.__dict__.__setitem__('stypy_function_name', 'Simple.sample_method')
            Simple.sample_method.__dict__.__setitem__('stypy_param_names_list', [])
            Simple.sample_method.__dict__.__setitem__('stypy_varargs_param_name', None)
            Simple.sample_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Simple.sample_method.__dict__.__setitem__('stypy_call_defaults', defaults)
            Simple.sample_method.__dict__.__setitem__('stypy_call_varargs', varargs)
            Simple.sample_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Simple.sample_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple.sample_method', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'sample_method', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'sample_method(...)' code ##################

            
            # Assigning a Str to a Attribute (line 7):
            str_1875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'str', 'sample')
            # Getting the type of 'self' (line 7)
            self_1876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'self')
            # Setting the type of the member 'att' of a type (line 7)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 12), self_1876, 'att', str_1875)
            
            # ################# End of 'sample_method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'sample_method' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1877)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'sample_method'
            return stypy_return_type_1877


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 3, 4, False)
            # Assigning a type to the variable 'self' (line 4)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Simple' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'Simple', Simple)
    
    # Assigning a Num to a Name (line 4):
    int_1878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 21), 'int')
    # Getting the type of 'Simple'
    Simple_1879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
    # Setting the type of the member 'sample_att' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_1879, 'sample_att', int_1878)
    
    # Call to Simple(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_1881 = {}
    # Getting the type of 'Simple' (line 9)
    Simple_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'Simple', False)
    # Calling Simple(args, kwargs) (line 9)
    Simple_call_result_1882 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), Simple_1880, *[], **kwargs_1881)
    
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', Simple_call_result_1882)
    
    # ################# End of 'function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'function' in the type store
    # Getting the type of 'stypy_return_type' (line 2)
    stypy_return_type_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1883)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'function'
    return stypy_return_type_1883

# Assigning a type to the variable 'function' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'function', function)

# Assigning a Call to a Name (line 11):

# Call to function(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_1885 = {}
# Getting the type of 'function' (line 11)
function_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'function', False)
# Calling function(args, kwargs) (line 11)
function_call_result_1886 = invoke(stypy.reporting.localization.Localization(__file__, 11, 6), function_1884, *[], **kwargs_1885)

# Assigning a type to the variable 'ret' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'ret', function_call_result_1886)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
