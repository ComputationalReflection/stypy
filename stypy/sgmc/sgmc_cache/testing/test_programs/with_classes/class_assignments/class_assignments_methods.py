
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class ndenumerate(object):
2:     def __next__(self):
3:         return 0
4: 
5:     next = __next__
6: 
7: 
8: o = ndenumerate()
9: r = o.next
10: r2 = o.__next__
11: print r
12: print r2
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'ndenumerate' class

class ndenumerate(object, ):

    @norecursion
    def __next__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__next__'
        module_type_store = module_type_store.open_function_context('__next__', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ndenumerate.__next__.__dict__.__setitem__('stypy_localization', localization)
        ndenumerate.__next__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ndenumerate.__next__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ndenumerate.__next__.__dict__.__setitem__('stypy_function_name', 'ndenumerate.__next__')
        ndenumerate.__next__.__dict__.__setitem__('stypy_param_names_list', [])
        ndenumerate.__next__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ndenumerate.__next__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ndenumerate.__next__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ndenumerate.__next__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndenumerate.__next__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__next__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__next__(...)' code ##################

        int_1204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', int_1204)
        
        # ################# End of '__next__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__next__' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1205)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__next__'
        return stypy_return_type_1205


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ndenumerate.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ndenumerate' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'ndenumerate', ndenumerate)

# Assigning a Name to a Name (line 5):
# Getting the type of 'ndenumerate'
ndenumerate_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndenumerate')
# Obtaining the member '__next__' of a type
next___1207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndenumerate_1206, '__next__')
# Getting the type of 'ndenumerate'
ndenumerate_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ndenumerate')
# Setting the type of the member 'next' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ndenumerate_1208, 'next', next___1207)

# Assigning a Call to a Name (line 8):

# Call to ndenumerate(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_1210 = {}
# Getting the type of 'ndenumerate' (line 8)
ndenumerate_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'ndenumerate', False)
# Calling ndenumerate(args, kwargs) (line 8)
ndenumerate_call_result_1211 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), ndenumerate_1209, *[], **kwargs_1210)

# Assigning a type to the variable 'o' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'o', ndenumerate_call_result_1211)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'o' (line 9)
o_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'o')
# Obtaining the member 'next' of a type (line 9)
next_1213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), o_1212, 'next')
# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', next_1213)

# Assigning a Attribute to a Name (line 10):
# Getting the type of 'o' (line 10)
o_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'o')
# Obtaining the member '__next__' of a type (line 10)
next___1215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), o_1214, '__next__')
# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', next___1215)
# Getting the type of 'r' (line 11)
r_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'r')
# Getting the type of 'r2' (line 12)
r2_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'r2')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
