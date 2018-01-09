
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class Simple:
4:     sample_att = 3
5:     (a,b) = (6,7)
6: 
7:     def sample_method(self):
8:         self.att = "sample"
9: 
10: 
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'Simple' class

class Simple:
    
    # Assigning a Num to a Name (line 4):
    
    # Assigning a Tuple to a Tuple (line 5):

    @norecursion
    def sample_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sample_method'
        module_type_store = module_type_store.open_function_context('sample_method', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
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

        
        # Assigning a Str to a Attribute (line 8):
        
        # Assigning a Str to a Attribute (line 8):
        str_2331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'sample')
        # Getting the type of 'self' (line 8)
        self_2332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member 'att' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_2332, 'att', str_2331)
        
        # ################# End of 'sample_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sample_method' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_2333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2333)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sample_method'
        return stypy_return_type_2333


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
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'Simple', Simple)

# Assigning a Num to a Name (line 4):
int_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 17), 'int')
# Getting the type of 'Simple'
Simple_2335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'sample_att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2335, 'sample_att', int_2334)

# Assigning a Num to a Name (line 5):
int_2336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Getting the type of 'Simple'
Simple_2337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2329' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2337, 'tuple_assignment_2329', int_2336)

# Assigning a Num to a Name (line 5):
int_2338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
# Getting the type of 'Simple'
Simple_2339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2330' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2339, 'tuple_assignment_2330', int_2338)

# Assigning a Name to a Name (line 5):
# Getting the type of 'Simple'
Simple_2340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2329' of a type
tuple_assignment_2329_2341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2340, 'tuple_assignment_2329')
# Getting the type of 'Simple'
Simple_2342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2342, 'a', tuple_assignment_2329_2341)

# Assigning a Name to a Name (line 5):
# Getting the type of 'Simple'
Simple_2343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2330' of a type
tuple_assignment_2330_2344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2343, 'tuple_assignment_2330')
# Getting the type of 'Simple'
Simple_2345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'b' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2345, 'b', tuple_assignment_2330_2344)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
