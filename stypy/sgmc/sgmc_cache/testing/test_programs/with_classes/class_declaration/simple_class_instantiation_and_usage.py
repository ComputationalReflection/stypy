
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class Simple:
4:     sample_att = 3
5:     (a,b) = (6,7)
6: 
7:     def __init__(self):
8:         pass
9: 
10:     def sample_method(self):
11:         self.att = "sample"
12: 
13: 
14: s = Simple()
15: 
16: s.sample_method()
17: 
18: result = s.att
19: result2 = s.b
20: 
21: 

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
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
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


    @norecursion
    def sample_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sample_method'
        module_type_store = module_type_store.open_function_context('sample_method', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
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

        
        # Assigning a Str to a Attribute (line 11):
        
        # Assigning a Str to a Attribute (line 11):
        str_2348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'str', 'sample')
        # Getting the type of 'self' (line 11)
        self_2349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self')
        # Setting the type of the member 'att' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_2349, 'att', str_2348)
        
        # ################# End of 'sample_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sample_method' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_2350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2350)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sample_method'
        return stypy_return_type_2350


# Assigning a type to the variable 'Simple' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'Simple', Simple)

# Assigning a Num to a Name (line 4):
int_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 17), 'int')
# Getting the type of 'Simple'
Simple_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'sample_att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2352, 'sample_att', int_2351)

# Assigning a Num to a Name (line 5):
int_2353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Getting the type of 'Simple'
Simple_2354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2346' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2354, 'tuple_assignment_2346', int_2353)

# Assigning a Num to a Name (line 5):
int_2355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
# Getting the type of 'Simple'
Simple_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2347' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2356, 'tuple_assignment_2347', int_2355)

# Assigning a Name to a Name (line 5):
# Getting the type of 'Simple'
Simple_2357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2346' of a type
tuple_assignment_2346_2358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2357, 'tuple_assignment_2346')
# Getting the type of 'Simple'
Simple_2359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2359, 'a', tuple_assignment_2346_2358)

# Assigning a Name to a Name (line 5):
# Getting the type of 'Simple'
Simple_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2347' of a type
tuple_assignment_2347_2361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2360, 'tuple_assignment_2347')
# Getting the type of 'Simple'
Simple_2362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'b' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2362, 'b', tuple_assignment_2347_2361)

# Assigning a Call to a Name (line 14):

# Assigning a Call to a Name (line 14):

# Call to Simple(...): (line 14)
# Processing the call keyword arguments (line 14)
kwargs_2364 = {}
# Getting the type of 'Simple' (line 14)
Simple_2363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Simple', False)
# Calling Simple(args, kwargs) (line 14)
Simple_call_result_2365 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), Simple_2363, *[], **kwargs_2364)

# Assigning a type to the variable 's' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 's', Simple_call_result_2365)

# Call to sample_method(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_2368 = {}
# Getting the type of 's' (line 16)
s_2366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 's', False)
# Obtaining the member 'sample_method' of a type (line 16)
sample_method_2367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), s_2366, 'sample_method')
# Calling sample_method(args, kwargs) (line 16)
sample_method_call_result_2369 = invoke(stypy.reporting.localization.Localization(__file__, 16, 0), sample_method_2367, *[], **kwargs_2368)


# Assigning a Attribute to a Name (line 18):

# Assigning a Attribute to a Name (line 18):
# Getting the type of 's' (line 18)
s_2370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 's')
# Obtaining the member 'att' of a type (line 18)
att_2371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 9), s_2370, 'att')
# Assigning a type to the variable 'result' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'result', att_2371)

# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of 's' (line 19)
s_2372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 's')
# Obtaining the member 'b' of a type (line 19)
b_2373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), s_2372, 'b')
# Assigning a type to the variable 'result2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'result2', b_2373)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
