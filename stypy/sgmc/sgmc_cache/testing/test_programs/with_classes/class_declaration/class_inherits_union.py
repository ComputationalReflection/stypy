
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: class A:
3:     a_Att = 3
4: 
5:     def ma(self):
6:         return "A"
7: 
8: class B:
9:     def mb(self):
10:         return "B"
11: 
12: if True:
13:     base = A
14: else:
15:     base = B
16: 
17: 
18: class Simple(base):
19:     sample_att = 3
20:     (a,b) = (6,7)
21: 
22:     def from_a(self):
23:         return self.a_Att
24: 
25:     def sample_method(self):
26:         self.att = "sample"
27:         return self.att
28: 
29: 
30: x = Simple()
31: y = x.sample_method()
32: z = x.ma()
33: w = x.mb()
34: k = x.from_a()
35: 
36: 
37: 
38: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'A' class

class A:
    
    # Assigning a Num to a Name (line 3):

    @norecursion
    def ma(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ma'
        module_type_store = module_type_store.open_function_context('ma', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        A.ma.__dict__.__setitem__('stypy_localization', localization)
        A.ma.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        A.ma.__dict__.__setitem__('stypy_type_store', module_type_store)
        A.ma.__dict__.__setitem__('stypy_function_name', 'A.ma')
        A.ma.__dict__.__setitem__('stypy_param_names_list', [])
        A.ma.__dict__.__setitem__('stypy_varargs_param_name', None)
        A.ma.__dict__.__setitem__('stypy_kwargs_param_name', None)
        A.ma.__dict__.__setitem__('stypy_call_defaults', defaults)
        A.ma.__dict__.__setitem__('stypy_call_varargs', varargs)
        A.ma.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        A.ma.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'A.ma', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ma', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ma(...)' code ##################

        str_2465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', str_2465)
        
        # ################# End of 'ma(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ma' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_2466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2466)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ma'
        return stypy_return_type_2466


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'A.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'A' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'A', A)

# Assigning a Num to a Name (line 3):
int_2467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 12), 'int')
# Getting the type of 'A'
A_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'A')
# Setting the type of the member 'a_Att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), A_2468, 'a_Att', int_2467)
# Declaration of the 'B' class

class B:

    @norecursion
    def mb(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mb'
        module_type_store = module_type_store.open_function_context('mb', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        B.mb.__dict__.__setitem__('stypy_localization', localization)
        B.mb.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        B.mb.__dict__.__setitem__('stypy_type_store', module_type_store)
        B.mb.__dict__.__setitem__('stypy_function_name', 'B.mb')
        B.mb.__dict__.__setitem__('stypy_param_names_list', [])
        B.mb.__dict__.__setitem__('stypy_varargs_param_name', None)
        B.mb.__dict__.__setitem__('stypy_kwargs_param_name', None)
        B.mb.__dict__.__setitem__('stypy_call_defaults', defaults)
        B.mb.__dict__.__setitem__('stypy_call_varargs', varargs)
        B.mb.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        B.mb.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'B.mb', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mb', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mb(...)' code ##################

        str_2469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'str', 'B')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', str_2469)
        
        # ################# End of 'mb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mb' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mb'
        return stypy_return_type_2470


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'B.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'B' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'B', B)

# Getting the type of 'True' (line 12)
True_2471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 3), 'True')
# Testing the type of an if condition (line 12)
if_condition_2472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 0), True_2471)
# Assigning a type to the variable 'if_condition_2472' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'if_condition_2472', if_condition_2472)
# SSA begins for if statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 13):

# Assigning a Name to a Name (line 13):
# Getting the type of 'A' (line 13)
A_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'A')
# Assigning a type to the variable 'base' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'base', A_2473)
# SSA branch for the else part of an if statement (line 12)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'B' (line 15)
B_2474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'B')
# Assigning a type to the variable 'base' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'base', B_2474)
# SSA join for if statement (line 12)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Simple' class
# Getting the type of 'base' (line 18)
base_2475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'base')

class Simple(base_2475, ):
    
    # Assigning a Num to a Name (line 19):
    
    # Assigning a Tuple to a Tuple (line 20):

    @norecursion
    def from_a(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'from_a'
        module_type_store = module_type_store.open_function_context('from_a', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Simple.from_a.__dict__.__setitem__('stypy_localization', localization)
        Simple.from_a.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Simple.from_a.__dict__.__setitem__('stypy_type_store', module_type_store)
        Simple.from_a.__dict__.__setitem__('stypy_function_name', 'Simple.from_a')
        Simple.from_a.__dict__.__setitem__('stypy_param_names_list', [])
        Simple.from_a.__dict__.__setitem__('stypy_varargs_param_name', None)
        Simple.from_a.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Simple.from_a.__dict__.__setitem__('stypy_call_defaults', defaults)
        Simple.from_a.__dict__.__setitem__('stypy_call_varargs', varargs)
        Simple.from_a.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Simple.from_a.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Simple.from_a', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_a', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_a(...)' code ##################

        # Getting the type of 'self' (line 23)
        self_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'self')
        # Obtaining the member 'a_Att' of a type (line 23)
        a_Att_2477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 15), self_2476, 'a_Att')
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', a_Att_2477)
        
        # ################# End of 'from_a(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_a' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_2478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_a'
        return stypy_return_type_2478


    @norecursion
    def sample_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sample_method'
        module_type_store = module_type_store.open_function_context('sample_method', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
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

        
        # Assigning a Str to a Attribute (line 26):
        
        # Assigning a Str to a Attribute (line 26):
        str_2479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'sample')
        # Getting the type of 'self' (line 26)
        self_2480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'att' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_2480, 'att', str_2479)
        # Getting the type of 'self' (line 27)
        self_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'self')
        # Obtaining the member 'att' of a type (line 27)
        att_2482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), self_2481, 'att')
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', att_2482)
        
        # ################# End of 'sample_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sample_method' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2483)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sample_method'
        return stypy_return_type_2483


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 0, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'Simple' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'Simple', Simple)

# Assigning a Num to a Name (line 19):
int_2484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
# Getting the type of 'Simple'
Simple_2485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'sample_att' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2485, 'sample_att', int_2484)

# Assigning a Num to a Name (line 20):
int_2486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 13), 'int')
# Getting the type of 'Simple'
Simple_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2463' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2487, 'tuple_assignment_2463', int_2486)

# Assigning a Num to a Name (line 20):
int_2488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
# Getting the type of 'Simple'
Simple_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'tuple_assignment_2464' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2489, 'tuple_assignment_2464', int_2488)

# Assigning a Name to a Name (line 20):
# Getting the type of 'Simple'
Simple_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2463' of a type
tuple_assignment_2463_2491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2490, 'tuple_assignment_2463')
# Getting the type of 'Simple'
Simple_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'a' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2492, 'a', tuple_assignment_2463_2491)

# Assigning a Name to a Name (line 20):
# Getting the type of 'Simple'
Simple_2493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Obtaining the member 'tuple_assignment_2464' of a type
tuple_assignment_2464_2494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2493, 'tuple_assignment_2464')
# Getting the type of 'Simple'
Simple_2495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Simple')
# Setting the type of the member 'b' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Simple_2495, 'b', tuple_assignment_2464_2494)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to Simple(...): (line 30)
# Processing the call keyword arguments (line 30)
kwargs_2497 = {}
# Getting the type of 'Simple' (line 30)
Simple_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'Simple', False)
# Calling Simple(args, kwargs) (line 30)
Simple_call_result_2498 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), Simple_2496, *[], **kwargs_2497)

# Assigning a type to the variable 'x' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'x', Simple_call_result_2498)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to sample_method(...): (line 31)
# Processing the call keyword arguments (line 31)
kwargs_2501 = {}
# Getting the type of 'x' (line 31)
x_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'x', False)
# Obtaining the member 'sample_method' of a type (line 31)
sample_method_2500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), x_2499, 'sample_method')
# Calling sample_method(args, kwargs) (line 31)
sample_method_call_result_2502 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), sample_method_2500, *[], **kwargs_2501)

# Assigning a type to the variable 'y' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'y', sample_method_call_result_2502)

# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to ma(...): (line 32)
# Processing the call keyword arguments (line 32)
kwargs_2505 = {}
# Getting the type of 'x' (line 32)
x_2503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'x', False)
# Obtaining the member 'ma' of a type (line 32)
ma_2504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), x_2503, 'ma')
# Calling ma(args, kwargs) (line 32)
ma_call_result_2506 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), ma_2504, *[], **kwargs_2505)

# Assigning a type to the variable 'z' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'z', ma_call_result_2506)

# Assigning a Call to a Name (line 33):

# Assigning a Call to a Name (line 33):

# Call to mb(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_2509 = {}
# Getting the type of 'x' (line 33)
x_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'x', False)
# Obtaining the member 'mb' of a type (line 33)
mb_2508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), x_2507, 'mb')
# Calling mb(args, kwargs) (line 33)
mb_call_result_2510 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), mb_2508, *[], **kwargs_2509)

# Assigning a type to the variable 'w' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'w', mb_call_result_2510)

# Assigning a Call to a Name (line 34):

# Assigning a Call to a Name (line 34):

# Call to from_a(...): (line 34)
# Processing the call keyword arguments (line 34)
kwargs_2513 = {}
# Getting the type of 'x' (line 34)
x_2511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'x', False)
# Obtaining the member 'from_a' of a type (line 34)
from_a_2512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), x_2511, 'from_a')
# Calling from_a(args, kwargs) (line 34)
from_a_call_result_2514 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), from_a_2512, *[], **kwargs_2513)

# Assigning a type to the variable 'k' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'k', from_a_call_result_2514)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
