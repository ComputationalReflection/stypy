
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: class C:
4:     def __init__(self):
5:         pass
6: 
7:     def method(self):
8:         self.r = "str"
9: 
10: c = C()
11: 
12: c.method()
13: 
14: x = c.r == 5
15: 
16: 
17: class Counter:
18:     count = 0
19: 
20:     def __init__(self):
21:         pass
22: 
23:     def inc(self, value):
24:         self.count += value
25:         return self.count
26: 
27: obj = Counter()
28: sum = obj.inc(1) + obj.inc(0.2)
29: 
30: if obj:
31:     resul = obj.inc(1)
32: else:
33:     resul = obj.inc(0.5)
34: 
35: 
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'C' class

class C:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 4, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.__init__', [], None, None, defaults, varargs, kwargs)

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
    def method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        C.method.__dict__.__setitem__('stypy_localization', localization)
        C.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        C.method.__dict__.__setitem__('stypy_type_store', module_type_store)
        C.method.__dict__.__setitem__('stypy_function_name', 'C.method')
        C.method.__dict__.__setitem__('stypy_param_names_list', [])
        C.method.__dict__.__setitem__('stypy_varargs_param_name', None)
        C.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        C.method.__dict__.__setitem__('stypy_call_defaults', defaults)
        C.method.__dict__.__setitem__('stypy_call_varargs', varargs)
        C.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        C.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'C.method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        
        # Assigning a Str to a Attribute (line 8):
        str_2279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'str', 'str')
        # Getting the type of 'self' (line 8)
        self_2280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member 'r' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_2280, 'r', str_2279)
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_2281


# Assigning a type to the variable 'C' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'C', C)

# Assigning a Call to a Name (line 10):

# Call to C(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_2283 = {}
# Getting the type of 'C' (line 10)
C_2282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'C', False)
# Calling C(args, kwargs) (line 10)
C_call_result_2284 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), C_2282, *[], **kwargs_2283)

# Assigning a type to the variable 'c' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'c', C_call_result_2284)

# Call to method(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_2287 = {}
# Getting the type of 'c' (line 12)
c_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'c', False)
# Obtaining the member 'method' of a type (line 12)
method_2286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), c_2285, 'method')
# Calling method(args, kwargs) (line 12)
method_call_result_2288 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), method_2286, *[], **kwargs_2287)


# Assigning a Compare to a Name (line 14):

# Getting the type of 'c' (line 14)
c_2289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'c')
# Obtaining the member 'r' of a type (line 14)
r_2290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), c_2289, 'r')
int_2291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'int')
# Applying the binary operator '==' (line 14)
result_eq_2292 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 4), '==', r_2290, int_2291)

# Assigning a type to the variable 'x' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'x', result_eq_2292)
# Declaration of the 'Counter' class

class Counter:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Counter.__init__', [], None, None, defaults, varargs, kwargs)

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
    def inc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inc'
        module_type_store = module_type_store.open_function_context('inc', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Counter.inc.__dict__.__setitem__('stypy_localization', localization)
        Counter.inc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Counter.inc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Counter.inc.__dict__.__setitem__('stypy_function_name', 'Counter.inc')
        Counter.inc.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Counter.inc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Counter.inc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Counter.inc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Counter.inc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Counter.inc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Counter.inc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Counter.inc', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inc', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inc(...)' code ##################

        
        # Getting the type of 'self' (line 24)
        self_2293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Obtaining the member 'count' of a type (line 24)
        count_2294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_2293, 'count')
        # Getting the type of 'value' (line 24)
        value_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'value')
        # Applying the binary operator '+=' (line 24)
        result_iadd_2296 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), '+=', count_2294, value_2295)
        # Getting the type of 'self' (line 24)
        self_2297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'count' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_2297, 'count', result_iadd_2296)
        
        # Getting the type of 'self' (line 25)
        self_2298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'self')
        # Obtaining the member 'count' of a type (line 25)
        count_2299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), self_2298, 'count')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', count_2299)
        
        # ################# End of 'inc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inc' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inc'
        return stypy_return_type_2300


# Assigning a type to the variable 'Counter' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Counter', Counter)

# Assigning a Num to a Name (line 18):
int_2301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'int')
# Getting the type of 'Counter'
Counter_2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Counter')
# Setting the type of the member 'count' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Counter_2302, 'count', int_2301)

# Assigning a Call to a Name (line 27):

# Call to Counter(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_2304 = {}
# Getting the type of 'Counter' (line 27)
Counter_2303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'Counter', False)
# Calling Counter(args, kwargs) (line 27)
Counter_call_result_2305 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), Counter_2303, *[], **kwargs_2304)

# Assigning a type to the variable 'obj' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'obj', Counter_call_result_2305)

# Assigning a BinOp to a Name (line 28):

# Call to inc(...): (line 28)
# Processing the call arguments (line 28)
int_2308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'int')
# Processing the call keyword arguments (line 28)
kwargs_2309 = {}
# Getting the type of 'obj' (line 28)
obj_2306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'obj', False)
# Obtaining the member 'inc' of a type (line 28)
inc_2307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), obj_2306, 'inc')
# Calling inc(args, kwargs) (line 28)
inc_call_result_2310 = invoke(stypy.reporting.localization.Localization(__file__, 28, 6), inc_2307, *[int_2308], **kwargs_2309)


# Call to inc(...): (line 28)
# Processing the call arguments (line 28)
float_2313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'float')
# Processing the call keyword arguments (line 28)
kwargs_2314 = {}
# Getting the type of 'obj' (line 28)
obj_2311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'obj', False)
# Obtaining the member 'inc' of a type (line 28)
inc_2312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 19), obj_2311, 'inc')
# Calling inc(args, kwargs) (line 28)
inc_call_result_2315 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), inc_2312, *[float_2313], **kwargs_2314)

# Applying the binary operator '+' (line 28)
result_add_2316 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 6), '+', inc_call_result_2310, inc_call_result_2315)

# Assigning a type to the variable 'sum' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'sum', result_add_2316)

# Getting the type of 'obj' (line 30)
obj_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 3), 'obj')
# Testing the type of an if condition (line 30)
if_condition_2318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 0), obj_2317)
# Assigning a type to the variable 'if_condition_2318' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'if_condition_2318', if_condition_2318)
# SSA begins for if statement (line 30)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Name (line 31):

# Call to inc(...): (line 31)
# Processing the call arguments (line 31)
int_2321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
# Processing the call keyword arguments (line 31)
kwargs_2322 = {}
# Getting the type of 'obj' (line 31)
obj_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'obj', False)
# Obtaining the member 'inc' of a type (line 31)
inc_2320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), obj_2319, 'inc')
# Calling inc(args, kwargs) (line 31)
inc_call_result_2323 = invoke(stypy.reporting.localization.Localization(__file__, 31, 12), inc_2320, *[int_2321], **kwargs_2322)

# Assigning a type to the variable 'resul' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'resul', inc_call_result_2323)
# SSA branch for the else part of an if statement (line 30)
module_type_store.open_ssa_branch('else')

# Assigning a Call to a Name (line 33):

# Call to inc(...): (line 33)
# Processing the call arguments (line 33)
float_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'float')
# Processing the call keyword arguments (line 33)
kwargs_2327 = {}
# Getting the type of 'obj' (line 33)
obj_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'obj', False)
# Obtaining the member 'inc' of a type (line 33)
inc_2325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), obj_2324, 'inc')
# Calling inc(args, kwargs) (line 33)
inc_call_result_2328 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), inc_2325, *[float_2326], **kwargs_2327)

# Assigning a type to the variable 'resul' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'resul', inc_call_result_2328)
# SSA join for if statement (line 30)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
