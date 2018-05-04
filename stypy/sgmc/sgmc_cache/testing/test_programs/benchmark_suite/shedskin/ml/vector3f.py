
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from math import sqrt
8: 
9: def Vector3f_str(s):
10:     split = s.lstrip(' (').rstrip(') ').split()
11:     return Vector3f(float(split[0]), float(split[1]), float(split[2]))
12: 
13: def Vector3f_seq(seq):
14:     return Vector3f(seq[0], seq[1], seq[2])
15: 
16: def Vector3f_scalar(s):
17:     return Vector3f(s, s, s)
18: 
19: class Vector3f(object):
20: 
21:     def __init__(self, x, y, z):
22:         self.x, self.y, self.z = float(x), float(y), float(z)
23: 
24:     def as_list(self):
25:         return [self.x, self.y, self.z]
26: 
27:     def copy(self):
28:         return Vector3f(self.x, self.y, self.z)
29: 
30:     def __getitem__(self, key):
31:         if key == 2:
32:             return self.z
33:         elif key == 1:
34:             return self.y
35:         else:
36:             return self.x
37: 
38:     def __neg__(self):
39:         return Vector3f(-self.x, -self.y, -self.z)
40: 
41:     def __add__(self, other):
42:         return Vector3f(self.x + other.x, self.y + other.y, self.z + other.z)
43: 
44:     def __sub__(self, other):
45:         return Vector3f(self.x - other.x, self.y - other.y, self.z - other.z)
46: 
47:     def __mul__(self, other):
48:         return Vector3f(self.x * other, self.y * other, self.z * other)
49: 
50:     def mul(self, other):
51:         return Vector3f(self.x * other.x, self.y * other.y, self.z * other.z)
52: 
53:     def is_zero(self):
54:         return self.x == 0.0 and self.y == 0.0 and self.z == 0.0
55: 
56:     def dot(self, other):
57:         return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
58: 
59:     def unitize(self):
60:         length = sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
61:         one_over_length = 1.0 / length if length != 0.0 else 0.0
62:         return Vector3f(self.x * one_over_length, self.y * one_over_length, self.z * one_over_length)
63: 
64:     def cross(self, other):
65:         return Vector3f((self.y * other.z) - (self.z * other.y),
66:                         (self.z * other.x) - (self.x * other.z),
67:                         (self.x * other.y) - (self.y * other.x))
68: 
69:     def clamped(self, lo, hi):
70:         return Vector3f(min(max(self.x, lo.x), hi.x),
71:                         min(max(self.y, lo.y), hi.y),
72:                         min(max(self.z, lo.z), hi.z))
73: 
74: ZERO = Vector3f_scalar(0.0)
75: ONE = Vector3f_scalar(1.0)
76: MAX = Vector3f_scalar(1.797e308)
77: ##ALMOST_ONE?
78: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import sqrt' statement (line 7)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])


@norecursion
def Vector3f_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Vector3f_str'
    module_type_store = module_type_store.open_function_context('Vector3f_str', 9, 0, False)
    
    # Passed parameters checking function
    Vector3f_str.stypy_localization = localization
    Vector3f_str.stypy_type_of_self = None
    Vector3f_str.stypy_type_store = module_type_store
    Vector3f_str.stypy_function_name = 'Vector3f_str'
    Vector3f_str.stypy_param_names_list = ['s']
    Vector3f_str.stypy_varargs_param_name = None
    Vector3f_str.stypy_kwargs_param_name = None
    Vector3f_str.stypy_call_defaults = defaults
    Vector3f_str.stypy_call_varargs = varargs
    Vector3f_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Vector3f_str', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Vector3f_str', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Vector3f_str(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Assigning a Call to a Name (line 10):
    
    # Call to split(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_2340 = {}
    
    # Call to rstrip(...): (line 10)
    # Processing the call arguments (line 10)
    str_2336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'str', ') ')
    # Processing the call keyword arguments (line 10)
    kwargs_2337 = {}
    
    # Call to lstrip(...): (line 10)
    # Processing the call arguments (line 10)
    str_2332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', ' (')
    # Processing the call keyword arguments (line 10)
    kwargs_2333 = {}
    # Getting the type of 's' (line 10)
    s_2330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 's', False)
    # Obtaining the member 'lstrip' of a type (line 10)
    lstrip_2331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), s_2330, 'lstrip')
    # Calling lstrip(args, kwargs) (line 10)
    lstrip_call_result_2334 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), lstrip_2331, *[str_2332], **kwargs_2333)
    
    # Obtaining the member 'rstrip' of a type (line 10)
    rstrip_2335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), lstrip_call_result_2334, 'rstrip')
    # Calling rstrip(args, kwargs) (line 10)
    rstrip_call_result_2338 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), rstrip_2335, *[str_2336], **kwargs_2337)
    
    # Obtaining the member 'split' of a type (line 10)
    split_2339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), rstrip_call_result_2338, 'split')
    # Calling split(args, kwargs) (line 10)
    split_call_result_2341 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), split_2339, *[], **kwargs_2340)
    
    # Assigning a type to the variable 'split' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'split', split_call_result_2341)
    
    # Call to Vector3f(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'int')
    # Getting the type of 'split' (line 11)
    split_2345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 26), split_2345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2347 = invoke(stypy.reporting.localization.Localization(__file__, 11, 26), getitem___2346, int_2344)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2348 = {}
    # Getting the type of 'float' (line 11)
    float_2343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2349 = invoke(stypy.reporting.localization.Localization(__file__, 11, 20), float_2343, *[subscript_call_result_2347], **kwargs_2348)
    
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'int')
    # Getting the type of 'split' (line 11)
    split_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 43), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 43), split_2352, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2354 = invoke(stypy.reporting.localization.Localization(__file__, 11, 43), getitem___2353, int_2351)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2355 = {}
    # Getting the type of 'float' (line 11)
    float_2350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 37), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2356 = invoke(stypy.reporting.localization.Localization(__file__, 11, 37), float_2350, *[subscript_call_result_2354], **kwargs_2355)
    
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 66), 'int')
    # Getting the type of 'split' (line 11)
    split_2359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 60), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 60), split_2359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2361 = invoke(stypy.reporting.localization.Localization(__file__, 11, 60), getitem___2360, int_2358)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2362 = {}
    # Getting the type of 'float' (line 11)
    float_2357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 54), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2363 = invoke(stypy.reporting.localization.Localization(__file__, 11, 54), float_2357, *[subscript_call_result_2361], **kwargs_2362)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2364 = {}
    # Getting the type of 'Vector3f' (line 11)
    Vector3f_2342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 11)
    Vector3f_call_result_2365 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), Vector3f_2342, *[float_call_result_2349, float_call_result_2356, float_call_result_2363], **kwargs_2364)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', Vector3f_call_result_2365)
    
    # ################# End of 'Vector3f_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_str' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_2366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_str'
    return stypy_return_type_2366

# Assigning a type to the variable 'Vector3f_str' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'Vector3f_str', Vector3f_str)

@norecursion
def Vector3f_seq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Vector3f_seq'
    module_type_store = module_type_store.open_function_context('Vector3f_seq', 13, 0, False)
    
    # Passed parameters checking function
    Vector3f_seq.stypy_localization = localization
    Vector3f_seq.stypy_type_of_self = None
    Vector3f_seq.stypy_type_store = module_type_store
    Vector3f_seq.stypy_function_name = 'Vector3f_seq'
    Vector3f_seq.stypy_param_names_list = ['seq']
    Vector3f_seq.stypy_varargs_param_name = None
    Vector3f_seq.stypy_kwargs_param_name = None
    Vector3f_seq.stypy_call_defaults = defaults
    Vector3f_seq.stypy_call_varargs = varargs
    Vector3f_seq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Vector3f_seq', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Vector3f_seq', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Vector3f_seq(...)' code ##################

    
    # Call to Vector3f(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Obtaining the type of the subscript
    int_2368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 20), seq_2369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2371 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), getitem___2370, int_2368)
    
    
    # Obtaining the type of the subscript
    int_2372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), seq_2373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2375 = invoke(stypy.reporting.localization.Localization(__file__, 14, 28), getitem___2374, int_2372)
    
    
    # Obtaining the type of the subscript
    int_2376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 36), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 36), seq_2377, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2379 = invoke(stypy.reporting.localization.Localization(__file__, 14, 36), getitem___2378, int_2376)
    
    # Processing the call keyword arguments (line 14)
    kwargs_2380 = {}
    # Getting the type of 'Vector3f' (line 14)
    Vector3f_2367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 14)
    Vector3f_call_result_2381 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), Vector3f_2367, *[subscript_call_result_2371, subscript_call_result_2375, subscript_call_result_2379], **kwargs_2380)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', Vector3f_call_result_2381)
    
    # ################# End of 'Vector3f_seq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_seq' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_2382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2382)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_seq'
    return stypy_return_type_2382

# Assigning a type to the variable 'Vector3f_seq' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'Vector3f_seq', Vector3f_seq)

@norecursion
def Vector3f_scalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Vector3f_scalar'
    module_type_store = module_type_store.open_function_context('Vector3f_scalar', 16, 0, False)
    
    # Passed parameters checking function
    Vector3f_scalar.stypy_localization = localization
    Vector3f_scalar.stypy_type_of_self = None
    Vector3f_scalar.stypy_type_store = module_type_store
    Vector3f_scalar.stypy_function_name = 'Vector3f_scalar'
    Vector3f_scalar.stypy_param_names_list = ['s']
    Vector3f_scalar.stypy_varargs_param_name = None
    Vector3f_scalar.stypy_kwargs_param_name = None
    Vector3f_scalar.stypy_call_defaults = defaults
    Vector3f_scalar.stypy_call_varargs = varargs
    Vector3f_scalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Vector3f_scalar', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Vector3f_scalar', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Vector3f_scalar(...)' code ##################

    
    # Call to Vector3f(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 's' (line 17)
    s_2384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 's', False)
    # Getting the type of 's' (line 17)
    s_2385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 's', False)
    # Getting the type of 's' (line 17)
    s_2386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 's', False)
    # Processing the call keyword arguments (line 17)
    kwargs_2387 = {}
    # Getting the type of 'Vector3f' (line 17)
    Vector3f_2383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 17)
    Vector3f_call_result_2388 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), Vector3f_2383, *[s_2384, s_2385, s_2386], **kwargs_2387)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', Vector3f_call_result_2388)
    
    # ################# End of 'Vector3f_scalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_scalar' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_2389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2389)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_scalar'
    return stypy_return_type_2389

# Assigning a type to the variable 'Vector3f_scalar' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'Vector3f_scalar', Vector3f_scalar)
# Declaration of the 'Vector3f' class

class Vector3f(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to float(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'x' (line 22)
        x_2391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2392 = {}
        # Getting the type of 'float' (line 22)
        float_2390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2393 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), float_2390, *[x_2391], **kwargs_2392)
        
        # Assigning a type to the variable 'tuple_assignment_2327' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2327', float_call_result_2393)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to float(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'y' (line 22)
        y_2395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'y', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2396 = {}
        # Getting the type of 'float' (line 22)
        float_2394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2397 = invoke(stypy.reporting.localization.Localization(__file__, 22, 43), float_2394, *[y_2395], **kwargs_2396)
        
        # Assigning a type to the variable 'tuple_assignment_2328' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2328', float_call_result_2397)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to float(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'z' (line 22)
        z_2399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 59), 'z', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2400 = {}
        # Getting the type of 'float' (line 22)
        float_2398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2401 = invoke(stypy.reporting.localization.Localization(__file__, 22, 53), float_2398, *[z_2399], **kwargs_2400)
        
        # Assigning a type to the variable 'tuple_assignment_2329' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2329', float_call_result_2401)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2327' (line 22)
        tuple_assignment_2327_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2327')
        # Getting the type of 'self' (line 22)
        self_2403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'x' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_2403, 'x', tuple_assignment_2327_2402)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2328' (line 22)
        tuple_assignment_2328_2404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2328')
        # Getting the type of 'self' (line 22)
        self_2405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
        # Setting the type of the member 'y' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_2405, 'y', tuple_assignment_2328_2404)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2329' (line 22)
        tuple_assignment_2329_2406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2329')
        # Getting the type of 'self' (line 22)
        self_2407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'self')
        # Setting the type of the member 'z' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), self_2407, 'z', tuple_assignment_2329_2406)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def as_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'as_list'
        module_type_store = module_type_store.open_function_context('as_list', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.as_list.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.as_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.as_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.as_list.__dict__.__setitem__('stypy_function_name', 'Vector3f.as_list')
        Vector3f.as_list.__dict__.__setitem__('stypy_param_names_list', [])
        Vector3f.as_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.as_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.as_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.as_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.as_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.as_list.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.as_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'as_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'as_list(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_2408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
        # Obtaining the member 'x' of a type (line 25)
        x_2410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_2409, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2408, x_2410)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'self')
        # Obtaining the member 'y' of a type (line 25)
        y_2412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), self_2411, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2408, y_2412)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'self')
        # Obtaining the member 'z' of a type (line 25)
        z_2414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 32), self_2413, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2408, z_2414)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', list_2408)
        
        # ################# End of 'as_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'as_list' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_2415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'as_list'
        return stypy_return_type_2415


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.copy.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.copy.__dict__.__setitem__('stypy_function_name', 'Vector3f.copy')
        Vector3f.copy.__dict__.__setitem__('stypy_param_names_list', [])
        Vector3f.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        
        # Call to Vector3f(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 28)
        x_2418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_2417, 'x')
        # Getting the type of 'self' (line 28)
        self_2419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 28)
        y_2420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 32), self_2419, 'y')
        # Getting the type of 'self' (line 28)
        self_2421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'self', False)
        # Obtaining the member 'z' of a type (line 28)
        z_2422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), self_2421, 'z')
        # Processing the call keyword arguments (line 28)
        kwargs_2423 = {}
        # Getting the type of 'Vector3f' (line 28)
        Vector3f_2416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 28)
        Vector3f_call_result_2424 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), Vector3f_2416, *[x_2418, y_2420, z_2422], **kwargs_2423)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', Vector3f_call_result_2424)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_2425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2425)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_2425


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_function_name', 'Vector3f.__getitem__')
        Vector3f.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        Vector3f.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Getting the type of 'key' (line 31)
        key_2426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'key')
        int_2427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
        # Applying the binary operator '==' (line 31)
        result_eq_2428 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '==', key_2426, int_2427)
        
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), result_eq_2428):
            
            # Getting the type of 'key' (line 33)
            key_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'key')
            int_2433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
            # Applying the binary operator '==' (line 33)
            result_eq_2434 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '==', key_2432, int_2433)
            
            # Testing if the type of an if condition is none (line 33)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2434):
                # Getting the type of 'self' (line 36)
                self_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2438, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2439)
            else:
                
                # Testing the type of an if condition (line 33)
                if_condition_2435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2434)
                # Assigning a type to the variable 'if_condition_2435' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'if_condition_2435', if_condition_2435)
                # SSA begins for if statement (line 33)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 34)
                self_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
                # Obtaining the member 'y' of a type (line 34)
                y_2437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_2436, 'y')
                # Assigning a type to the variable 'stypy_return_type' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', y_2437)
                # SSA branch for the else part of an if statement (line 33)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 36)
                self_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2438, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2439)
                # SSA join for if statement (line 33)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_2429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_eq_2428)
            # Assigning a type to the variable 'if_condition_2429' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_2429', if_condition_2429)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 32)
            self_2430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'self')
            # Obtaining the member 'z' of a type (line 32)
            z_2431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), self_2430, 'z')
            # Assigning a type to the variable 'stypy_return_type' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', z_2431)
            # SSA branch for the else part of an if statement (line 31)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'key' (line 33)
            key_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'key')
            int_2433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
            # Applying the binary operator '==' (line 33)
            result_eq_2434 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '==', key_2432, int_2433)
            
            # Testing if the type of an if condition is none (line 33)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2434):
                # Getting the type of 'self' (line 36)
                self_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2438, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2439)
            else:
                
                # Testing the type of an if condition (line 33)
                if_condition_2435 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2434)
                # Assigning a type to the variable 'if_condition_2435' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'if_condition_2435', if_condition_2435)
                # SSA begins for if statement (line 33)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 34)
                self_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
                # Obtaining the member 'y' of a type (line 34)
                y_2437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_2436, 'y')
                # Assigning a type to the variable 'stypy_return_type' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', y_2437)
                # SSA branch for the else part of an if statement (line 33)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 36)
                self_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2438, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2439)
                # SSA join for if statement (line 33)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_2440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_2440


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.__neg__.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.__neg__.__dict__.__setitem__('stypy_function_name', 'Vector3f.__neg__')
        Vector3f.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        Vector3f.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to Vector3f(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Getting the type of 'self' (line 39)
        self_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 39)
        x_2443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), self_2442, 'x')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2444 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 24), 'usub', x_2443)
        
        
        # Getting the type of 'self' (line 39)
        self_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'self', False)
        # Obtaining the member 'y' of a type (line 39)
        y_2446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 34), self_2445, 'y')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2447 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 33), 'usub', y_2446)
        
        
        # Getting the type of 'self' (line 39)
        self_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'self', False)
        # Obtaining the member 'z' of a type (line 39)
        z_2449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 43), self_2448, 'z')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2450 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 42), 'usub', z_2449)
        
        # Processing the call keyword arguments (line 39)
        kwargs_2451 = {}
        # Getting the type of 'Vector3f' (line 39)
        Vector3f_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 39)
        Vector3f_call_result_2452 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), Vector3f_2441, *[result___neg___2444, result___neg___2447, result___neg___2450], **kwargs_2451)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', Vector3f_call_result_2452)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_2453


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.__add__.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.__add__.__dict__.__setitem__('stypy_function_name', 'Vector3f.__add__')
        Vector3f.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__add__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to Vector3f(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 42)
        x_2456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), self_2455, 'x')
        # Getting the type of 'other' (line 42)
        other_2457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 42)
        x_2458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 33), other_2457, 'x')
        # Applying the binary operator '+' (line 42)
        result_add_2459 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 24), '+', x_2456, x_2458)
        
        # Getting the type of 'self' (line 42)
        self_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 42)
        y_2461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), self_2460, 'y')
        # Getting the type of 'other' (line 42)
        other_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 42)
        y_2463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 51), other_2462, 'y')
        # Applying the binary operator '+' (line 42)
        result_add_2464 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 42), '+', y_2461, y_2463)
        
        # Getting the type of 'self' (line 42)
        self_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 42)
        z_2466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 60), self_2465, 'z')
        # Getting the type of 'other' (line 42)
        other_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 42)
        z_2468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 69), other_2467, 'z')
        # Applying the binary operator '+' (line 42)
        result_add_2469 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 60), '+', z_2466, z_2468)
        
        # Processing the call keyword arguments (line 42)
        kwargs_2470 = {}
        # Getting the type of 'Vector3f' (line 42)
        Vector3f_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 42)
        Vector3f_call_result_2471 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), Vector3f_2454, *[result_add_2459, result_add_2464, result_add_2469], **kwargs_2470)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', Vector3f_call_result_2471)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_2472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2472)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_2472


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.__sub__.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.__sub__.__dict__.__setitem__('stypy_function_name', 'Vector3f.__sub__')
        Vector3f.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        # Call to Vector3f(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'self' (line 45)
        self_2474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 45)
        x_2475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), self_2474, 'x')
        # Getting the type of 'other' (line 45)
        other_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 45)
        x_2477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 33), other_2476, 'x')
        # Applying the binary operator '-' (line 45)
        result_sub_2478 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 24), '-', x_2475, x_2477)
        
        # Getting the type of 'self' (line 45)
        self_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 45)
        y_2480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 42), self_2479, 'y')
        # Getting the type of 'other' (line 45)
        other_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 45)
        y_2482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 51), other_2481, 'y')
        # Applying the binary operator '-' (line 45)
        result_sub_2483 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 42), '-', y_2480, y_2482)
        
        # Getting the type of 'self' (line 45)
        self_2484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 45)
        z_2485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 60), self_2484, 'z')
        # Getting the type of 'other' (line 45)
        other_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 45)
        z_2487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 69), other_2486, 'z')
        # Applying the binary operator '-' (line 45)
        result_sub_2488 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 60), '-', z_2485, z_2487)
        
        # Processing the call keyword arguments (line 45)
        kwargs_2489 = {}
        # Getting the type of 'Vector3f' (line 45)
        Vector3f_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 45)
        Vector3f_call_result_2490 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), Vector3f_2473, *[result_sub_2478, result_sub_2483, result_sub_2488], **kwargs_2489)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', Vector3f_call_result_2490)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_2491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2491)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_2491


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.__mul__.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.__mul__.__dict__.__setitem__('stypy_function_name', 'Vector3f.__mul__')
        Vector3f.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Call to Vector3f(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_2493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 48)
        x_2494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), self_2493, 'x')
        # Getting the type of 'other' (line 48)
        other_2495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2496 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 24), '*', x_2494, other_2495)
        
        # Getting the type of 'self' (line 48)
        self_2497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'self', False)
        # Obtaining the member 'y' of a type (line 48)
        y_2498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 40), self_2497, 'y')
        # Getting the type of 'other' (line 48)
        other_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2500 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 40), '*', y_2498, other_2499)
        
        # Getting the type of 'self' (line 48)
        self_2501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 56), 'self', False)
        # Obtaining the member 'z' of a type (line 48)
        z_2502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 56), self_2501, 'z')
        # Getting the type of 'other' (line 48)
        other_2503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 65), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2504 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 56), '*', z_2502, other_2503)
        
        # Processing the call keyword arguments (line 48)
        kwargs_2505 = {}
        # Getting the type of 'Vector3f' (line 48)
        Vector3f_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 48)
        Vector3f_call_result_2506 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), Vector3f_2492, *[result_mul_2496, result_mul_2500, result_mul_2504], **kwargs_2505)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', Vector3f_call_result_2506)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2507)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_2507


    @norecursion
    def mul(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mul'
        module_type_store = module_type_store.open_function_context('mul', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.mul.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.mul.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.mul.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.mul.__dict__.__setitem__('stypy_function_name', 'Vector3f.mul')
        Vector3f.mul.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.mul.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.mul.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.mul.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.mul.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.mul.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.mul.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.mul', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mul', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mul(...)' code ##################

        
        # Call to Vector3f(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_2509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 51)
        x_2510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 24), self_2509, 'x')
        # Getting the type of 'other' (line 51)
        other_2511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 51)
        x_2512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 33), other_2511, 'x')
        # Applying the binary operator '*' (line 51)
        result_mul_2513 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 24), '*', x_2510, x_2512)
        
        # Getting the type of 'self' (line 51)
        self_2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 51)
        y_2515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 42), self_2514, 'y')
        # Getting the type of 'other' (line 51)
        other_2516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 51)
        y_2517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 51), other_2516, 'y')
        # Applying the binary operator '*' (line 51)
        result_mul_2518 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '*', y_2515, y_2517)
        
        # Getting the type of 'self' (line 51)
        self_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 51)
        z_2520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 60), self_2519, 'z')
        # Getting the type of 'other' (line 51)
        other_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 51)
        z_2522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 69), other_2521, 'z')
        # Applying the binary operator '*' (line 51)
        result_mul_2523 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 60), '*', z_2520, z_2522)
        
        # Processing the call keyword arguments (line 51)
        kwargs_2524 = {}
        # Getting the type of 'Vector3f' (line 51)
        Vector3f_2508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 51)
        Vector3f_call_result_2525 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), Vector3f_2508, *[result_mul_2513, result_mul_2518, result_mul_2523], **kwargs_2524)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', Vector3f_call_result_2525)
        
        # ################# End of 'mul(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mul' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2526)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mul'
        return stypy_return_type_2526


    @norecursion
    def is_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_zero'
        module_type_store = module_type_store.open_function_context('is_zero', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.is_zero.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.is_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.is_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.is_zero.__dict__.__setitem__('stypy_function_name', 'Vector3f.is_zero')
        Vector3f.is_zero.__dict__.__setitem__('stypy_param_names_list', [])
        Vector3f.is_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.is_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.is_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.is_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.is_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.is_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.is_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_zero(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 54)
        self_2527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'self')
        # Obtaining the member 'x' of a type (line 54)
        x_2528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 15), self_2527, 'x')
        float_2529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2530 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '==', x_2528, float_2529)
        
        
        # Getting the type of 'self' (line 54)
        self_2531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'self')
        # Obtaining the member 'y' of a type (line 54)
        y_2532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 33), self_2531, 'y')
        float_2533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2534 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 33), '==', y_2532, float_2533)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2535 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'and', result_eq_2530, result_eq_2534)
        
        # Getting the type of 'self' (line 54)
        self_2536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'self')
        # Obtaining the member 'z' of a type (line 54)
        z_2537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 51), self_2536, 'z')
        float_2538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2539 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 51), '==', z_2537, float_2538)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2540 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'and', result_and_keyword_2535, result_eq_2539)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', result_and_keyword_2540)
        
        # ################# End of 'is_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_2541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2541)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_zero'
        return stypy_return_type_2541


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.dot.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.dot.__dict__.__setitem__('stypy_function_name', 'Vector3f.dot')
        Vector3f.dot.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.dot.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.dot', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        # Getting the type of 'self' (line 57)
        self_2542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'self')
        # Obtaining the member 'x' of a type (line 57)
        x_2543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), self_2542, 'x')
        # Getting the type of 'other' (line 57)
        other_2544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'other')
        # Obtaining the member 'x' of a type (line 57)
        x_2545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), other_2544, 'x')
        # Applying the binary operator '*' (line 57)
        result_mul_2546 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 16), '*', x_2543, x_2545)
        
        # Getting the type of 'self' (line 57)
        self_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'self')
        # Obtaining the member 'y' of a type (line 57)
        y_2548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 37), self_2547, 'y')
        # Getting the type of 'other' (line 57)
        other_2549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 'other')
        # Obtaining the member 'y' of a type (line 57)
        y_2550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 46), other_2549, 'y')
        # Applying the binary operator '*' (line 57)
        result_mul_2551 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 37), '*', y_2548, y_2550)
        
        # Applying the binary operator '+' (line 57)
        result_add_2552 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '+', result_mul_2546, result_mul_2551)
        
        # Getting the type of 'self' (line 57)
        self_2553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 58), 'self')
        # Obtaining the member 'z' of a type (line 57)
        z_2554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 58), self_2553, 'z')
        # Getting the type of 'other' (line 57)
        other_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 67), 'other')
        # Obtaining the member 'z' of a type (line 57)
        z_2556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 67), other_2555, 'z')
        # Applying the binary operator '*' (line 57)
        result_mul_2557 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 58), '*', z_2554, z_2556)
        
        # Applying the binary operator '+' (line 57)
        result_add_2558 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 55), '+', result_add_2552, result_mul_2557)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', result_add_2558)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_2559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_2559


    @norecursion
    def unitize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unitize'
        module_type_store = module_type_store.open_function_context('unitize', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.unitize.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.unitize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.unitize.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.unitize.__dict__.__setitem__('stypy_function_name', 'Vector3f.unitize')
        Vector3f.unitize.__dict__.__setitem__('stypy_param_names_list', [])
        Vector3f.unitize.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.unitize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.unitize.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.unitize.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.unitize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.unitize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.unitize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unitize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unitize(...)' code ##################

        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to sqrt(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'self', False)
        # Obtaining the member 'x' of a type (line 60)
        x_2562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), self_2561, 'x')
        # Getting the type of 'self' (line 60)
        self_2563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'self', False)
        # Obtaining the member 'x' of a type (line 60)
        x_2564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 31), self_2563, 'x')
        # Applying the binary operator '*' (line 60)
        result_mul_2565 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '*', x_2562, x_2564)
        
        # Getting the type of 'self' (line 60)
        self_2566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'self', False)
        # Obtaining the member 'y' of a type (line 60)
        y_2567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), self_2566, 'y')
        # Getting the type of 'self' (line 60)
        self_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'self', False)
        # Obtaining the member 'y' of a type (line 60)
        y_2569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), self_2568, 'y')
        # Applying the binary operator '*' (line 60)
        result_mul_2570 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 40), '*', y_2567, y_2569)
        
        # Applying the binary operator '+' (line 60)
        result_add_2571 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '+', result_mul_2565, result_mul_2570)
        
        # Getting the type of 'self' (line 60)
        self_2572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 58), 'self', False)
        # Obtaining the member 'z' of a type (line 60)
        z_2573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 58), self_2572, 'z')
        # Getting the type of 'self' (line 60)
        self_2574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'self', False)
        # Obtaining the member 'z' of a type (line 60)
        z_2575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 67), self_2574, 'z')
        # Applying the binary operator '*' (line 60)
        result_mul_2576 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 58), '*', z_2573, z_2575)
        
        # Applying the binary operator '+' (line 60)
        result_add_2577 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 56), '+', result_add_2571, result_mul_2576)
        
        # Processing the call keyword arguments (line 60)
        kwargs_2578 = {}
        # Getting the type of 'sqrt' (line 60)
        sqrt_2560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 60)
        sqrt_call_result_2579 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), sqrt_2560, *[result_add_2577], **kwargs_2578)
        
        # Assigning a type to the variable 'length' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'length', sqrt_call_result_2579)
        
        # Assigning a IfExp to a Name (line 61):
        
        # Assigning a IfExp to a Name (line 61):
        
        
        # Getting the type of 'length' (line 61)
        length_2580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'length')
        float_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 52), 'float')
        # Applying the binary operator '!=' (line 61)
        result_ne_2582 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 42), '!=', length_2580, float_2581)
        
        # Testing the type of an if expression (line 61)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 26), result_ne_2582)
        # SSA begins for if expression (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        float_2583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'float')
        # Getting the type of 'length' (line 61)
        length_2584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'length')
        # Applying the binary operator 'div' (line 61)
        result_div_2585 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), 'div', float_2583, length_2584)
        
        # SSA branch for the else part of an if expression (line 61)
        module_type_store.open_ssa_branch('if expression else')
        float_2586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 61), 'float')
        # SSA join for if expression (line 61)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_2587 = union_type.UnionType.add(result_div_2585, float_2586)
        
        # Assigning a type to the variable 'one_over_length' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'one_over_length', if_exp_2587)
        
        # Call to Vector3f(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_2589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 62)
        x_2590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), self_2589, 'x')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2592 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 24), '*', x_2590, one_over_length_2591)
        
        # Getting the type of 'self' (line 62)
        self_2593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'self', False)
        # Obtaining the member 'y' of a type (line 62)
        y_2594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 50), self_2593, 'y')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 59), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2596 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 50), '*', y_2594, one_over_length_2595)
        
        # Getting the type of 'self' (line 62)
        self_2597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 76), 'self', False)
        # Obtaining the member 'z' of a type (line 62)
        z_2598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 76), self_2597, 'z')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 85), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2600 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 76), '*', z_2598, one_over_length_2599)
        
        # Processing the call keyword arguments (line 62)
        kwargs_2601 = {}
        # Getting the type of 'Vector3f' (line 62)
        Vector3f_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 62)
        Vector3f_call_result_2602 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), Vector3f_2588, *[result_mul_2592, result_mul_2596, result_mul_2600], **kwargs_2601)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', Vector3f_call_result_2602)
        
        # ################# End of 'unitize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unitize' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_2603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2603)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unitize'
        return stypy_return_type_2603


    @norecursion
    def cross(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cross'
        module_type_store = module_type_store.open_function_context('cross', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.cross.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.cross.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.cross.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.cross.__dict__.__setitem__('stypy_function_name', 'Vector3f.cross')
        Vector3f.cross.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Vector3f.cross.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.cross.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.cross.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.cross.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.cross.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.cross.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.cross', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cross', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cross(...)' code ##################

        
        # Call to Vector3f(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_2605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'self', False)
        # Obtaining the member 'y' of a type (line 65)
        y_2606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), self_2605, 'y')
        # Getting the type of 'other' (line 65)
        other_2607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'other', False)
        # Obtaining the member 'z' of a type (line 65)
        z_2608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), other_2607, 'z')
        # Applying the binary operator '*' (line 65)
        result_mul_2609 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 25), '*', y_2606, z_2608)
        
        # Getting the type of 'self' (line 65)
        self_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 65)
        z_2611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 46), self_2610, 'z')
        # Getting the type of 'other' (line 65)
        other_2612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 55), 'other', False)
        # Obtaining the member 'y' of a type (line 65)
        y_2613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 55), other_2612, 'y')
        # Applying the binary operator '*' (line 65)
        result_mul_2614 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '*', z_2611, y_2613)
        
        # Applying the binary operator '-' (line 65)
        result_sub_2615 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 24), '-', result_mul_2609, result_mul_2614)
        
        # Getting the type of 'self' (line 66)
        self_2616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'self', False)
        # Obtaining the member 'z' of a type (line 66)
        z_2617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), self_2616, 'z')
        # Getting the type of 'other' (line 66)
        other_2618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'other', False)
        # Obtaining the member 'x' of a type (line 66)
        x_2619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 34), other_2618, 'x')
        # Applying the binary operator '*' (line 66)
        result_mul_2620 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '*', z_2617, x_2619)
        
        # Getting the type of 'self' (line 66)
        self_2621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 46), 'self', False)
        # Obtaining the member 'x' of a type (line 66)
        x_2622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 46), self_2621, 'x')
        # Getting the type of 'other' (line 66)
        other_2623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 55), 'other', False)
        # Obtaining the member 'z' of a type (line 66)
        z_2624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 55), other_2623, 'z')
        # Applying the binary operator '*' (line 66)
        result_mul_2625 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 46), '*', x_2622, z_2624)
        
        # Applying the binary operator '-' (line 66)
        result_sub_2626 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 24), '-', result_mul_2620, result_mul_2625)
        
        # Getting the type of 'self' (line 67)
        self_2627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 67)
        x_2628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), self_2627, 'x')
        # Getting the type of 'other' (line 67)
        other_2629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'other', False)
        # Obtaining the member 'y' of a type (line 67)
        y_2630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 34), other_2629, 'y')
        # Applying the binary operator '*' (line 67)
        result_mul_2631 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '*', x_2628, y_2630)
        
        # Getting the type of 'self' (line 67)
        self_2632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 46), 'self', False)
        # Obtaining the member 'y' of a type (line 67)
        y_2633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 46), self_2632, 'y')
        # Getting the type of 'other' (line 67)
        other_2634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'other', False)
        # Obtaining the member 'x' of a type (line 67)
        x_2635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 55), other_2634, 'x')
        # Applying the binary operator '*' (line 67)
        result_mul_2636 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 46), '*', y_2633, x_2635)
        
        # Applying the binary operator '-' (line 67)
        result_sub_2637 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 24), '-', result_mul_2631, result_mul_2636)
        
        # Processing the call keyword arguments (line 65)
        kwargs_2638 = {}
        # Getting the type of 'Vector3f' (line 65)
        Vector3f_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 65)
        Vector3f_call_result_2639 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), Vector3f_2604, *[result_sub_2615, result_sub_2626, result_sub_2637], **kwargs_2638)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', Vector3f_call_result_2639)
        
        # ################# End of 'cross(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cross' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_2640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cross'
        return stypy_return_type_2640


    @norecursion
    def clamped(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clamped'
        module_type_store = module_type_store.open_function_context('clamped', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vector3f.clamped.__dict__.__setitem__('stypy_localization', localization)
        Vector3f.clamped.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vector3f.clamped.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vector3f.clamped.__dict__.__setitem__('stypy_function_name', 'Vector3f.clamped')
        Vector3f.clamped.__dict__.__setitem__('stypy_param_names_list', ['lo', 'hi'])
        Vector3f.clamped.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vector3f.clamped.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vector3f.clamped.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vector3f.clamped.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vector3f.clamped.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vector3f.clamped.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vector3f.clamped', ['lo', 'hi'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clamped', localization, ['lo', 'hi'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clamped(...)' code ##################

        
        # Call to Vector3f(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to min(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to max(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'self', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 32), self_2644, 'x')
        # Getting the type of 'lo' (line 70)
        lo_2646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'lo', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 40), lo_2646, 'x')
        # Processing the call keyword arguments (line 70)
        kwargs_2648 = {}
        # Getting the type of 'max' (line 70)
        max_2643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'max', False)
        # Calling max(args, kwargs) (line 70)
        max_call_result_2649 = invoke(stypy.reporting.localization.Localization(__file__, 70, 28), max_2643, *[x_2645, x_2647], **kwargs_2648)
        
        # Getting the type of 'hi' (line 70)
        hi_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 47), 'hi', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 47), hi_2650, 'x')
        # Processing the call keyword arguments (line 70)
        kwargs_2652 = {}
        # Getting the type of 'min' (line 70)
        min_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'min', False)
        # Calling min(args, kwargs) (line 70)
        min_call_result_2653 = invoke(stypy.reporting.localization.Localization(__file__, 70, 24), min_2642, *[max_call_result_2649, x_2651], **kwargs_2652)
        
        
        # Call to min(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to max(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), self_2656, 'y')
        # Getting the type of 'lo' (line 71)
        lo_2658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'lo', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 40), lo_2658, 'y')
        # Processing the call keyword arguments (line 71)
        kwargs_2660 = {}
        # Getting the type of 'max' (line 71)
        max_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'max', False)
        # Calling max(args, kwargs) (line 71)
        max_call_result_2661 = invoke(stypy.reporting.localization.Localization(__file__, 71, 28), max_2655, *[y_2657, y_2659], **kwargs_2660)
        
        # Getting the type of 'hi' (line 71)
        hi_2662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'hi', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 47), hi_2662, 'y')
        # Processing the call keyword arguments (line 71)
        kwargs_2664 = {}
        # Getting the type of 'min' (line 71)
        min_2654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'min', False)
        # Calling min(args, kwargs) (line 71)
        min_call_result_2665 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), min_2654, *[max_call_result_2661, y_2663], **kwargs_2664)
        
        
        # Call to min(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to max(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_2668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'self', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), self_2668, 'z')
        # Getting the type of 'lo' (line 72)
        lo_2670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'lo', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 40), lo_2670, 'z')
        # Processing the call keyword arguments (line 72)
        kwargs_2672 = {}
        # Getting the type of 'max' (line 72)
        max_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'max', False)
        # Calling max(args, kwargs) (line 72)
        max_call_result_2673 = invoke(stypy.reporting.localization.Localization(__file__, 72, 28), max_2667, *[z_2669, z_2671], **kwargs_2672)
        
        # Getting the type of 'hi' (line 72)
        hi_2674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'hi', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 47), hi_2674, 'z')
        # Processing the call keyword arguments (line 72)
        kwargs_2676 = {}
        # Getting the type of 'min' (line 72)
        min_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'min', False)
        # Calling min(args, kwargs) (line 72)
        min_call_result_2677 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), min_2666, *[max_call_result_2673, z_2675], **kwargs_2676)
        
        # Processing the call keyword arguments (line 70)
        kwargs_2678 = {}
        # Getting the type of 'Vector3f' (line 70)
        Vector3f_2641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 70)
        Vector3f_call_result_2679 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), Vector3f_2641, *[min_call_result_2653, min_call_result_2665, min_call_result_2677], **kwargs_2678)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', Vector3f_call_result_2679)
        
        # ################# End of 'clamped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clamped' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_2680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clamped'
        return stypy_return_type_2680


# Assigning a type to the variable 'Vector3f' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'Vector3f', Vector3f)

# Assigning a Call to a Name (line 74):

# Assigning a Call to a Name (line 74):

# Call to Vector3f_scalar(...): (line 74)
# Processing the call arguments (line 74)
float_2682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'float')
# Processing the call keyword arguments (line 74)
kwargs_2683 = {}
# Getting the type of 'Vector3f_scalar' (line 74)
Vector3f_scalar_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 74)
Vector3f_scalar_call_result_2684 = invoke(stypy.reporting.localization.Localization(__file__, 74, 7), Vector3f_scalar_2681, *[float_2682], **kwargs_2683)

# Assigning a type to the variable 'ZERO' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'ZERO', Vector3f_scalar_call_result_2684)

# Assigning a Call to a Name (line 75):

# Assigning a Call to a Name (line 75):

# Call to Vector3f_scalar(...): (line 75)
# Processing the call arguments (line 75)
float_2686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'float')
# Processing the call keyword arguments (line 75)
kwargs_2687 = {}
# Getting the type of 'Vector3f_scalar' (line 75)
Vector3f_scalar_2685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 75)
Vector3f_scalar_call_result_2688 = invoke(stypy.reporting.localization.Localization(__file__, 75, 6), Vector3f_scalar_2685, *[float_2686], **kwargs_2687)

# Assigning a type to the variable 'ONE' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'ONE', Vector3f_scalar_call_result_2688)

# Assigning a Call to a Name (line 76):

# Assigning a Call to a Name (line 76):

# Call to Vector3f_scalar(...): (line 76)
# Processing the call arguments (line 76)
float_2690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'float')
# Processing the call keyword arguments (line 76)
kwargs_2691 = {}
# Getting the type of 'Vector3f_scalar' (line 76)
Vector3f_scalar_2689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 76)
Vector3f_scalar_call_result_2692 = invoke(stypy.reporting.localization.Localization(__file__, 76, 6), Vector3f_scalar_2689, *[float_2690], **kwargs_2691)

# Assigning a type to the variable 'MAX' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'MAX', Vector3f_scalar_call_result_2692)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
