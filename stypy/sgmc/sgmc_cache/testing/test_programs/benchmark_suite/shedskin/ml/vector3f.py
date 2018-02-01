
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
    kwargs_2400 = {}
    
    # Call to rstrip(...): (line 10)
    # Processing the call arguments (line 10)
    str_2396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'str', ') ')
    # Processing the call keyword arguments (line 10)
    kwargs_2397 = {}
    
    # Call to lstrip(...): (line 10)
    # Processing the call arguments (line 10)
    str_2392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', ' (')
    # Processing the call keyword arguments (line 10)
    kwargs_2393 = {}
    # Getting the type of 's' (line 10)
    s_2390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 's', False)
    # Obtaining the member 'lstrip' of a type (line 10)
    lstrip_2391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), s_2390, 'lstrip')
    # Calling lstrip(args, kwargs) (line 10)
    lstrip_call_result_2394 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), lstrip_2391, *[str_2392], **kwargs_2393)
    
    # Obtaining the member 'rstrip' of a type (line 10)
    rstrip_2395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), lstrip_call_result_2394, 'rstrip')
    # Calling rstrip(args, kwargs) (line 10)
    rstrip_call_result_2398 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), rstrip_2395, *[str_2396], **kwargs_2397)
    
    # Obtaining the member 'split' of a type (line 10)
    split_2399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), rstrip_call_result_2398, 'split')
    # Calling split(args, kwargs) (line 10)
    split_call_result_2401 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), split_2399, *[], **kwargs_2400)
    
    # Assigning a type to the variable 'split' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'split', split_call_result_2401)
    
    # Call to Vector3f(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'int')
    # Getting the type of 'split' (line 11)
    split_2405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 26), split_2405, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2407 = invoke(stypy.reporting.localization.Localization(__file__, 11, 26), getitem___2406, int_2404)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2408 = {}
    # Getting the type of 'float' (line 11)
    float_2403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 20), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2409 = invoke(stypy.reporting.localization.Localization(__file__, 11, 20), float_2403, *[subscript_call_result_2407], **kwargs_2408)
    
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 49), 'int')
    # Getting the type of 'split' (line 11)
    split_2412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 43), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 43), split_2412, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2414 = invoke(stypy.reporting.localization.Localization(__file__, 11, 43), getitem___2413, int_2411)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2415 = {}
    # Getting the type of 'float' (line 11)
    float_2410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 37), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2416 = invoke(stypy.reporting.localization.Localization(__file__, 11, 37), float_2410, *[subscript_call_result_2414], **kwargs_2415)
    
    
    # Call to float(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Obtaining the type of the subscript
    int_2418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 66), 'int')
    # Getting the type of 'split' (line 11)
    split_2419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 60), 'split', False)
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___2420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 60), split_2419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_2421 = invoke(stypy.reporting.localization.Localization(__file__, 11, 60), getitem___2420, int_2418)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2422 = {}
    # Getting the type of 'float' (line 11)
    float_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 54), 'float', False)
    # Calling float(args, kwargs) (line 11)
    float_call_result_2423 = invoke(stypy.reporting.localization.Localization(__file__, 11, 54), float_2417, *[subscript_call_result_2421], **kwargs_2422)
    
    # Processing the call keyword arguments (line 11)
    kwargs_2424 = {}
    # Getting the type of 'Vector3f' (line 11)
    Vector3f_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 11)
    Vector3f_call_result_2425 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), Vector3f_2402, *[float_call_result_2409, float_call_result_2416, float_call_result_2423], **kwargs_2424)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', Vector3f_call_result_2425)
    
    # ################# End of 'Vector3f_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_str' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_2426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2426)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_str'
    return stypy_return_type_2426

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
    int_2428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 20), seq_2429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2431 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), getitem___2430, int_2428)
    
    
    # Obtaining the type of the subscript
    int_2432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 28), seq_2433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2435 = invoke(stypy.reporting.localization.Localization(__file__, 14, 28), getitem___2434, int_2432)
    
    
    # Obtaining the type of the subscript
    int_2436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
    # Getting the type of 'seq' (line 14)
    seq_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 36), 'seq', False)
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___2438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 36), seq_2437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_2439 = invoke(stypy.reporting.localization.Localization(__file__, 14, 36), getitem___2438, int_2436)
    
    # Processing the call keyword arguments (line 14)
    kwargs_2440 = {}
    # Getting the type of 'Vector3f' (line 14)
    Vector3f_2427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 14)
    Vector3f_call_result_2441 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), Vector3f_2427, *[subscript_call_result_2431, subscript_call_result_2435, subscript_call_result_2439], **kwargs_2440)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', Vector3f_call_result_2441)
    
    # ################# End of 'Vector3f_seq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_seq' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_seq'
    return stypy_return_type_2442

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
    s_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 's', False)
    # Getting the type of 's' (line 17)
    s_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 's', False)
    # Getting the type of 's' (line 17)
    s_2446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 26), 's', False)
    # Processing the call keyword arguments (line 17)
    kwargs_2447 = {}
    # Getting the type of 'Vector3f' (line 17)
    Vector3f_2443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'Vector3f', False)
    # Calling Vector3f(args, kwargs) (line 17)
    Vector3f_call_result_2448 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), Vector3f_2443, *[s_2444, s_2445, s_2446], **kwargs_2447)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', Vector3f_call_result_2448)
    
    # ################# End of 'Vector3f_scalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Vector3f_scalar' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_2449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Vector3f_scalar'
    return stypy_return_type_2449

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
        x_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2452 = {}
        # Getting the type of 'float' (line 22)
        float_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2453 = invoke(stypy.reporting.localization.Localization(__file__, 22, 33), float_2450, *[x_2451], **kwargs_2452)
        
        # Assigning a type to the variable 'tuple_assignment_2387' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2387', float_call_result_2453)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to float(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'y' (line 22)
        y_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'y', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2456 = {}
        # Getting the type of 'float' (line 22)
        float_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 43), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2457 = invoke(stypy.reporting.localization.Localization(__file__, 22, 43), float_2454, *[y_2455], **kwargs_2456)
        
        # Assigning a type to the variable 'tuple_assignment_2388' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2388', float_call_result_2457)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to float(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'z' (line 22)
        z_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 59), 'z', False)
        # Processing the call keyword arguments (line 22)
        kwargs_2460 = {}
        # Getting the type of 'float' (line 22)
        float_2458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'float', False)
        # Calling float(args, kwargs) (line 22)
        float_call_result_2461 = invoke(stypy.reporting.localization.Localization(__file__, 22, 53), float_2458, *[z_2459], **kwargs_2460)
        
        # Assigning a type to the variable 'tuple_assignment_2389' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2389', float_call_result_2461)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2387' (line 22)
        tuple_assignment_2387_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2387')
        # Getting the type of 'self' (line 22)
        self_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'x' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_2463, 'x', tuple_assignment_2387_2462)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2388' (line 22)
        tuple_assignment_2388_2464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2388')
        # Getting the type of 'self' (line 22)
        self_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
        # Setting the type of the member 'y' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_2465, 'y', tuple_assignment_2388_2464)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'tuple_assignment_2389' (line 22)
        tuple_assignment_2389_2466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2389')
        # Getting the type of 'self' (line 22)
        self_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'self')
        # Setting the type of the member 'z' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), self_2467, 'z', tuple_assignment_2389_2466)
        
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
        list_2468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
        # Obtaining the member 'x' of a type (line 25)
        x_2470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_2469, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2468, x_2470)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'self')
        # Obtaining the member 'y' of a type (line 25)
        y_2472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), self_2471, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2468, y_2472)
        # Adding element type (line 25)
        # Getting the type of 'self' (line 25)
        self_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'self')
        # Obtaining the member 'z' of a type (line 25)
        z_2474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 32), self_2473, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), list_2468, z_2474)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', list_2468)
        
        # ################# End of 'as_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'as_list' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_2475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'as_list'
        return stypy_return_type_2475


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
        self_2477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 28)
        x_2478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_2477, 'x')
        # Getting the type of 'self' (line 28)
        self_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 28)
        y_2480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 32), self_2479, 'y')
        # Getting the type of 'self' (line 28)
        self_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'self', False)
        # Obtaining the member 'z' of a type (line 28)
        z_2482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 40), self_2481, 'z')
        # Processing the call keyword arguments (line 28)
        kwargs_2483 = {}
        # Getting the type of 'Vector3f' (line 28)
        Vector3f_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 28)
        Vector3f_call_result_2484 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), Vector3f_2476, *[x_2478, y_2480, z_2482], **kwargs_2483)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', Vector3f_call_result_2484)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_2485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2485)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_2485


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
        key_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'key')
        int_2487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
        # Applying the binary operator '==' (line 31)
        result_eq_2488 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '==', key_2486, int_2487)
        
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), result_eq_2488):
            
            # Getting the type of 'key' (line 33)
            key_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'key')
            int_2493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
            # Applying the binary operator '==' (line 33)
            result_eq_2494 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '==', key_2492, int_2493)
            
            # Testing if the type of an if condition is none (line 33)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2494):
                # Getting the type of 'self' (line 36)
                self_2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2498, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2499)
            else:
                
                # Testing the type of an if condition (line 33)
                if_condition_2495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2494)
                # Assigning a type to the variable 'if_condition_2495' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'if_condition_2495', if_condition_2495)
                # SSA begins for if statement (line 33)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 34)
                self_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
                # Obtaining the member 'y' of a type (line 34)
                y_2497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_2496, 'y')
                # Assigning a type to the variable 'stypy_return_type' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', y_2497)
                # SSA branch for the else part of an if statement (line 33)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 36)
                self_2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2498, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2499)
                # SSA join for if statement (line 33)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_2489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_eq_2488)
            # Assigning a type to the variable 'if_condition_2489' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_2489', if_condition_2489)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'self' (line 32)
            self_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'self')
            # Obtaining the member 'z' of a type (line 32)
            z_2491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 19), self_2490, 'z')
            # Assigning a type to the variable 'stypy_return_type' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', z_2491)
            # SSA branch for the else part of an if statement (line 31)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'key' (line 33)
            key_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'key')
            int_2493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
            # Applying the binary operator '==' (line 33)
            result_eq_2494 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '==', key_2492, int_2493)
            
            # Testing if the type of an if condition is none (line 33)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2494):
                # Getting the type of 'self' (line 36)
                self_2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2498, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2499)
            else:
                
                # Testing the type of an if condition (line 33)
                if_condition_2495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 13), result_eq_2494)
                # Assigning a type to the variable 'if_condition_2495' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'if_condition_2495', if_condition_2495)
                # SSA begins for if statement (line 33)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'self' (line 34)
                self_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
                # Obtaining the member 'y' of a type (line 34)
                y_2497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_2496, 'y')
                # Assigning a type to the variable 'stypy_return_type' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', y_2497)
                # SSA branch for the else part of an if statement (line 33)
                module_type_store.open_ssa_branch('else')
                # Getting the type of 'self' (line 36)
                self_2498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self')
                # Obtaining the member 'x' of a type (line 36)
                x_2499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_2498, 'x')
                # Assigning a type to the variable 'stypy_return_type' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', x_2499)
                # SSA join for if statement (line 33)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_2500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_2500


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
        self_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 39)
        x_2503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), self_2502, 'x')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2504 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 24), 'usub', x_2503)
        
        
        # Getting the type of 'self' (line 39)
        self_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'self', False)
        # Obtaining the member 'y' of a type (line 39)
        y_2506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 34), self_2505, 'y')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2507 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 33), 'usub', y_2506)
        
        
        # Getting the type of 'self' (line 39)
        self_2508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 43), 'self', False)
        # Obtaining the member 'z' of a type (line 39)
        z_2509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 43), self_2508, 'z')
        # Applying the 'usub' unary operator (line 39)
        result___neg___2510 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 42), 'usub', z_2509)
        
        # Processing the call keyword arguments (line 39)
        kwargs_2511 = {}
        # Getting the type of 'Vector3f' (line 39)
        Vector3f_2501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 39)
        Vector3f_call_result_2512 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), Vector3f_2501, *[result___neg___2504, result___neg___2507, result___neg___2510], **kwargs_2511)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', Vector3f_call_result_2512)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_2513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_2513


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
        self_2515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 42)
        x_2516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), self_2515, 'x')
        # Getting the type of 'other' (line 42)
        other_2517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 42)
        x_2518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 33), other_2517, 'x')
        # Applying the binary operator '+' (line 42)
        result_add_2519 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 24), '+', x_2516, x_2518)
        
        # Getting the type of 'self' (line 42)
        self_2520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 42)
        y_2521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), self_2520, 'y')
        # Getting the type of 'other' (line 42)
        other_2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 42)
        y_2523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 51), other_2522, 'y')
        # Applying the binary operator '+' (line 42)
        result_add_2524 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 42), '+', y_2521, y_2523)
        
        # Getting the type of 'self' (line 42)
        self_2525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 42)
        z_2526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 60), self_2525, 'z')
        # Getting the type of 'other' (line 42)
        other_2527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 42)
        z_2528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 69), other_2527, 'z')
        # Applying the binary operator '+' (line 42)
        result_add_2529 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 60), '+', z_2526, z_2528)
        
        # Processing the call keyword arguments (line 42)
        kwargs_2530 = {}
        # Getting the type of 'Vector3f' (line 42)
        Vector3f_2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 42)
        Vector3f_call_result_2531 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), Vector3f_2514, *[result_add_2519, result_add_2524, result_add_2529], **kwargs_2530)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', Vector3f_call_result_2531)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_2532


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
        self_2534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 45)
        x_2535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), self_2534, 'x')
        # Getting the type of 'other' (line 45)
        other_2536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 45)
        x_2537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 33), other_2536, 'x')
        # Applying the binary operator '-' (line 45)
        result_sub_2538 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 24), '-', x_2535, x_2537)
        
        # Getting the type of 'self' (line 45)
        self_2539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 45)
        y_2540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 42), self_2539, 'y')
        # Getting the type of 'other' (line 45)
        other_2541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 45)
        y_2542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 51), other_2541, 'y')
        # Applying the binary operator '-' (line 45)
        result_sub_2543 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 42), '-', y_2540, y_2542)
        
        # Getting the type of 'self' (line 45)
        self_2544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 45)
        z_2545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 60), self_2544, 'z')
        # Getting the type of 'other' (line 45)
        other_2546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 45)
        z_2547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 69), other_2546, 'z')
        # Applying the binary operator '-' (line 45)
        result_sub_2548 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 60), '-', z_2545, z_2547)
        
        # Processing the call keyword arguments (line 45)
        kwargs_2549 = {}
        # Getting the type of 'Vector3f' (line 45)
        Vector3f_2533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 45)
        Vector3f_call_result_2550 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), Vector3f_2533, *[result_sub_2538, result_sub_2543, result_sub_2548], **kwargs_2549)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', Vector3f_call_result_2550)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_2551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2551)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_2551


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
        self_2553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 48)
        x_2554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), self_2553, 'x')
        # Getting the type of 'other' (line 48)
        other_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2556 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 24), '*', x_2554, other_2555)
        
        # Getting the type of 'self' (line 48)
        self_2557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'self', False)
        # Obtaining the member 'y' of a type (line 48)
        y_2558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 40), self_2557, 'y')
        # Getting the type of 'other' (line 48)
        other_2559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2560 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 40), '*', y_2558, other_2559)
        
        # Getting the type of 'self' (line 48)
        self_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 56), 'self', False)
        # Obtaining the member 'z' of a type (line 48)
        z_2562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 56), self_2561, 'z')
        # Getting the type of 'other' (line 48)
        other_2563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 65), 'other', False)
        # Applying the binary operator '*' (line 48)
        result_mul_2564 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 56), '*', z_2562, other_2563)
        
        # Processing the call keyword arguments (line 48)
        kwargs_2565 = {}
        # Getting the type of 'Vector3f' (line 48)
        Vector3f_2552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 48)
        Vector3f_call_result_2566 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), Vector3f_2552, *[result_mul_2556, result_mul_2560, result_mul_2564], **kwargs_2565)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', Vector3f_call_result_2566)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_2567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2567)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_2567


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
        self_2569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 51)
        x_2570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 24), self_2569, 'x')
        # Getting the type of 'other' (line 51)
        other_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'other', False)
        # Obtaining the member 'x' of a type (line 51)
        x_2572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 33), other_2571, 'x')
        # Applying the binary operator '*' (line 51)
        result_mul_2573 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 24), '*', x_2570, x_2572)
        
        # Getting the type of 'self' (line 51)
        self_2574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 42), 'self', False)
        # Obtaining the member 'y' of a type (line 51)
        y_2575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 42), self_2574, 'y')
        # Getting the type of 'other' (line 51)
        other_2576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'other', False)
        # Obtaining the member 'y' of a type (line 51)
        y_2577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 51), other_2576, 'y')
        # Applying the binary operator '*' (line 51)
        result_mul_2578 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 42), '*', y_2575, y_2577)
        
        # Getting the type of 'self' (line 51)
        self_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 60), 'self', False)
        # Obtaining the member 'z' of a type (line 51)
        z_2580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 60), self_2579, 'z')
        # Getting the type of 'other' (line 51)
        other_2581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 69), 'other', False)
        # Obtaining the member 'z' of a type (line 51)
        z_2582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 69), other_2581, 'z')
        # Applying the binary operator '*' (line 51)
        result_mul_2583 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 60), '*', z_2580, z_2582)
        
        # Processing the call keyword arguments (line 51)
        kwargs_2584 = {}
        # Getting the type of 'Vector3f' (line 51)
        Vector3f_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 51)
        Vector3f_call_result_2585 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), Vector3f_2568, *[result_mul_2573, result_mul_2578, result_mul_2583], **kwargs_2584)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', Vector3f_call_result_2585)
        
        # ################# End of 'mul(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mul' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_2586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2586)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mul'
        return stypy_return_type_2586


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
        self_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'self')
        # Obtaining the member 'x' of a type (line 54)
        x_2588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 15), self_2587, 'x')
        float_2589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2590 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '==', x_2588, float_2589)
        
        
        # Getting the type of 'self' (line 54)
        self_2591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'self')
        # Obtaining the member 'y' of a type (line 54)
        y_2592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 33), self_2591, 'y')
        float_2593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 43), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2594 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 33), '==', y_2592, float_2593)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2595 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'and', result_eq_2590, result_eq_2594)
        
        # Getting the type of 'self' (line 54)
        self_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'self')
        # Obtaining the member 'z' of a type (line 54)
        z_2597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 51), self_2596, 'z')
        float_2598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 61), 'float')
        # Applying the binary operator '==' (line 54)
        result_eq_2599 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 51), '==', z_2597, float_2598)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2600 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), 'and', result_and_keyword_2595, result_eq_2599)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', result_and_keyword_2600)
        
        # ################# End of 'is_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_2601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_zero'
        return stypy_return_type_2601


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
        self_2602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'self')
        # Obtaining the member 'x' of a type (line 57)
        x_2603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 16), self_2602, 'x')
        # Getting the type of 'other' (line 57)
        other_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'other')
        # Obtaining the member 'x' of a type (line 57)
        x_2605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 25), other_2604, 'x')
        # Applying the binary operator '*' (line 57)
        result_mul_2606 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 16), '*', x_2603, x_2605)
        
        # Getting the type of 'self' (line 57)
        self_2607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'self')
        # Obtaining the member 'y' of a type (line 57)
        y_2608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 37), self_2607, 'y')
        # Getting the type of 'other' (line 57)
        other_2609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 'other')
        # Obtaining the member 'y' of a type (line 57)
        y_2610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 46), other_2609, 'y')
        # Applying the binary operator '*' (line 57)
        result_mul_2611 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 37), '*', y_2608, y_2610)
        
        # Applying the binary operator '+' (line 57)
        result_add_2612 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '+', result_mul_2606, result_mul_2611)
        
        # Getting the type of 'self' (line 57)
        self_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 58), 'self')
        # Obtaining the member 'z' of a type (line 57)
        z_2614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 58), self_2613, 'z')
        # Getting the type of 'other' (line 57)
        other_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 67), 'other')
        # Obtaining the member 'z' of a type (line 57)
        z_2616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 67), other_2615, 'z')
        # Applying the binary operator '*' (line 57)
        result_mul_2617 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 58), '*', z_2614, z_2616)
        
        # Applying the binary operator '+' (line 57)
        result_add_2618 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 55), '+', result_add_2612, result_mul_2617)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', result_add_2618)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_2619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_2619


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
        self_2621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 22), 'self', False)
        # Obtaining the member 'x' of a type (line 60)
        x_2622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 22), self_2621, 'x')
        # Getting the type of 'self' (line 60)
        self_2623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'self', False)
        # Obtaining the member 'x' of a type (line 60)
        x_2624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 31), self_2623, 'x')
        # Applying the binary operator '*' (line 60)
        result_mul_2625 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '*', x_2622, x_2624)
        
        # Getting the type of 'self' (line 60)
        self_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'self', False)
        # Obtaining the member 'y' of a type (line 60)
        y_2627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), self_2626, 'y')
        # Getting the type of 'self' (line 60)
        self_2628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 49), 'self', False)
        # Obtaining the member 'y' of a type (line 60)
        y_2629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 49), self_2628, 'y')
        # Applying the binary operator '*' (line 60)
        result_mul_2630 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 40), '*', y_2627, y_2629)
        
        # Applying the binary operator '+' (line 60)
        result_add_2631 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '+', result_mul_2625, result_mul_2630)
        
        # Getting the type of 'self' (line 60)
        self_2632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 58), 'self', False)
        # Obtaining the member 'z' of a type (line 60)
        z_2633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 58), self_2632, 'z')
        # Getting the type of 'self' (line 60)
        self_2634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 67), 'self', False)
        # Obtaining the member 'z' of a type (line 60)
        z_2635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 67), self_2634, 'z')
        # Applying the binary operator '*' (line 60)
        result_mul_2636 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 58), '*', z_2633, z_2635)
        
        # Applying the binary operator '+' (line 60)
        result_add_2637 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 56), '+', result_add_2631, result_mul_2636)
        
        # Processing the call keyword arguments (line 60)
        kwargs_2638 = {}
        # Getting the type of 'sqrt' (line 60)
        sqrt_2620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 60)
        sqrt_call_result_2639 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), sqrt_2620, *[result_add_2637], **kwargs_2638)
        
        # Assigning a type to the variable 'length' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'length', sqrt_call_result_2639)
        
        # Assigning a IfExp to a Name (line 61):
        
        # Assigning a IfExp to a Name (line 61):
        
        
        # Getting the type of 'length' (line 61)
        length_2640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'length')
        float_2641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 52), 'float')
        # Applying the binary operator '!=' (line 61)
        result_ne_2642 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 42), '!=', length_2640, float_2641)
        
        # Testing the type of an if expression (line 61)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 26), result_ne_2642)
        # SSA begins for if expression (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        float_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 26), 'float')
        # Getting the type of 'length' (line 61)
        length_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'length')
        # Applying the binary operator 'div' (line 61)
        result_div_2645 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), 'div', float_2643, length_2644)
        
        # SSA branch for the else part of an if expression (line 61)
        module_type_store.open_ssa_branch('if expression else')
        float_2646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 61), 'float')
        # SSA join for if expression (line 61)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_2647 = union_type.UnionType.add(result_div_2645, float_2646)
        
        # Assigning a type to the variable 'one_over_length' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'one_over_length', if_exp_2647)
        
        # Call to Vector3f(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_2649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'self', False)
        # Obtaining the member 'x' of a type (line 62)
        x_2650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), self_2649, 'x')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2652 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 24), '*', x_2650, one_over_length_2651)
        
        # Getting the type of 'self' (line 62)
        self_2653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 50), 'self', False)
        # Obtaining the member 'y' of a type (line 62)
        y_2654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 50), self_2653, 'y')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 59), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2656 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 50), '*', y_2654, one_over_length_2655)
        
        # Getting the type of 'self' (line 62)
        self_2657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 76), 'self', False)
        # Obtaining the member 'z' of a type (line 62)
        z_2658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 76), self_2657, 'z')
        # Getting the type of 'one_over_length' (line 62)
        one_over_length_2659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 85), 'one_over_length', False)
        # Applying the binary operator '*' (line 62)
        result_mul_2660 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 76), '*', z_2658, one_over_length_2659)
        
        # Processing the call keyword arguments (line 62)
        kwargs_2661 = {}
        # Getting the type of 'Vector3f' (line 62)
        Vector3f_2648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 62)
        Vector3f_call_result_2662 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), Vector3f_2648, *[result_mul_2652, result_mul_2656, result_mul_2660], **kwargs_2661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', Vector3f_call_result_2662)
        
        # ################# End of 'unitize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unitize' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_2663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unitize'
        return stypy_return_type_2663


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
        self_2665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'self', False)
        # Obtaining the member 'y' of a type (line 65)
        y_2666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 25), self_2665, 'y')
        # Getting the type of 'other' (line 65)
        other_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'other', False)
        # Obtaining the member 'z' of a type (line 65)
        z_2668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 34), other_2667, 'z')
        # Applying the binary operator '*' (line 65)
        result_mul_2669 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 25), '*', y_2666, z_2668)
        
        # Getting the type of 'self' (line 65)
        self_2670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 65)
        z_2671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 46), self_2670, 'z')
        # Getting the type of 'other' (line 65)
        other_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 55), 'other', False)
        # Obtaining the member 'y' of a type (line 65)
        y_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 55), other_2672, 'y')
        # Applying the binary operator '*' (line 65)
        result_mul_2674 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 46), '*', z_2671, y_2673)
        
        # Applying the binary operator '-' (line 65)
        result_sub_2675 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 24), '-', result_mul_2669, result_mul_2674)
        
        # Getting the type of 'self' (line 66)
        self_2676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'self', False)
        # Obtaining the member 'z' of a type (line 66)
        z_2677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), self_2676, 'z')
        # Getting the type of 'other' (line 66)
        other_2678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'other', False)
        # Obtaining the member 'x' of a type (line 66)
        x_2679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 34), other_2678, 'x')
        # Applying the binary operator '*' (line 66)
        result_mul_2680 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '*', z_2677, x_2679)
        
        # Getting the type of 'self' (line 66)
        self_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 46), 'self', False)
        # Obtaining the member 'x' of a type (line 66)
        x_2682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 46), self_2681, 'x')
        # Getting the type of 'other' (line 66)
        other_2683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 55), 'other', False)
        # Obtaining the member 'z' of a type (line 66)
        z_2684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 55), other_2683, 'z')
        # Applying the binary operator '*' (line 66)
        result_mul_2685 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 46), '*', x_2682, z_2684)
        
        # Applying the binary operator '-' (line 66)
        result_sub_2686 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 24), '-', result_mul_2680, result_mul_2685)
        
        # Getting the type of 'self' (line 67)
        self_2687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 67)
        x_2688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), self_2687, 'x')
        # Getting the type of 'other' (line 67)
        other_2689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'other', False)
        # Obtaining the member 'y' of a type (line 67)
        y_2690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 34), other_2689, 'y')
        # Applying the binary operator '*' (line 67)
        result_mul_2691 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 25), '*', x_2688, y_2690)
        
        # Getting the type of 'self' (line 67)
        self_2692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 46), 'self', False)
        # Obtaining the member 'y' of a type (line 67)
        y_2693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 46), self_2692, 'y')
        # Getting the type of 'other' (line 67)
        other_2694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'other', False)
        # Obtaining the member 'x' of a type (line 67)
        x_2695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 55), other_2694, 'x')
        # Applying the binary operator '*' (line 67)
        result_mul_2696 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 46), '*', y_2693, x_2695)
        
        # Applying the binary operator '-' (line 67)
        result_sub_2697 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 24), '-', result_mul_2691, result_mul_2696)
        
        # Processing the call keyword arguments (line 65)
        kwargs_2698 = {}
        # Getting the type of 'Vector3f' (line 65)
        Vector3f_2664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 65)
        Vector3f_call_result_2699 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), Vector3f_2664, *[result_sub_2675, result_sub_2686, result_sub_2697], **kwargs_2698)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', Vector3f_call_result_2699)
        
        # ################# End of 'cross(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cross' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_2700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2700)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cross'
        return stypy_return_type_2700


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
        self_2704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 32), 'self', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 32), self_2704, 'x')
        # Getting the type of 'lo' (line 70)
        lo_2706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 40), 'lo', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 40), lo_2706, 'x')
        # Processing the call keyword arguments (line 70)
        kwargs_2708 = {}
        # Getting the type of 'max' (line 70)
        max_2703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'max', False)
        # Calling max(args, kwargs) (line 70)
        max_call_result_2709 = invoke(stypy.reporting.localization.Localization(__file__, 70, 28), max_2703, *[x_2705, x_2707], **kwargs_2708)
        
        # Getting the type of 'hi' (line 70)
        hi_2710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 47), 'hi', False)
        # Obtaining the member 'x' of a type (line 70)
        x_2711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 47), hi_2710, 'x')
        # Processing the call keyword arguments (line 70)
        kwargs_2712 = {}
        # Getting the type of 'min' (line 70)
        min_2702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'min', False)
        # Calling min(args, kwargs) (line 70)
        min_call_result_2713 = invoke(stypy.reporting.localization.Localization(__file__, 70, 24), min_2702, *[max_call_result_2709, x_2711], **kwargs_2712)
        
        
        # Call to min(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to max(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'self' (line 71)
        self_2716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), self_2716, 'y')
        # Getting the type of 'lo' (line 71)
        lo_2718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'lo', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 40), lo_2718, 'y')
        # Processing the call keyword arguments (line 71)
        kwargs_2720 = {}
        # Getting the type of 'max' (line 71)
        max_2715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 28), 'max', False)
        # Calling max(args, kwargs) (line 71)
        max_call_result_2721 = invoke(stypy.reporting.localization.Localization(__file__, 71, 28), max_2715, *[y_2717, y_2719], **kwargs_2720)
        
        # Getting the type of 'hi' (line 71)
        hi_2722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'hi', False)
        # Obtaining the member 'y' of a type (line 71)
        y_2723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 47), hi_2722, 'y')
        # Processing the call keyword arguments (line 71)
        kwargs_2724 = {}
        # Getting the type of 'min' (line 71)
        min_2714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'min', False)
        # Calling min(args, kwargs) (line 71)
        min_call_result_2725 = invoke(stypy.reporting.localization.Localization(__file__, 71, 24), min_2714, *[max_call_result_2721, y_2723], **kwargs_2724)
        
        
        # Call to min(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Call to max(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_2728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'self', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), self_2728, 'z')
        # Getting the type of 'lo' (line 72)
        lo_2730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'lo', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 40), lo_2730, 'z')
        # Processing the call keyword arguments (line 72)
        kwargs_2732 = {}
        # Getting the type of 'max' (line 72)
        max_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'max', False)
        # Calling max(args, kwargs) (line 72)
        max_call_result_2733 = invoke(stypy.reporting.localization.Localization(__file__, 72, 28), max_2727, *[z_2729, z_2731], **kwargs_2732)
        
        # Getting the type of 'hi' (line 72)
        hi_2734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'hi', False)
        # Obtaining the member 'z' of a type (line 72)
        z_2735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 47), hi_2734, 'z')
        # Processing the call keyword arguments (line 72)
        kwargs_2736 = {}
        # Getting the type of 'min' (line 72)
        min_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'min', False)
        # Calling min(args, kwargs) (line 72)
        min_call_result_2737 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), min_2726, *[max_call_result_2733, z_2735], **kwargs_2736)
        
        # Processing the call keyword arguments (line 70)
        kwargs_2738 = {}
        # Getting the type of 'Vector3f' (line 70)
        Vector3f_2701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'Vector3f', False)
        # Calling Vector3f(args, kwargs) (line 70)
        Vector3f_call_result_2739 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), Vector3f_2701, *[min_call_result_2713, min_call_result_2725, min_call_result_2737], **kwargs_2738)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', Vector3f_call_result_2739)
        
        # ################# End of 'clamped(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clamped' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_2740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2740)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clamped'
        return stypy_return_type_2740


# Assigning a type to the variable 'Vector3f' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'Vector3f', Vector3f)

# Assigning a Call to a Name (line 74):

# Assigning a Call to a Name (line 74):

# Call to Vector3f_scalar(...): (line 74)
# Processing the call arguments (line 74)
float_2742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'float')
# Processing the call keyword arguments (line 74)
kwargs_2743 = {}
# Getting the type of 'Vector3f_scalar' (line 74)
Vector3f_scalar_2741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 74)
Vector3f_scalar_call_result_2744 = invoke(stypy.reporting.localization.Localization(__file__, 74, 7), Vector3f_scalar_2741, *[float_2742], **kwargs_2743)

# Assigning a type to the variable 'ZERO' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'ZERO', Vector3f_scalar_call_result_2744)

# Assigning a Call to a Name (line 75):

# Assigning a Call to a Name (line 75):

# Call to Vector3f_scalar(...): (line 75)
# Processing the call arguments (line 75)
float_2746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'float')
# Processing the call keyword arguments (line 75)
kwargs_2747 = {}
# Getting the type of 'Vector3f_scalar' (line 75)
Vector3f_scalar_2745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 6), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 75)
Vector3f_scalar_call_result_2748 = invoke(stypy.reporting.localization.Localization(__file__, 75, 6), Vector3f_scalar_2745, *[float_2746], **kwargs_2747)

# Assigning a type to the variable 'ONE' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'ONE', Vector3f_scalar_call_result_2748)

# Assigning a Call to a Name (line 76):

# Assigning a Call to a Name (line 76):

# Call to Vector3f_scalar(...): (line 76)
# Processing the call arguments (line 76)
float_2750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'float')
# Processing the call keyword arguments (line 76)
kwargs_2751 = {}
# Getting the type of 'Vector3f_scalar' (line 76)
Vector3f_scalar_2749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 6), 'Vector3f_scalar', False)
# Calling Vector3f_scalar(args, kwargs) (line 76)
Vector3f_scalar_call_result_2752 = invoke(stypy.reporting.localization.Localization(__file__, 76, 6), Vector3f_scalar_2749, *[float_2750], **kwargs_2751)

# Assigning a type to the variable 'MAX' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'MAX', Vector3f_scalar_call_result_2752)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
