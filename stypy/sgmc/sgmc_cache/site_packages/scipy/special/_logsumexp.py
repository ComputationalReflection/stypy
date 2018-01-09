
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy._lib._util import _asarray_validated
5: 
6: __all__ = ["logsumexp"]
7: 
8: def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
9:     '''Compute the log of the sum of exponentials of input elements.
10: 
11:     Parameters
12:     ----------
13:     a : array_like
14:         Input array.
15:     axis : None or int or tuple of ints, optional
16:         Axis or axes over which the sum is taken. By default `axis` is None,
17:         and all elements are summed.
18: 
19:         .. versionadded:: 0.11.0
20:     keepdims : bool, optional
21:         If this is set to True, the axes which are reduced are left in the
22:         result as dimensions with size one. With this option, the result
23:         will broadcast correctly against the original array.
24: 
25:         .. versionadded:: 0.15.0
26:     b : array-like, optional
27:         Scaling factor for exp(`a`) must be of the same shape as `a` or
28:         broadcastable to `a`. These values may be negative in order to
29:         implement subtraction.
30: 
31:         .. versionadded:: 0.12.0
32:     return_sign : bool, optional
33:         If this is set to True, the result will be a pair containing sign
34:         information; if False, results that are negative will be returned
35:         as NaN. Default is False (no sign information).
36: 
37:         .. versionadded:: 0.16.0
38:     Returns
39:     -------
40:     res : ndarray
41:         The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
42:         more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
43:         is returned.
44:     sgn : ndarray
45:         If return_sign is True, this will be an array of floating-point
46:         numbers matching res and +1, 0, or -1 depending on the sign
47:         of the result. If False, only one result is returned.
48: 
49:     See Also
50:     --------
51:     numpy.logaddexp, numpy.logaddexp2
52: 
53:     Notes
54:     -----
55:     Numpy has a logaddexp function which is very similar to `logsumexp`, but
56:     only handles two arguments. `logaddexp.reduce` is similar to this
57:     function, but may be less stable.
58: 
59:     Examples
60:     --------
61:     >>> from scipy.special import logsumexp
62:     >>> a = np.arange(10)
63:     >>> np.log(np.sum(np.exp(a)))
64:     9.4586297444267107
65:     >>> logsumexp(a)
66:     9.4586297444267107
67: 
68:     With weights
69: 
70:     >>> a = np.arange(10)
71:     >>> b = np.arange(10, 0, -1)
72:     >>> logsumexp(a, b=b)
73:     9.9170178533034665
74:     >>> np.log(np.sum(b*np.exp(a)))
75:     9.9170178533034647
76: 
77:     Returning a sign flag
78: 
79:     >>> logsumexp([1,2],b=[1,-1],return_sign=True)
80:     (1.5413248546129181, -1.0)
81: 
82:     Notice that `logsumexp` does not directly support masked arrays. To use it
83:     on a masked array, convert the mask into zero weights:
84: 
85:     >>> a = np.ma.array([np.log(2), 2, np.log(3)],
86:     ...                  mask=[False, True, False])
87:     >>> b = (~a.mask).astype(int)
88:     >>> logsumexp(a.data, b=b), np.log(5)
89:     1.6094379124341005, 1.6094379124341005
90: 
91:     '''
92:     a = _asarray_validated(a, check_finite=False)
93:     if b is not None:
94:         a, b = np.broadcast_arrays(a,b)
95:         if np.any(b == 0):
96:             a = a + 0.  # promote to at least float
97:             a[b == 0] = -np.inf
98: 
99:     a_max = np.amax(a, axis=axis, keepdims=True)
100: 
101:     if a_max.ndim > 0:
102:         a_max[~np.isfinite(a_max)] = 0
103:     elif not np.isfinite(a_max):
104:         a_max = 0
105: 
106:     if b is not None:
107:         b = np.asarray(b)
108:         tmp = b * np.exp(a - a_max)
109:     else:
110:         tmp = np.exp(a - a_max)
111: 
112:     # suppress warnings about log of zero
113:     with np.errstate(divide='ignore'):
114:         s = np.sum(tmp, axis=axis, keepdims=keepdims)
115:         if return_sign:
116:             sgn = np.sign(s)
117:             s *= sgn  # /= makes more sense but we need zero -> zero
118:         out = np.log(s)
119: 
120:     if not keepdims:
121:         a_max = np.squeeze(a_max, axis=axis)
122:     out += a_max
123: 
124:     if return_sign:
125:         return out, sgn
126:     else:
127:         return out
128: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509457 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_509457) is not StypyTypeError):

    if (import_509457 != 'pyd_module'):
        __import__(import_509457)
        sys_modules_509458 = sys.modules[import_509457]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_509458.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_509457)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_509459 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._util')

if (type(import_509459) is not StypyTypeError):

    if (import_509459 != 'pyd_module'):
        __import__(import_509459)
        sys_modules_509460 = sys.modules[import_509459]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._util', sys_modules_509460.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_509460, sys_modules_509460.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._util', import_509459)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['logsumexp']
module_type_store.set_exportable_members(['logsumexp'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_509461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_509462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'logsumexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_509461, str_509462)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_509461)

@norecursion
def logsumexp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 8)
    None_509463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'None')
    # Getting the type of 'None' (line 8)
    None_509464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 30), 'None')
    # Getting the type of 'False' (line 8)
    False_509465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 45), 'False')
    # Getting the type of 'False' (line 8)
    False_509466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 64), 'False')
    defaults = [None_509463, None_509464, False_509465, False_509466]
    # Create a new context for function 'logsumexp'
    module_type_store = module_type_store.open_function_context('logsumexp', 8, 0, False)
    
    # Passed parameters checking function
    logsumexp.stypy_localization = localization
    logsumexp.stypy_type_of_self = None
    logsumexp.stypy_type_store = module_type_store
    logsumexp.stypy_function_name = 'logsumexp'
    logsumexp.stypy_param_names_list = ['a', 'axis', 'b', 'keepdims', 'return_sign']
    logsumexp.stypy_varargs_param_name = None
    logsumexp.stypy_kwargs_param_name = None
    logsumexp.stypy_call_defaults = defaults
    logsumexp.stypy_call_varargs = varargs
    logsumexp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'logsumexp', ['a', 'axis', 'b', 'keepdims', 'return_sign'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'logsumexp', localization, ['a', 'axis', 'b', 'keepdims', 'return_sign'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'logsumexp(...)' code ##################

    str_509467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', 'Compute the log of the sum of exponentials of input elements.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : None or int or tuple of ints, optional\n        Axis or axes over which the sum is taken. By default `axis` is None,\n        and all elements are summed.\n\n        .. versionadded:: 0.11.0\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the original array.\n\n        .. versionadded:: 0.15.0\n    b : array-like, optional\n        Scaling factor for exp(`a`) must be of the same shape as `a` or\n        broadcastable to `a`. These values may be negative in order to\n        implement subtraction.\n\n        .. versionadded:: 0.12.0\n    return_sign : bool, optional\n        If this is set to True, the result will be a pair containing sign\n        information; if False, results that are negative will be returned\n        as NaN. Default is False (no sign information).\n\n        .. versionadded:: 0.16.0\n    Returns\n    -------\n    res : ndarray\n        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically\n        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``\n        is returned.\n    sgn : ndarray\n        If return_sign is True, this will be an array of floating-point\n        numbers matching res and +1, 0, or -1 depending on the sign\n        of the result. If False, only one result is returned.\n\n    See Also\n    --------\n    numpy.logaddexp, numpy.logaddexp2\n\n    Notes\n    -----\n    Numpy has a logaddexp function which is very similar to `logsumexp`, but\n    only handles two arguments. `logaddexp.reduce` is similar to this\n    function, but may be less stable.\n\n    Examples\n    --------\n    >>> from scipy.special import logsumexp\n    >>> a = np.arange(10)\n    >>> np.log(np.sum(np.exp(a)))\n    9.4586297444267107\n    >>> logsumexp(a)\n    9.4586297444267107\n\n    With weights\n\n    >>> a = np.arange(10)\n    >>> b = np.arange(10, 0, -1)\n    >>> logsumexp(a, b=b)\n    9.9170178533034665\n    >>> np.log(np.sum(b*np.exp(a)))\n    9.9170178533034647\n\n    Returning a sign flag\n\n    >>> logsumexp([1,2],b=[1,-1],return_sign=True)\n    (1.5413248546129181, -1.0)\n\n    Notice that `logsumexp` does not directly support masked arrays. To use it\n    on a masked array, convert the mask into zero weights:\n\n    >>> a = np.ma.array([np.log(2), 2, np.log(3)],\n    ...                  mask=[False, True, False])\n    >>> b = (~a.mask).astype(int)\n    >>> logsumexp(a.data, b=b), np.log(5)\n    1.6094379124341005, 1.6094379124341005\n\n    ')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to _asarray_validated(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'a' (line 92)
    a_509469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'a', False)
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'False' (line 92)
    False_509470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'False', False)
    keyword_509471 = False_509470
    kwargs_509472 = {'check_finite': keyword_509471}
    # Getting the type of '_asarray_validated' (line 92)
    _asarray_validated_509468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 92)
    _asarray_validated_call_result_509473 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), _asarray_validated_509468, *[a_509469], **kwargs_509472)
    
    # Assigning a type to the variable 'a' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'a', _asarray_validated_call_result_509473)
    
    # Type idiom detected: calculating its left and rigth part (line 93)
    # Getting the type of 'b' (line 93)
    b_509474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'b')
    # Getting the type of 'None' (line 93)
    None_509475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'None')
    
    (may_be_509476, more_types_in_union_509477) = may_not_be_none(b_509474, None_509475)

    if may_be_509476:

        if more_types_in_union_509477:
            # Runtime conditional SSA (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Tuple (line 94):
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_509478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'a' (line 94)
        a_509481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'a', False)
        # Getting the type of 'b' (line 94)
        b_509482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'b', False)
        # Processing the call keyword arguments (line 94)
        kwargs_509483 = {}
        # Getting the type of 'np' (line 94)
        np_509479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 94)
        broadcast_arrays_509480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), np_509479, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 94)
        broadcast_arrays_call_result_509484 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), broadcast_arrays_509480, *[a_509481, b_509482], **kwargs_509483)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___509485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), broadcast_arrays_call_result_509484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_509486 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___509485, int_509478)
        
        # Assigning a type to the variable 'tuple_var_assignment_509455' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_509455', subscript_call_result_509486)
        
        # Assigning a Subscript to a Name (line 94):
        
        # Obtaining the type of the subscript
        int_509487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'a' (line 94)
        a_509490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'a', False)
        # Getting the type of 'b' (line 94)
        b_509491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'b', False)
        # Processing the call keyword arguments (line 94)
        kwargs_509492 = {}
        # Getting the type of 'np' (line 94)
        np_509488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 94)
        broadcast_arrays_509489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), np_509488, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 94)
        broadcast_arrays_call_result_509493 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), broadcast_arrays_509489, *[a_509490, b_509491], **kwargs_509492)
        
        # Obtaining the member '__getitem__' of a type (line 94)
        getitem___509494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), broadcast_arrays_call_result_509493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 94)
        subscript_call_result_509495 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), getitem___509494, int_509487)
        
        # Assigning a type to the variable 'tuple_var_assignment_509456' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_509456', subscript_call_result_509495)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_509455' (line 94)
        tuple_var_assignment_509455_509496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_509455')
        # Assigning a type to the variable 'a' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'a', tuple_var_assignment_509455_509496)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_var_assignment_509456' (line 94)
        tuple_var_assignment_509456_509497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_var_assignment_509456')
        # Assigning a type to the variable 'b' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'b', tuple_var_assignment_509456_509497)
        
        
        # Call to any(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Getting the type of 'b' (line 95)
        b_509500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'b', False)
        int_509501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'int')
        # Applying the binary operator '==' (line 95)
        result_eq_509502 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 18), '==', b_509500, int_509501)
        
        # Processing the call keyword arguments (line 95)
        kwargs_509503 = {}
        # Getting the type of 'np' (line 95)
        np_509498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 95)
        any_509499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), np_509498, 'any')
        # Calling any(args, kwargs) (line 95)
        any_call_result_509504 = invoke(stypy.reporting.localization.Localization(__file__, 95, 11), any_509499, *[result_eq_509502], **kwargs_509503)
        
        # Testing the type of an if condition (line 95)
        if_condition_509505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), any_call_result_509504)
        # Assigning a type to the variable 'if_condition_509505' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_509505', if_condition_509505)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 96):
        
        # Assigning a BinOp to a Name (line 96):
        # Getting the type of 'a' (line 96)
        a_509506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'a')
        float_509507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'float')
        # Applying the binary operator '+' (line 96)
        result_add_509508 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 16), '+', a_509506, float_509507)
        
        # Assigning a type to the variable 'a' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'a', result_add_509508)
        
        # Assigning a UnaryOp to a Subscript (line 97):
        
        # Assigning a UnaryOp to a Subscript (line 97):
        
        # Getting the type of 'np' (line 97)
        np_509509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'np')
        # Obtaining the member 'inf' of a type (line 97)
        inf_509510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 25), np_509509, 'inf')
        # Applying the 'usub' unary operator (line 97)
        result___neg___509511 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 24), 'usub', inf_509510)
        
        # Getting the type of 'a' (line 97)
        a_509512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'a')
        
        # Getting the type of 'b' (line 97)
        b_509513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'b')
        int_509514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 19), 'int')
        # Applying the binary operator '==' (line 97)
        result_eq_509515 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 14), '==', b_509513, int_509514)
        
        # Storing an element on a container (line 97)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 12), a_509512, (result_eq_509515, result___neg___509511))
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_509477:
            # SSA join for if statement (line 93)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to amax(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'a' (line 99)
    a_509518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'a', False)
    # Processing the call keyword arguments (line 99)
    # Getting the type of 'axis' (line 99)
    axis_509519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'axis', False)
    keyword_509520 = axis_509519
    # Getting the type of 'True' (line 99)
    True_509521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 43), 'True', False)
    keyword_509522 = True_509521
    kwargs_509523 = {'keepdims': keyword_509522, 'axis': keyword_509520}
    # Getting the type of 'np' (line 99)
    np_509516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'np', False)
    # Obtaining the member 'amax' of a type (line 99)
    amax_509517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), np_509516, 'amax')
    # Calling amax(args, kwargs) (line 99)
    amax_call_result_509524 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), amax_509517, *[a_509518], **kwargs_509523)
    
    # Assigning a type to the variable 'a_max' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'a_max', amax_call_result_509524)
    
    
    # Getting the type of 'a_max' (line 101)
    a_max_509525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'a_max')
    # Obtaining the member 'ndim' of a type (line 101)
    ndim_509526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 7), a_max_509525, 'ndim')
    int_509527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'int')
    # Applying the binary operator '>' (line 101)
    result_gt_509528 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '>', ndim_509526, int_509527)
    
    # Testing the type of an if condition (line 101)
    if_condition_509529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_gt_509528)
    # Assigning a type to the variable 'if_condition_509529' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_509529', if_condition_509529)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 102):
    
    # Assigning a Num to a Subscript (line 102):
    int_509530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
    # Getting the type of 'a_max' (line 102)
    a_max_509531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'a_max')
    
    
    # Call to isfinite(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'a_max' (line 102)
    a_max_509534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'a_max', False)
    # Processing the call keyword arguments (line 102)
    kwargs_509535 = {}
    # Getting the type of 'np' (line 102)
    np_509532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 102)
    isfinite_509533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), np_509532, 'isfinite')
    # Calling isfinite(args, kwargs) (line 102)
    isfinite_call_result_509536 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), isfinite_509533, *[a_max_509534], **kwargs_509535)
    
    # Applying the '~' unary operator (line 102)
    result_inv_509537 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 14), '~', isfinite_call_result_509536)
    
    # Storing an element on a container (line 102)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 8), a_max_509531, (result_inv_509537, int_509530))
    # SSA branch for the else part of an if statement (line 101)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to isfinite(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'a_max' (line 103)
    a_max_509540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'a_max', False)
    # Processing the call keyword arguments (line 103)
    kwargs_509541 = {}
    # Getting the type of 'np' (line 103)
    np_509538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 103)
    isfinite_509539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), np_509538, 'isfinite')
    # Calling isfinite(args, kwargs) (line 103)
    isfinite_call_result_509542 = invoke(stypy.reporting.localization.Localization(__file__, 103, 13), isfinite_509539, *[a_max_509540], **kwargs_509541)
    
    # Applying the 'not' unary operator (line 103)
    result_not__509543 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 9), 'not', isfinite_call_result_509542)
    
    # Testing the type of an if condition (line 103)
    if_condition_509544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 9), result_not__509543)
    # Assigning a type to the variable 'if_condition_509544' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 9), 'if_condition_509544', if_condition_509544)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 104):
    
    # Assigning a Num to a Name (line 104):
    int_509545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'int')
    # Assigning a type to the variable 'a_max' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'a_max', int_509545)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 106)
    # Getting the type of 'b' (line 106)
    b_509546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'b')
    # Getting the type of 'None' (line 106)
    None_509547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'None')
    
    (may_be_509548, more_types_in_union_509549) = may_not_be_none(b_509546, None_509547)

    if may_be_509548:

        if more_types_in_union_509549:
            # Runtime conditional SSA (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to asarray(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'b' (line 107)
        b_509552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'b', False)
        # Processing the call keyword arguments (line 107)
        kwargs_509553 = {}
        # Getting the type of 'np' (line 107)
        np_509550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 107)
        asarray_509551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), np_509550, 'asarray')
        # Calling asarray(args, kwargs) (line 107)
        asarray_call_result_509554 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), asarray_509551, *[b_509552], **kwargs_509553)
        
        # Assigning a type to the variable 'b' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'b', asarray_call_result_509554)
        
        # Assigning a BinOp to a Name (line 108):
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'b' (line 108)
        b_509555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 14), 'b')
        
        # Call to exp(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'a' (line 108)
        a_509558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'a', False)
        # Getting the type of 'a_max' (line 108)
        a_max_509559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 29), 'a_max', False)
        # Applying the binary operator '-' (line 108)
        result_sub_509560 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 25), '-', a_509558, a_max_509559)
        
        # Processing the call keyword arguments (line 108)
        kwargs_509561 = {}
        # Getting the type of 'np' (line 108)
        np_509556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'np', False)
        # Obtaining the member 'exp' of a type (line 108)
        exp_509557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 18), np_509556, 'exp')
        # Calling exp(args, kwargs) (line 108)
        exp_call_result_509562 = invoke(stypy.reporting.localization.Localization(__file__, 108, 18), exp_509557, *[result_sub_509560], **kwargs_509561)
        
        # Applying the binary operator '*' (line 108)
        result_mul_509563 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 14), '*', b_509555, exp_call_result_509562)
        
        # Assigning a type to the variable 'tmp' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tmp', result_mul_509563)

        if more_types_in_union_509549:
            # Runtime conditional SSA for else branch (line 106)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_509548) or more_types_in_union_509549):
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to exp(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'a' (line 110)
        a_509566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'a', False)
        # Getting the type of 'a_max' (line 110)
        a_max_509567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'a_max', False)
        # Applying the binary operator '-' (line 110)
        result_sub_509568 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 21), '-', a_509566, a_max_509567)
        
        # Processing the call keyword arguments (line 110)
        kwargs_509569 = {}
        # Getting the type of 'np' (line 110)
        np_509564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'np', False)
        # Obtaining the member 'exp' of a type (line 110)
        exp_509565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 14), np_509564, 'exp')
        # Calling exp(args, kwargs) (line 110)
        exp_call_result_509570 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), exp_509565, *[result_sub_509568], **kwargs_509569)
        
        # Assigning a type to the variable 'tmp' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'tmp', exp_call_result_509570)

        if (may_be_509548 and more_types_in_union_509549):
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to errstate(...): (line 113)
    # Processing the call keyword arguments (line 113)
    str_509573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'str', 'ignore')
    keyword_509574 = str_509573
    kwargs_509575 = {'divide': keyword_509574}
    # Getting the type of 'np' (line 113)
    np_509571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 113)
    errstate_509572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), np_509571, 'errstate')
    # Calling errstate(args, kwargs) (line 113)
    errstate_call_result_509576 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), errstate_509572, *[], **kwargs_509575)
    
    with_509577 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 113, 9), errstate_call_result_509576, 'with parameter', '__enter__', '__exit__')

    if with_509577:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 113)
        enter___509578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), errstate_call_result_509576, '__enter__')
        with_enter_509579 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), enter___509578)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to sum(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'tmp' (line 114)
        tmp_509582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'tmp', False)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'axis' (line 114)
        axis_509583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'axis', False)
        keyword_509584 = axis_509583
        # Getting the type of 'keepdims' (line 114)
        keepdims_509585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 44), 'keepdims', False)
        keyword_509586 = keepdims_509585
        kwargs_509587 = {'keepdims': keyword_509586, 'axis': keyword_509584}
        # Getting the type of 'np' (line 114)
        np_509580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'np', False)
        # Obtaining the member 'sum' of a type (line 114)
        sum_509581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), np_509580, 'sum')
        # Calling sum(args, kwargs) (line 114)
        sum_call_result_509588 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), sum_509581, *[tmp_509582], **kwargs_509587)
        
        # Assigning a type to the variable 's' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 's', sum_call_result_509588)
        
        # Getting the type of 'return_sign' (line 115)
        return_sign_509589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'return_sign')
        # Testing the type of an if condition (line 115)
        if_condition_509590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), return_sign_509589)
        # Assigning a type to the variable 'if_condition_509590' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_509590', if_condition_509590)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to sign(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 's' (line 116)
        s_509593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 's', False)
        # Processing the call keyword arguments (line 116)
        kwargs_509594 = {}
        # Getting the type of 'np' (line 116)
        np_509591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'np', False)
        # Obtaining the member 'sign' of a type (line 116)
        sign_509592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 18), np_509591, 'sign')
        # Calling sign(args, kwargs) (line 116)
        sign_call_result_509595 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), sign_509592, *[s_509593], **kwargs_509594)
        
        # Assigning a type to the variable 'sgn' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'sgn', sign_call_result_509595)
        
        # Getting the type of 's' (line 117)
        s_509596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 's')
        # Getting the type of 'sgn' (line 117)
        sgn_509597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'sgn')
        # Applying the binary operator '*=' (line 117)
        result_imul_509598 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '*=', s_509596, sgn_509597)
        # Assigning a type to the variable 's' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 's', result_imul_509598)
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to log(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 's' (line 118)
        s_509601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 's', False)
        # Processing the call keyword arguments (line 118)
        kwargs_509602 = {}
        # Getting the type of 'np' (line 118)
        np_509599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'np', False)
        # Obtaining the member 'log' of a type (line 118)
        log_509600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 14), np_509599, 'log')
        # Calling log(args, kwargs) (line 118)
        log_call_result_509603 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), log_509600, *[s_509601], **kwargs_509602)
        
        # Assigning a type to the variable 'out' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'out', log_call_result_509603)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 113)
        exit___509604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), errstate_call_result_509576, '__exit__')
        with_exit_509605 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), exit___509604, None, None, None)

    
    
    # Getting the type of 'keepdims' (line 120)
    keepdims_509606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'keepdims')
    # Applying the 'not' unary operator (line 120)
    result_not__509607 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 7), 'not', keepdims_509606)
    
    # Testing the type of an if condition (line 120)
    if_condition_509608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 4), result_not__509607)
    # Assigning a type to the variable 'if_condition_509608' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'if_condition_509608', if_condition_509608)
    # SSA begins for if statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to squeeze(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'a_max' (line 121)
    a_max_509611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'a_max', False)
    # Processing the call keyword arguments (line 121)
    # Getting the type of 'axis' (line 121)
    axis_509612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'axis', False)
    keyword_509613 = axis_509612
    kwargs_509614 = {'axis': keyword_509613}
    # Getting the type of 'np' (line 121)
    np_509609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'np', False)
    # Obtaining the member 'squeeze' of a type (line 121)
    squeeze_509610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), np_509609, 'squeeze')
    # Calling squeeze(args, kwargs) (line 121)
    squeeze_call_result_509615 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), squeeze_509610, *[a_max_509611], **kwargs_509614)
    
    # Assigning a type to the variable 'a_max' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'a_max', squeeze_call_result_509615)
    # SSA join for if statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'out' (line 122)
    out_509616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'out')
    # Getting the type of 'a_max' (line 122)
    a_max_509617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'a_max')
    # Applying the binary operator '+=' (line 122)
    result_iadd_509618 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 4), '+=', out_509616, a_max_509617)
    # Assigning a type to the variable 'out' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'out', result_iadd_509618)
    
    
    # Getting the type of 'return_sign' (line 124)
    return_sign_509619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 7), 'return_sign')
    # Testing the type of an if condition (line 124)
    if_condition_509620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 4), return_sign_509619)
    # Assigning a type to the variable 'if_condition_509620' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'if_condition_509620', if_condition_509620)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 125)
    tuple_509621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 125)
    # Adding element type (line 125)
    # Getting the type of 'out' (line 125)
    out_509622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'out')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_509621, out_509622)
    # Adding element type (line 125)
    # Getting the type of 'sgn' (line 125)
    sgn_509623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'sgn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 15), tuple_509621, sgn_509623)
    
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', tuple_509621)
    # SSA branch for the else part of an if statement (line 124)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'out' (line 127)
    out_509624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'stypy_return_type', out_509624)
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'logsumexp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'logsumexp' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_509625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_509625)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'logsumexp'
    return stypy_return_type_509625

# Assigning a type to the variable 'logsumexp' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'logsumexp', logsumexp)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
