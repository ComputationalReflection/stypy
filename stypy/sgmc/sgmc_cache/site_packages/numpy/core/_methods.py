
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Array methods which are called by both the C-code for the method
3: and the Python code for the NumPy-namespace function
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: import warnings
9: 
10: from numpy.core import multiarray as mu
11: from numpy.core import umath as um
12: from numpy.core.numeric import asanyarray
13: from numpy.core import numerictypes as nt
14: 
15: # save those O(100) nanoseconds!
16: umr_maximum = um.maximum.reduce
17: umr_minimum = um.minimum.reduce
18: umr_sum = um.add.reduce
19: umr_prod = um.multiply.reduce
20: umr_any = um.logical_or.reduce
21: umr_all = um.logical_and.reduce
22: 
23: # avoid keyword arguments to speed up parsing, saves about 15%-20% for very
24: # small reductions
25: def _amax(a, axis=None, out=None, keepdims=False):
26:     return umr_maximum(a, axis, None, out, keepdims)
27: 
28: def _amin(a, axis=None, out=None, keepdims=False):
29:     return umr_minimum(a, axis, None, out, keepdims)
30: 
31: def _sum(a, axis=None, dtype=None, out=None, keepdims=False):
32:     return umr_sum(a, axis, dtype, out, keepdims)
33: 
34: def _prod(a, axis=None, dtype=None, out=None, keepdims=False):
35:     return umr_prod(a, axis, dtype, out, keepdims)
36: 
37: def _any(a, axis=None, dtype=None, out=None, keepdims=False):
38:     return umr_any(a, axis, dtype, out, keepdims)
39: 
40: def _all(a, axis=None, dtype=None, out=None, keepdims=False):
41:     return umr_all(a, axis, dtype, out, keepdims)
42: 
43: def _count_reduce_items(arr, axis):
44:     if axis is None:
45:         axis = tuple(range(arr.ndim))
46:     if not isinstance(axis, tuple):
47:         axis = (axis,)
48:     items = 1
49:     for ax in axis:
50:         items *= arr.shape[ax]
51:     return items
52: 
53: def _mean(a, axis=None, dtype=None, out=None, keepdims=False):
54:     arr = asanyarray(a)
55: 
56:     rcount = _count_reduce_items(arr, axis)
57:     # Make this warning show up first
58:     if rcount == 0:
59:         warnings.warn("Mean of empty slice.", RuntimeWarning)
60: 
61:     # Cast bool, unsigned int, and int to float64 by default
62:     if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
63:         dtype = mu.dtype('f8')
64: 
65:     ret = umr_sum(arr, axis, dtype, out, keepdims)
66:     if isinstance(ret, mu.ndarray):
67:         ret = um.true_divide(
68:                 ret, rcount, out=ret, casting='unsafe', subok=False)
69:     elif hasattr(ret, 'dtype'):
70:         ret = ret.dtype.type(ret / rcount)
71:     else:
72:         ret = ret / rcount
73: 
74:     return ret
75: 
76: def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
77:     arr = asanyarray(a)
78: 
79:     rcount = _count_reduce_items(arr, axis)
80:     # Make this warning show up on top.
81:     if ddof >= rcount:
82:         warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning)
83: 
84:     # Cast bool, unsigned int, and int to float64 by default
85:     if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
86:         dtype = mu.dtype('f8')
87: 
88:     # Compute the mean.
89:     # Note that if dtype is not of inexact type then arraymean will
90:     # not be either.
91:     arrmean = umr_sum(arr, axis, dtype, keepdims=True)
92:     if isinstance(arrmean, mu.ndarray):
93:         arrmean = um.true_divide(
94:                 arrmean, rcount, out=arrmean, casting='unsafe', subok=False)
95:     else:
96:         arrmean = arrmean.dtype.type(arrmean / rcount)
97: 
98:     # Compute sum of squared deviations from mean
99:     # Note that x may not be inexact and that we need it to be an array,
100:     # not a scalar.
101:     x = asanyarray(arr - arrmean)
102:     if issubclass(arr.dtype.type, nt.complexfloating):
103:         x = um.multiply(x, um.conjugate(x), out=x).real
104:     else:
105:         x = um.multiply(x, x, out=x)
106:     ret = umr_sum(x, axis, dtype, out, keepdims)
107: 
108:     # Compute degrees of freedom and make sure it is not negative.
109:     rcount = max([rcount - ddof, 0])
110: 
111:     # divide by degrees of freedom
112:     if isinstance(ret, mu.ndarray):
113:         ret = um.true_divide(
114:                 ret, rcount, out=ret, casting='unsafe', subok=False)
115:     elif hasattr(ret, 'dtype'):
116:         ret = ret.dtype.type(ret / rcount)
117:     else:
118:         ret = ret / rcount
119: 
120:     return ret
121: 
122: def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
123:     ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
124:                keepdims=keepdims)
125: 
126:     if isinstance(ret, mu.ndarray):
127:         ret = um.sqrt(ret, out=ret)
128:     elif hasattr(ret, 'dtype'):
129:         ret = ret.dtype.type(um.sqrt(ret))
130:     else:
131:         ret = um.sqrt(ret)
132: 
133:     return ret
134: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nArray methods which are called by both the C-code for the method\nand the Python code for the NumPy-namespace function\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import mu' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20472 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_20472) is not StypyTypeError):

    if (import_20472 != 'pyd_module'):
        __import__(import_20472)
        sys_modules_20473 = sys.modules[import_20472]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_20473.module_type_store, module_type_store, ['multiarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_20473, sys_modules_20473.module_type_store, module_type_store)
    else:
        from numpy.core import multiarray as mu

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['multiarray'], [mu])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_20472)

# Adding an alias
module_type_store.add_alias('mu', 'multiarray')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.core import um' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20474 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core')

if (type(import_20474) is not StypyTypeError):

    if (import_20474 != 'pyd_module'):
        __import__(import_20474)
        sys_modules_20475 = sys.modules[import_20474]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', sys_modules_20475.module_type_store, module_type_store, ['umath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_20475, sys_modules_20475.module_type_store, module_type_store)
    else:
        from numpy.core import umath as um

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', None, module_type_store, ['umath'], [um])

else:
    # Assigning a type to the variable 'numpy.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', import_20474)

# Adding an alias
module_type_store.add_alias('um', 'umath')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.core.numeric import asanyarray' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20476 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric')

if (type(import_20476) is not StypyTypeError):

    if (import_20476 != 'pyd_module'):
        __import__(import_20476)
        sys_modules_20477 = sys.modules[import_20476]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', sys_modules_20477.module_type_store, module_type_store, ['asanyarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_20477, sys_modules_20477.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asanyarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', None, module_type_store, ['asanyarray'], [asanyarray])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core.numeric', import_20476)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.core import nt' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20478 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core')

if (type(import_20478) is not StypyTypeError):

    if (import_20478 != 'pyd_module'):
        __import__(import_20478)
        sys_modules_20479 = sys.modules[import_20478]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', sys_modules_20479.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_20479, sys_modules_20479.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as nt

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [nt])

else:
    # Assigning a type to the variable 'numpy.core' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', import_20478)

# Adding an alias
module_type_store.add_alias('nt', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Attribute to a Name (line 16):
# Getting the type of 'um' (line 16)
um_20480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'um')
# Obtaining the member 'maximum' of a type (line 16)
maximum_20481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), um_20480, 'maximum')
# Obtaining the member 'reduce' of a type (line 16)
reduce_20482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), maximum_20481, 'reduce')
# Assigning a type to the variable 'umr_maximum' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'umr_maximum', reduce_20482)

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'um' (line 17)
um_20483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'um')
# Obtaining the member 'minimum' of a type (line 17)
minimum_20484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), um_20483, 'minimum')
# Obtaining the member 'reduce' of a type (line 17)
reduce_20485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), minimum_20484, 'reduce')
# Assigning a type to the variable 'umr_minimum' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'umr_minimum', reduce_20485)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'um' (line 18)
um_20486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'um')
# Obtaining the member 'add' of a type (line 18)
add_20487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), um_20486, 'add')
# Obtaining the member 'reduce' of a type (line 18)
reduce_20488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), add_20487, 'reduce')
# Assigning a type to the variable 'umr_sum' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'umr_sum', reduce_20488)

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'um' (line 19)
um_20489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'um')
# Obtaining the member 'multiply' of a type (line 19)
multiply_20490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), um_20489, 'multiply')
# Obtaining the member 'reduce' of a type (line 19)
reduce_20491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 11), multiply_20490, 'reduce')
# Assigning a type to the variable 'umr_prod' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'umr_prod', reduce_20491)

# Assigning a Attribute to a Name (line 20):
# Getting the type of 'um' (line 20)
um_20492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'um')
# Obtaining the member 'logical_or' of a type (line 20)
logical_or_20493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 10), um_20492, 'logical_or')
# Obtaining the member 'reduce' of a type (line 20)
reduce_20494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 10), logical_or_20493, 'reduce')
# Assigning a type to the variable 'umr_any' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'umr_any', reduce_20494)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'um' (line 21)
um_20495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'um')
# Obtaining the member 'logical_and' of a type (line 21)
logical_and_20496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), um_20495, 'logical_and')
# Obtaining the member 'reduce' of a type (line 21)
reduce_20497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), logical_and_20496, 'reduce')
# Assigning a type to the variable 'umr_all' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'umr_all', reduce_20497)

@norecursion
def _amax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 25)
    None_20498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'None')
    # Getting the type of 'None' (line 25)
    None_20499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'None')
    # Getting the type of 'False' (line 25)
    False_20500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 43), 'False')
    defaults = [None_20498, None_20499, False_20500]
    # Create a new context for function '_amax'
    module_type_store = module_type_store.open_function_context('_amax', 25, 0, False)
    
    # Passed parameters checking function
    _amax.stypy_localization = localization
    _amax.stypy_type_of_self = None
    _amax.stypy_type_store = module_type_store
    _amax.stypy_function_name = '_amax'
    _amax.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    _amax.stypy_varargs_param_name = None
    _amax.stypy_kwargs_param_name = None
    _amax.stypy_call_defaults = defaults
    _amax.stypy_call_varargs = varargs
    _amax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_amax', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_amax', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_amax(...)' code ##################

    
    # Call to umr_maximum(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'a' (line 26)
    a_20502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'a', False)
    # Getting the type of 'axis' (line 26)
    axis_20503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'axis', False)
    # Getting the type of 'None' (line 26)
    None_20504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 32), 'None', False)
    # Getting the type of 'out' (line 26)
    out_20505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'out', False)
    # Getting the type of 'keepdims' (line 26)
    keepdims_20506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 43), 'keepdims', False)
    # Processing the call keyword arguments (line 26)
    kwargs_20507 = {}
    # Getting the type of 'umr_maximum' (line 26)
    umr_maximum_20501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'umr_maximum', False)
    # Calling umr_maximum(args, kwargs) (line 26)
    umr_maximum_call_result_20508 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), umr_maximum_20501, *[a_20502, axis_20503, None_20504, out_20505, keepdims_20506], **kwargs_20507)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', umr_maximum_call_result_20508)
    
    # ################# End of '_amax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_amax' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_20509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20509)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_amax'
    return stypy_return_type_20509

# Assigning a type to the variable '_amax' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_amax', _amax)

@norecursion
def _amin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 28)
    None_20510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'None')
    # Getting the type of 'None' (line 28)
    None_20511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'None')
    # Getting the type of 'False' (line 28)
    False_20512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'False')
    defaults = [None_20510, None_20511, False_20512]
    # Create a new context for function '_amin'
    module_type_store = module_type_store.open_function_context('_amin', 28, 0, False)
    
    # Passed parameters checking function
    _amin.stypy_localization = localization
    _amin.stypy_type_of_self = None
    _amin.stypy_type_store = module_type_store
    _amin.stypy_function_name = '_amin'
    _amin.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    _amin.stypy_varargs_param_name = None
    _amin.stypy_kwargs_param_name = None
    _amin.stypy_call_defaults = defaults
    _amin.stypy_call_varargs = varargs
    _amin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_amin', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_amin', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_amin(...)' code ##################

    
    # Call to umr_minimum(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'a' (line 29)
    a_20514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'a', False)
    # Getting the type of 'axis' (line 29)
    axis_20515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'axis', False)
    # Getting the type of 'None' (line 29)
    None_20516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 32), 'None', False)
    # Getting the type of 'out' (line 29)
    out_20517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 38), 'out', False)
    # Getting the type of 'keepdims' (line 29)
    keepdims_20518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'keepdims', False)
    # Processing the call keyword arguments (line 29)
    kwargs_20519 = {}
    # Getting the type of 'umr_minimum' (line 29)
    umr_minimum_20513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'umr_minimum', False)
    # Calling umr_minimum(args, kwargs) (line 29)
    umr_minimum_call_result_20520 = invoke(stypy.reporting.localization.Localization(__file__, 29, 11), umr_minimum_20513, *[a_20514, axis_20515, None_20516, out_20517, keepdims_20518], **kwargs_20519)
    
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', umr_minimum_call_result_20520)
    
    # ################# End of '_amin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_amin' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_20521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20521)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_amin'
    return stypy_return_type_20521

# Assigning a type to the variable '_amin' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_amin', _amin)

@norecursion
def _sum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 31)
    None_20522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'None')
    # Getting the type of 'None' (line 31)
    None_20523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'None')
    # Getting the type of 'None' (line 31)
    None_20524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 39), 'None')
    # Getting the type of 'False' (line 31)
    False_20525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 54), 'False')
    defaults = [None_20522, None_20523, None_20524, False_20525]
    # Create a new context for function '_sum'
    module_type_store = module_type_store.open_function_context('_sum', 31, 0, False)
    
    # Passed parameters checking function
    _sum.stypy_localization = localization
    _sum.stypy_type_of_self = None
    _sum.stypy_type_store = module_type_store
    _sum.stypy_function_name = '_sum'
    _sum.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    _sum.stypy_varargs_param_name = None
    _sum.stypy_kwargs_param_name = None
    _sum.stypy_call_defaults = defaults
    _sum.stypy_call_varargs = varargs
    _sum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sum', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sum', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sum(...)' code ##################

    
    # Call to umr_sum(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'a' (line 32)
    a_20527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'a', False)
    # Getting the type of 'axis' (line 32)
    axis_20528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'axis', False)
    # Getting the type of 'dtype' (line 32)
    dtype_20529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'dtype', False)
    # Getting the type of 'out' (line 32)
    out_20530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'out', False)
    # Getting the type of 'keepdims' (line 32)
    keepdims_20531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'keepdims', False)
    # Processing the call keyword arguments (line 32)
    kwargs_20532 = {}
    # Getting the type of 'umr_sum' (line 32)
    umr_sum_20526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'umr_sum', False)
    # Calling umr_sum(args, kwargs) (line 32)
    umr_sum_call_result_20533 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), umr_sum_20526, *[a_20527, axis_20528, dtype_20529, out_20530, keepdims_20531], **kwargs_20532)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', umr_sum_call_result_20533)
    
    # ################# End of '_sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sum' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_20534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20534)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sum'
    return stypy_return_type_20534

# Assigning a type to the variable '_sum' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_sum', _sum)

@norecursion
def _prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 34)
    None_20535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'None')
    # Getting the type of 'None' (line 34)
    None_20536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'None')
    # Getting the type of 'None' (line 34)
    None_20537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'None')
    # Getting the type of 'False' (line 34)
    False_20538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 55), 'False')
    defaults = [None_20535, None_20536, None_20537, False_20538]
    # Create a new context for function '_prod'
    module_type_store = module_type_store.open_function_context('_prod', 34, 0, False)
    
    # Passed parameters checking function
    _prod.stypy_localization = localization
    _prod.stypy_type_of_self = None
    _prod.stypy_type_store = module_type_store
    _prod.stypy_function_name = '_prod'
    _prod.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    _prod.stypy_varargs_param_name = None
    _prod.stypy_kwargs_param_name = None
    _prod.stypy_call_defaults = defaults
    _prod.stypy_call_varargs = varargs
    _prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prod', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prod', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prod(...)' code ##################

    
    # Call to umr_prod(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'a' (line 35)
    a_20540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'a', False)
    # Getting the type of 'axis' (line 35)
    axis_20541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'axis', False)
    # Getting the type of 'dtype' (line 35)
    dtype_20542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'dtype', False)
    # Getting the type of 'out' (line 35)
    out_20543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'out', False)
    # Getting the type of 'keepdims' (line 35)
    keepdims_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'keepdims', False)
    # Processing the call keyword arguments (line 35)
    kwargs_20545 = {}
    # Getting the type of 'umr_prod' (line 35)
    umr_prod_20539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'umr_prod', False)
    # Calling umr_prod(args, kwargs) (line 35)
    umr_prod_call_result_20546 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), umr_prod_20539, *[a_20540, axis_20541, dtype_20542, out_20543, keepdims_20544], **kwargs_20545)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', umr_prod_call_result_20546)
    
    # ################# End of '_prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prod' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_20547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20547)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prod'
    return stypy_return_type_20547

# Assigning a type to the variable '_prod' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '_prod', _prod)

@norecursion
def _any(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 37)
    None_20548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'None')
    # Getting the type of 'None' (line 37)
    None_20549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'None')
    # Getting the type of 'None' (line 37)
    None_20550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'None')
    # Getting the type of 'False' (line 37)
    False_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 54), 'False')
    defaults = [None_20548, None_20549, None_20550, False_20551]
    # Create a new context for function '_any'
    module_type_store = module_type_store.open_function_context('_any', 37, 0, False)
    
    # Passed parameters checking function
    _any.stypy_localization = localization
    _any.stypy_type_of_self = None
    _any.stypy_type_store = module_type_store
    _any.stypy_function_name = '_any'
    _any.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    _any.stypy_varargs_param_name = None
    _any.stypy_kwargs_param_name = None
    _any.stypy_call_defaults = defaults
    _any.stypy_call_varargs = varargs
    _any.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_any', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_any', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_any(...)' code ##################

    
    # Call to umr_any(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'a' (line 38)
    a_20553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'a', False)
    # Getting the type of 'axis' (line 38)
    axis_20554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'axis', False)
    # Getting the type of 'dtype' (line 38)
    dtype_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'dtype', False)
    # Getting the type of 'out' (line 38)
    out_20556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), 'out', False)
    # Getting the type of 'keepdims' (line 38)
    keepdims_20557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 40), 'keepdims', False)
    # Processing the call keyword arguments (line 38)
    kwargs_20558 = {}
    # Getting the type of 'umr_any' (line 38)
    umr_any_20552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'umr_any', False)
    # Calling umr_any(args, kwargs) (line 38)
    umr_any_call_result_20559 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), umr_any_20552, *[a_20553, axis_20554, dtype_20555, out_20556, keepdims_20557], **kwargs_20558)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', umr_any_call_result_20559)
    
    # ################# End of '_any(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_any' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_20560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20560)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_any'
    return stypy_return_type_20560

# Assigning a type to the variable '_any' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_any', _any)

@norecursion
def _all(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 40)
    None_20561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'None')
    # Getting the type of 'None' (line 40)
    None_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'None')
    # Getting the type of 'None' (line 40)
    None_20563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'None')
    # Getting the type of 'False' (line 40)
    False_20564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 54), 'False')
    defaults = [None_20561, None_20562, None_20563, False_20564]
    # Create a new context for function '_all'
    module_type_store = module_type_store.open_function_context('_all', 40, 0, False)
    
    # Passed parameters checking function
    _all.stypy_localization = localization
    _all.stypy_type_of_self = None
    _all.stypy_type_store = module_type_store
    _all.stypy_function_name = '_all'
    _all.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    _all.stypy_varargs_param_name = None
    _all.stypy_kwargs_param_name = None
    _all.stypy_call_defaults = defaults
    _all.stypy_call_varargs = varargs
    _all.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_all', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_all', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_all(...)' code ##################

    
    # Call to umr_all(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'a' (line 41)
    a_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'a', False)
    # Getting the type of 'axis' (line 41)
    axis_20567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'axis', False)
    # Getting the type of 'dtype' (line 41)
    dtype_20568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'dtype', False)
    # Getting the type of 'out' (line 41)
    out_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'out', False)
    # Getting the type of 'keepdims' (line 41)
    keepdims_20570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'keepdims', False)
    # Processing the call keyword arguments (line 41)
    kwargs_20571 = {}
    # Getting the type of 'umr_all' (line 41)
    umr_all_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'umr_all', False)
    # Calling umr_all(args, kwargs) (line 41)
    umr_all_call_result_20572 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), umr_all_20565, *[a_20566, axis_20567, dtype_20568, out_20569, keepdims_20570], **kwargs_20571)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', umr_all_call_result_20572)
    
    # ################# End of '_all(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_all' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_20573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_all'
    return stypy_return_type_20573

# Assigning a type to the variable '_all' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '_all', _all)

@norecursion
def _count_reduce_items(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_count_reduce_items'
    module_type_store = module_type_store.open_function_context('_count_reduce_items', 43, 0, False)
    
    # Passed parameters checking function
    _count_reduce_items.stypy_localization = localization
    _count_reduce_items.stypy_type_of_self = None
    _count_reduce_items.stypy_type_store = module_type_store
    _count_reduce_items.stypy_function_name = '_count_reduce_items'
    _count_reduce_items.stypy_param_names_list = ['arr', 'axis']
    _count_reduce_items.stypy_varargs_param_name = None
    _count_reduce_items.stypy_kwargs_param_name = None
    _count_reduce_items.stypy_call_defaults = defaults
    _count_reduce_items.stypy_call_varargs = varargs
    _count_reduce_items.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_count_reduce_items', ['arr', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_count_reduce_items', localization, ['arr', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_count_reduce_items(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 44)
    # Getting the type of 'axis' (line 44)
    axis_20574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'axis')
    # Getting the type of 'None' (line 44)
    None_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'None')
    
    (may_be_20576, more_types_in_union_20577) = may_be_none(axis_20574, None_20575)

    if may_be_20576:

        if more_types_in_union_20577:
            # Runtime conditional SSA (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 45):
        
        # Call to tuple(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to range(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'arr' (line 45)
        arr_20580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'arr', False)
        # Obtaining the member 'ndim' of a type (line 45)
        ndim_20581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), arr_20580, 'ndim')
        # Processing the call keyword arguments (line 45)
        kwargs_20582 = {}
        # Getting the type of 'range' (line 45)
        range_20579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'range', False)
        # Calling range(args, kwargs) (line 45)
        range_call_result_20583 = invoke(stypy.reporting.localization.Localization(__file__, 45, 21), range_20579, *[ndim_20581], **kwargs_20582)
        
        # Processing the call keyword arguments (line 45)
        kwargs_20584 = {}
        # Getting the type of 'tuple' (line 45)
        tuple_20578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 45)
        tuple_call_result_20585 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), tuple_20578, *[range_call_result_20583], **kwargs_20584)
        
        # Assigning a type to the variable 'axis' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'axis', tuple_call_result_20585)

        if more_types_in_union_20577:
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 46)
    # Getting the type of 'tuple' (line 46)
    tuple_20586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'tuple')
    # Getting the type of 'axis' (line 46)
    axis_20587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'axis')
    
    (may_be_20588, more_types_in_union_20589) = may_not_be_subtype(tuple_20586, axis_20587)

    if may_be_20588:

        if more_types_in_union_20589:
            # Runtime conditional SSA (line 46)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'axis' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'axis', remove_subtype_from_union(axis_20587, tuple))
        
        # Assigning a Tuple to a Name (line 47):
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_20590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        # Getting the type of 'axis' (line 47)
        axis_20591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'axis')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 16), tuple_20590, axis_20591)
        
        # Assigning a type to the variable 'axis' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'axis', tuple_20590)

        if more_types_in_union_20589:
            # SSA join for if statement (line 46)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Name (line 48):
    int_20592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 12), 'int')
    # Assigning a type to the variable 'items' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'items', int_20592)
    
    # Getting the type of 'axis' (line 49)
    axis_20593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 14), 'axis')
    # Testing the type of a for loop iterable (line 49)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 4), axis_20593)
    # Getting the type of the for loop variable (line 49)
    for_loop_var_20594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 4), axis_20593)
    # Assigning a type to the variable 'ax' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'ax', for_loop_var_20594)
    # SSA begins for a for statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'items' (line 50)
    items_20595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'items')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ax' (line 50)
    ax_20596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'ax')
    # Getting the type of 'arr' (line 50)
    arr_20597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'arr')
    # Obtaining the member 'shape' of a type (line 50)
    shape_20598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), arr_20597, 'shape')
    # Obtaining the member '__getitem__' of a type (line 50)
    getitem___20599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), shape_20598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 50)
    subscript_call_result_20600 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), getitem___20599, ax_20596)
    
    # Applying the binary operator '*=' (line 50)
    result_imul_20601 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 8), '*=', items_20595, subscript_call_result_20600)
    # Assigning a type to the variable 'items' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'items', result_imul_20601)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'items' (line 51)
    items_20602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'items')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', items_20602)
    
    # ################# End of '_count_reduce_items(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_count_reduce_items' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_20603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_count_reduce_items'
    return stypy_return_type_20603

# Assigning a type to the variable '_count_reduce_items' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '_count_reduce_items', _count_reduce_items)

@norecursion
def _mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 53)
    None_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'None')
    # Getting the type of 'None' (line 53)
    None_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 30), 'None')
    # Getting the type of 'None' (line 53)
    None_20606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'None')
    # Getting the type of 'False' (line 53)
    False_20607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 55), 'False')
    defaults = [None_20604, None_20605, None_20606, False_20607]
    # Create a new context for function '_mean'
    module_type_store = module_type_store.open_function_context('_mean', 53, 0, False)
    
    # Passed parameters checking function
    _mean.stypy_localization = localization
    _mean.stypy_type_of_self = None
    _mean.stypy_type_store = module_type_store
    _mean.stypy_function_name = '_mean'
    _mean.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    _mean.stypy_varargs_param_name = None
    _mean.stypy_kwargs_param_name = None
    _mean.stypy_call_defaults = defaults
    _mean.stypy_call_varargs = varargs
    _mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_mean', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_mean', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_mean(...)' code ##################

    
    # Assigning a Call to a Name (line 54):
    
    # Call to asanyarray(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'a' (line 54)
    a_20609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'a', False)
    # Processing the call keyword arguments (line 54)
    kwargs_20610 = {}
    # Getting the type of 'asanyarray' (line 54)
    asanyarray_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 54)
    asanyarray_call_result_20611 = invoke(stypy.reporting.localization.Localization(__file__, 54, 10), asanyarray_20608, *[a_20609], **kwargs_20610)
    
    # Assigning a type to the variable 'arr' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'arr', asanyarray_call_result_20611)
    
    # Assigning a Call to a Name (line 56):
    
    # Call to _count_reduce_items(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'arr' (line 56)
    arr_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'arr', False)
    # Getting the type of 'axis' (line 56)
    axis_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'axis', False)
    # Processing the call keyword arguments (line 56)
    kwargs_20615 = {}
    # Getting the type of '_count_reduce_items' (line 56)
    _count_reduce_items_20612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 13), '_count_reduce_items', False)
    # Calling _count_reduce_items(args, kwargs) (line 56)
    _count_reduce_items_call_result_20616 = invoke(stypy.reporting.localization.Localization(__file__, 56, 13), _count_reduce_items_20612, *[arr_20613, axis_20614], **kwargs_20615)
    
    # Assigning a type to the variable 'rcount' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'rcount', _count_reduce_items_call_result_20616)
    
    
    # Getting the type of 'rcount' (line 58)
    rcount_20617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'rcount')
    int_20618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'int')
    # Applying the binary operator '==' (line 58)
    result_eq_20619 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), '==', rcount_20617, int_20618)
    
    # Testing the type of an if condition (line 58)
    if_condition_20620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_eq_20619)
    # Assigning a type to the variable 'if_condition_20620' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_20620', if_condition_20620)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 59)
    # Processing the call arguments (line 59)
    str_20623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'str', 'Mean of empty slice.')
    # Getting the type of 'RuntimeWarning' (line 59)
    RuntimeWarning_20624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 59)
    kwargs_20625 = {}
    # Getting the type of 'warnings' (line 59)
    warnings_20621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 59)
    warn_20622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), warnings_20621, 'warn')
    # Calling warn(args, kwargs) (line 59)
    warn_call_result_20626 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), warn_20622, *[str_20623, RuntimeWarning_20624], **kwargs_20625)
    
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 62)
    dtype_20627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 7), 'dtype')
    # Getting the type of 'None' (line 62)
    None_20628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'None')
    # Applying the binary operator 'is' (line 62)
    result_is__20629 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 7), 'is', dtype_20627, None_20628)
    
    
    # Call to issubclass(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'arr' (line 62)
    arr_20631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 36), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 62)
    dtype_20632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 36), arr_20631, 'dtype')
    # Obtaining the member 'type' of a type (line 62)
    type_20633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 36), dtype_20632, 'type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_20634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    # Getting the type of 'nt' (line 62)
    nt_20635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 53), 'nt', False)
    # Obtaining the member 'integer' of a type (line 62)
    integer_20636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 53), nt_20635, 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 53), tuple_20634, integer_20636)
    # Adding element type (line 62)
    # Getting the type of 'nt' (line 62)
    nt_20637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 65), 'nt', False)
    # Obtaining the member 'bool_' of a type (line 62)
    bool__20638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 65), nt_20637, 'bool_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 53), tuple_20634, bool__20638)
    
    # Processing the call keyword arguments (line 62)
    kwargs_20639 = {}
    # Getting the type of 'issubclass' (line 62)
    issubclass_20630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 62)
    issubclass_call_result_20640 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), issubclass_20630, *[type_20633, tuple_20634], **kwargs_20639)
    
    # Applying the binary operator 'and' (line 62)
    result_and_keyword_20641 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 7), 'and', result_is__20629, issubclass_call_result_20640)
    
    # Testing the type of an if condition (line 62)
    if_condition_20642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), result_and_keyword_20641)
    # Assigning a type to the variable 'if_condition_20642' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'if_condition_20642', if_condition_20642)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to dtype(...): (line 63)
    # Processing the call arguments (line 63)
    str_20645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'str', 'f8')
    # Processing the call keyword arguments (line 63)
    kwargs_20646 = {}
    # Getting the type of 'mu' (line 63)
    mu_20643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'mu', False)
    # Obtaining the member 'dtype' of a type (line 63)
    dtype_20644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), mu_20643, 'dtype')
    # Calling dtype(args, kwargs) (line 63)
    dtype_call_result_20647 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), dtype_20644, *[str_20645], **kwargs_20646)
    
    # Assigning a type to the variable 'dtype' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dtype', dtype_call_result_20647)
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 65):
    
    # Call to umr_sum(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'arr' (line 65)
    arr_20649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'arr', False)
    # Getting the type of 'axis' (line 65)
    axis_20650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'axis', False)
    # Getting the type of 'dtype' (line 65)
    dtype_20651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'dtype', False)
    # Getting the type of 'out' (line 65)
    out_20652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'out', False)
    # Getting the type of 'keepdims' (line 65)
    keepdims_20653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'keepdims', False)
    # Processing the call keyword arguments (line 65)
    kwargs_20654 = {}
    # Getting the type of 'umr_sum' (line 65)
    umr_sum_20648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), 'umr_sum', False)
    # Calling umr_sum(args, kwargs) (line 65)
    umr_sum_call_result_20655 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), umr_sum_20648, *[arr_20649, axis_20650, dtype_20651, out_20652, keepdims_20653], **kwargs_20654)
    
    # Assigning a type to the variable 'ret' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'ret', umr_sum_call_result_20655)
    
    
    # Call to isinstance(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'ret' (line 66)
    ret_20657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 18), 'ret', False)
    # Getting the type of 'mu' (line 66)
    mu_20658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'mu', False)
    # Obtaining the member 'ndarray' of a type (line 66)
    ndarray_20659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), mu_20658, 'ndarray')
    # Processing the call keyword arguments (line 66)
    kwargs_20660 = {}
    # Getting the type of 'isinstance' (line 66)
    isinstance_20656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 66)
    isinstance_call_result_20661 = invoke(stypy.reporting.localization.Localization(__file__, 66, 7), isinstance_20656, *[ret_20657, ndarray_20659], **kwargs_20660)
    
    # Testing the type of an if condition (line 66)
    if_condition_20662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), isinstance_call_result_20661)
    # Assigning a type to the variable 'if_condition_20662' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_20662', if_condition_20662)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 67):
    
    # Call to true_divide(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'ret' (line 68)
    ret_20665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'ret', False)
    # Getting the type of 'rcount' (line 68)
    rcount_20666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'rcount', False)
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'ret' (line 68)
    ret_20667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'ret', False)
    keyword_20668 = ret_20667
    str_20669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'str', 'unsafe')
    keyword_20670 = str_20669
    # Getting the type of 'False' (line 68)
    False_20671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 62), 'False', False)
    keyword_20672 = False_20671
    kwargs_20673 = {'subok': keyword_20672, 'casting': keyword_20670, 'out': keyword_20668}
    # Getting the type of 'um' (line 67)
    um_20663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'um', False)
    # Obtaining the member 'true_divide' of a type (line 67)
    true_divide_20664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 14), um_20663, 'true_divide')
    # Calling true_divide(args, kwargs) (line 67)
    true_divide_call_result_20674 = invoke(stypy.reporting.localization.Localization(__file__, 67, 14), true_divide_20664, *[ret_20665, rcount_20666], **kwargs_20673)
    
    # Assigning a type to the variable 'ret' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'ret', true_divide_call_result_20674)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 69)
    str_20675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'str', 'dtype')
    # Getting the type of 'ret' (line 69)
    ret_20676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'ret')
    
    (may_be_20677, more_types_in_union_20678) = may_provide_member(str_20675, ret_20676)

    if may_be_20677:

        if more_types_in_union_20678:
            # Runtime conditional SSA (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'ret' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'ret', remove_not_member_provider_from_union(ret_20676, 'dtype'))
        
        # Assigning a Call to a Name (line 70):
        
        # Call to type(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'ret' (line 70)
        ret_20682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 29), 'ret', False)
        # Getting the type of 'rcount' (line 70)
        rcount_20683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'rcount', False)
        # Applying the binary operator 'div' (line 70)
        result_div_20684 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 29), 'div', ret_20682, rcount_20683)
        
        # Processing the call keyword arguments (line 70)
        kwargs_20685 = {}
        # Getting the type of 'ret' (line 70)
        ret_20679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'ret', False)
        # Obtaining the member 'dtype' of a type (line 70)
        dtype_20680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), ret_20679, 'dtype')
        # Obtaining the member 'type' of a type (line 70)
        type_20681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 14), dtype_20680, 'type')
        # Calling type(args, kwargs) (line 70)
        type_call_result_20686 = invoke(stypy.reporting.localization.Localization(__file__, 70, 14), type_20681, *[result_div_20684], **kwargs_20685)
        
        # Assigning a type to the variable 'ret' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'ret', type_call_result_20686)

        if more_types_in_union_20678:
            # Runtime conditional SSA for else branch (line 69)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20677) or more_types_in_union_20678):
        # Assigning a type to the variable 'ret' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'ret', remove_member_provider_from_union(ret_20676, 'dtype'))
        
        # Assigning a BinOp to a Name (line 72):
        # Getting the type of 'ret' (line 72)
        ret_20687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 14), 'ret')
        # Getting the type of 'rcount' (line 72)
        rcount_20688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'rcount')
        # Applying the binary operator 'div' (line 72)
        result_div_20689 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 14), 'div', ret_20687, rcount_20688)
        
        # Assigning a type to the variable 'ret' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'ret', result_div_20689)

        if (may_be_20677 and more_types_in_union_20678):
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 74)
    ret_20690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', ret_20690)
    
    # ################# End of '_mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_mean' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_20691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20691)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_mean'
    return stypy_return_type_20691

# Assigning a type to the variable '_mean' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '_mean', _mean)

@norecursion
def _var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 76)
    None_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'None')
    # Getting the type of 'None' (line 76)
    None_20693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'None')
    # Getting the type of 'None' (line 76)
    None_20694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 39), 'None')
    int_20695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 50), 'int')
    # Getting the type of 'False' (line 76)
    False_20696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 62), 'False')
    defaults = [None_20692, None_20693, None_20694, int_20695, False_20696]
    # Create a new context for function '_var'
    module_type_store = module_type_store.open_function_context('_var', 76, 0, False)
    
    # Passed parameters checking function
    _var.stypy_localization = localization
    _var.stypy_type_of_self = None
    _var.stypy_type_store = module_type_store
    _var.stypy_function_name = '_var'
    _var.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    _var.stypy_varargs_param_name = None
    _var.stypy_kwargs_param_name = None
    _var.stypy_call_defaults = defaults
    _var.stypy_call_varargs = varargs
    _var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_var', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_var', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_var(...)' code ##################

    
    # Assigning a Call to a Name (line 77):
    
    # Call to asanyarray(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'a' (line 77)
    a_20698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'a', False)
    # Processing the call keyword arguments (line 77)
    kwargs_20699 = {}
    # Getting the type of 'asanyarray' (line 77)
    asanyarray_20697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 77)
    asanyarray_call_result_20700 = invoke(stypy.reporting.localization.Localization(__file__, 77, 10), asanyarray_20697, *[a_20698], **kwargs_20699)
    
    # Assigning a type to the variable 'arr' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'arr', asanyarray_call_result_20700)
    
    # Assigning a Call to a Name (line 79):
    
    # Call to _count_reduce_items(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'arr' (line 79)
    arr_20702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'arr', False)
    # Getting the type of 'axis' (line 79)
    axis_20703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'axis', False)
    # Processing the call keyword arguments (line 79)
    kwargs_20704 = {}
    # Getting the type of '_count_reduce_items' (line 79)
    _count_reduce_items_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), '_count_reduce_items', False)
    # Calling _count_reduce_items(args, kwargs) (line 79)
    _count_reduce_items_call_result_20705 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), _count_reduce_items_20701, *[arr_20702, axis_20703], **kwargs_20704)
    
    # Assigning a type to the variable 'rcount' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'rcount', _count_reduce_items_call_result_20705)
    
    
    # Getting the type of 'ddof' (line 81)
    ddof_20706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'ddof')
    # Getting the type of 'rcount' (line 81)
    rcount_20707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'rcount')
    # Applying the binary operator '>=' (line 81)
    result_ge_20708 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 7), '>=', ddof_20706, rcount_20707)
    
    # Testing the type of an if condition (line 81)
    if_condition_20709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_ge_20708)
    # Assigning a type to the variable 'if_condition_20709' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_20709', if_condition_20709)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 82)
    # Processing the call arguments (line 82)
    str_20712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'str', 'Degrees of freedom <= 0 for slice')
    # Getting the type of 'RuntimeWarning' (line 82)
    RuntimeWarning_20713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 59), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 82)
    kwargs_20714 = {}
    # Getting the type of 'warnings' (line 82)
    warnings_20710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 82)
    warn_20711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), warnings_20710, 'warn')
    # Calling warn(args, kwargs) (line 82)
    warn_call_result_20715 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), warn_20711, *[str_20712, RuntimeWarning_20713], **kwargs_20714)
    
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 85)
    dtype_20716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'dtype')
    # Getting the type of 'None' (line 85)
    None_20717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'None')
    # Applying the binary operator 'is' (line 85)
    result_is__20718 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), 'is', dtype_20716, None_20717)
    
    
    # Call to issubclass(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'arr' (line 85)
    arr_20720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 85)
    dtype_20721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), arr_20720, 'dtype')
    # Obtaining the member 'type' of a type (line 85)
    type_20722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), dtype_20721, 'type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_20723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    # Getting the type of 'nt' (line 85)
    nt_20724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 53), 'nt', False)
    # Obtaining the member 'integer' of a type (line 85)
    integer_20725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 53), nt_20724, 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 53), tuple_20723, integer_20725)
    # Adding element type (line 85)
    # Getting the type of 'nt' (line 85)
    nt_20726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 65), 'nt', False)
    # Obtaining the member 'bool_' of a type (line 85)
    bool__20727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 65), nt_20726, 'bool_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 53), tuple_20723, bool__20727)
    
    # Processing the call keyword arguments (line 85)
    kwargs_20728 = {}
    # Getting the type of 'issubclass' (line 85)
    issubclass_20719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 85)
    issubclass_call_result_20729 = invoke(stypy.reporting.localization.Localization(__file__, 85, 25), issubclass_20719, *[type_20722, tuple_20723], **kwargs_20728)
    
    # Applying the binary operator 'and' (line 85)
    result_and_keyword_20730 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), 'and', result_is__20718, issubclass_call_result_20729)
    
    # Testing the type of an if condition (line 85)
    if_condition_20731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_and_keyword_20730)
    # Assigning a type to the variable 'if_condition_20731' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_20731', if_condition_20731)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 86):
    
    # Call to dtype(...): (line 86)
    # Processing the call arguments (line 86)
    str_20734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', 'f8')
    # Processing the call keyword arguments (line 86)
    kwargs_20735 = {}
    # Getting the type of 'mu' (line 86)
    mu_20732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'mu', False)
    # Obtaining the member 'dtype' of a type (line 86)
    dtype_20733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), mu_20732, 'dtype')
    # Calling dtype(args, kwargs) (line 86)
    dtype_call_result_20736 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), dtype_20733, *[str_20734], **kwargs_20735)
    
    # Assigning a type to the variable 'dtype' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'dtype', dtype_call_result_20736)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 91):
    
    # Call to umr_sum(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'arr' (line 91)
    arr_20738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'arr', False)
    # Getting the type of 'axis' (line 91)
    axis_20739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'axis', False)
    # Getting the type of 'dtype' (line 91)
    dtype_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'dtype', False)
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'True' (line 91)
    True_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'True', False)
    keyword_20742 = True_20741
    kwargs_20743 = {'keepdims': keyword_20742}
    # Getting the type of 'umr_sum' (line 91)
    umr_sum_20737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'umr_sum', False)
    # Calling umr_sum(args, kwargs) (line 91)
    umr_sum_call_result_20744 = invoke(stypy.reporting.localization.Localization(__file__, 91, 14), umr_sum_20737, *[arr_20738, axis_20739, dtype_20740], **kwargs_20743)
    
    # Assigning a type to the variable 'arrmean' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'arrmean', umr_sum_call_result_20744)
    
    
    # Call to isinstance(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'arrmean' (line 92)
    arrmean_20746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'arrmean', False)
    # Getting the type of 'mu' (line 92)
    mu_20747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'mu', False)
    # Obtaining the member 'ndarray' of a type (line 92)
    ndarray_20748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 27), mu_20747, 'ndarray')
    # Processing the call keyword arguments (line 92)
    kwargs_20749 = {}
    # Getting the type of 'isinstance' (line 92)
    isinstance_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 92)
    isinstance_call_result_20750 = invoke(stypy.reporting.localization.Localization(__file__, 92, 7), isinstance_20745, *[arrmean_20746, ndarray_20748], **kwargs_20749)
    
    # Testing the type of an if condition (line 92)
    if_condition_20751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 4), isinstance_call_result_20750)
    # Assigning a type to the variable 'if_condition_20751' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'if_condition_20751', if_condition_20751)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 93):
    
    # Call to true_divide(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'arrmean' (line 94)
    arrmean_20754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'arrmean', False)
    # Getting the type of 'rcount' (line 94)
    rcount_20755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'rcount', False)
    # Processing the call keyword arguments (line 93)
    # Getting the type of 'arrmean' (line 94)
    arrmean_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'arrmean', False)
    keyword_20757 = arrmean_20756
    str_20758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 54), 'str', 'unsafe')
    keyword_20759 = str_20758
    # Getting the type of 'False' (line 94)
    False_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 70), 'False', False)
    keyword_20761 = False_20760
    kwargs_20762 = {'subok': keyword_20761, 'casting': keyword_20759, 'out': keyword_20757}
    # Getting the type of 'um' (line 93)
    um_20752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'um', False)
    # Obtaining the member 'true_divide' of a type (line 93)
    true_divide_20753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), um_20752, 'true_divide')
    # Calling true_divide(args, kwargs) (line 93)
    true_divide_call_result_20763 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), true_divide_20753, *[arrmean_20754, rcount_20755], **kwargs_20762)
    
    # Assigning a type to the variable 'arrmean' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'arrmean', true_divide_call_result_20763)
    # SSA branch for the else part of an if statement (line 92)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 96):
    
    # Call to type(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'arrmean' (line 96)
    arrmean_20767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'arrmean', False)
    # Getting the type of 'rcount' (line 96)
    rcount_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'rcount', False)
    # Applying the binary operator 'div' (line 96)
    result_div_20769 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 37), 'div', arrmean_20767, rcount_20768)
    
    # Processing the call keyword arguments (line 96)
    kwargs_20770 = {}
    # Getting the type of 'arrmean' (line 96)
    arrmean_20764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'arrmean', False)
    # Obtaining the member 'dtype' of a type (line 96)
    dtype_20765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), arrmean_20764, 'dtype')
    # Obtaining the member 'type' of a type (line 96)
    type_20766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), dtype_20765, 'type')
    # Calling type(args, kwargs) (line 96)
    type_call_result_20771 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), type_20766, *[result_div_20769], **kwargs_20770)
    
    # Assigning a type to the variable 'arrmean' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'arrmean', type_call_result_20771)
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 101):
    
    # Call to asanyarray(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'arr' (line 101)
    arr_20773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'arr', False)
    # Getting the type of 'arrmean' (line 101)
    arrmean_20774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'arrmean', False)
    # Applying the binary operator '-' (line 101)
    result_sub_20775 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 19), '-', arr_20773, arrmean_20774)
    
    # Processing the call keyword arguments (line 101)
    kwargs_20776 = {}
    # Getting the type of 'asanyarray' (line 101)
    asanyarray_20772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 101)
    asanyarray_call_result_20777 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), asanyarray_20772, *[result_sub_20775], **kwargs_20776)
    
    # Assigning a type to the variable 'x' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'x', asanyarray_call_result_20777)
    
    
    # Call to issubclass(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'arr' (line 102)
    arr_20779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 102)
    dtype_20780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 18), arr_20779, 'dtype')
    # Obtaining the member 'type' of a type (line 102)
    type_20781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 18), dtype_20780, 'type')
    # Getting the type of 'nt' (line 102)
    nt_20782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'nt', False)
    # Obtaining the member 'complexfloating' of a type (line 102)
    complexfloating_20783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 34), nt_20782, 'complexfloating')
    # Processing the call keyword arguments (line 102)
    kwargs_20784 = {}
    # Getting the type of 'issubclass' (line 102)
    issubclass_20778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 102)
    issubclass_call_result_20785 = invoke(stypy.reporting.localization.Localization(__file__, 102, 7), issubclass_20778, *[type_20781, complexfloating_20783], **kwargs_20784)
    
    # Testing the type of an if condition (line 102)
    if_condition_20786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), issubclass_call_result_20785)
    # Assigning a type to the variable 'if_condition_20786' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_20786', if_condition_20786)
    # SSA begins for if statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 103):
    
    # Call to multiply(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'x' (line 103)
    x_20789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'x', False)
    
    # Call to conjugate(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'x' (line 103)
    x_20792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'x', False)
    # Processing the call keyword arguments (line 103)
    kwargs_20793 = {}
    # Getting the type of 'um' (line 103)
    um_20790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'um', False)
    # Obtaining the member 'conjugate' of a type (line 103)
    conjugate_20791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 27), um_20790, 'conjugate')
    # Calling conjugate(args, kwargs) (line 103)
    conjugate_call_result_20794 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), conjugate_20791, *[x_20792], **kwargs_20793)
    
    # Processing the call keyword arguments (line 103)
    # Getting the type of 'x' (line 103)
    x_20795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 48), 'x', False)
    keyword_20796 = x_20795
    kwargs_20797 = {'out': keyword_20796}
    # Getting the type of 'um' (line 103)
    um_20787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'um', False)
    # Obtaining the member 'multiply' of a type (line 103)
    multiply_20788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), um_20787, 'multiply')
    # Calling multiply(args, kwargs) (line 103)
    multiply_call_result_20798 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), multiply_20788, *[x_20789, conjugate_call_result_20794], **kwargs_20797)
    
    # Obtaining the member 'real' of a type (line 103)
    real_20799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), multiply_call_result_20798, 'real')
    # Assigning a type to the variable 'x' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'x', real_20799)
    # SSA branch for the else part of an if statement (line 102)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 105):
    
    # Call to multiply(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'x' (line 105)
    x_20802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'x', False)
    # Getting the type of 'x' (line 105)
    x_20803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'x', False)
    # Processing the call keyword arguments (line 105)
    # Getting the type of 'x' (line 105)
    x_20804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'x', False)
    keyword_20805 = x_20804
    kwargs_20806 = {'out': keyword_20805}
    # Getting the type of 'um' (line 105)
    um_20800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'um', False)
    # Obtaining the member 'multiply' of a type (line 105)
    multiply_20801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), um_20800, 'multiply')
    # Calling multiply(args, kwargs) (line 105)
    multiply_call_result_20807 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), multiply_20801, *[x_20802, x_20803], **kwargs_20806)
    
    # Assigning a type to the variable 'x' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'x', multiply_call_result_20807)
    # SSA join for if statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 106):
    
    # Call to umr_sum(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'x' (line 106)
    x_20809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'x', False)
    # Getting the type of 'axis' (line 106)
    axis_20810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'axis', False)
    # Getting the type of 'dtype' (line 106)
    dtype_20811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'dtype', False)
    # Getting the type of 'out' (line 106)
    out_20812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'out', False)
    # Getting the type of 'keepdims' (line 106)
    keepdims_20813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 39), 'keepdims', False)
    # Processing the call keyword arguments (line 106)
    kwargs_20814 = {}
    # Getting the type of 'umr_sum' (line 106)
    umr_sum_20808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 10), 'umr_sum', False)
    # Calling umr_sum(args, kwargs) (line 106)
    umr_sum_call_result_20815 = invoke(stypy.reporting.localization.Localization(__file__, 106, 10), umr_sum_20808, *[x_20809, axis_20810, dtype_20811, out_20812, keepdims_20813], **kwargs_20814)
    
    # Assigning a type to the variable 'ret' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'ret', umr_sum_call_result_20815)
    
    # Assigning a Call to a Name (line 109):
    
    # Call to max(...): (line 109)
    # Processing the call arguments (line 109)
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_20817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'rcount' (line 109)
    rcount_20818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'rcount', False)
    # Getting the type of 'ddof' (line 109)
    ddof_20819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'ddof', False)
    # Applying the binary operator '-' (line 109)
    result_sub_20820 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 18), '-', rcount_20818, ddof_20819)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 17), list_20817, result_sub_20820)
    # Adding element type (line 109)
    int_20821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 17), list_20817, int_20821)
    
    # Processing the call keyword arguments (line 109)
    kwargs_20822 = {}
    # Getting the type of 'max' (line 109)
    max_20816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'max', False)
    # Calling max(args, kwargs) (line 109)
    max_call_result_20823 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), max_20816, *[list_20817], **kwargs_20822)
    
    # Assigning a type to the variable 'rcount' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'rcount', max_call_result_20823)
    
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'ret' (line 112)
    ret_20825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'ret', False)
    # Getting the type of 'mu' (line 112)
    mu_20826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'mu', False)
    # Obtaining the member 'ndarray' of a type (line 112)
    ndarray_20827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 23), mu_20826, 'ndarray')
    # Processing the call keyword arguments (line 112)
    kwargs_20828 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_20824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_20829 = invoke(stypy.reporting.localization.Localization(__file__, 112, 7), isinstance_20824, *[ret_20825, ndarray_20827], **kwargs_20828)
    
    # Testing the type of an if condition (line 112)
    if_condition_20830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), isinstance_call_result_20829)
    # Assigning a type to the variable 'if_condition_20830' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_20830', if_condition_20830)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 113):
    
    # Call to true_divide(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'ret' (line 114)
    ret_20833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'ret', False)
    # Getting the type of 'rcount' (line 114)
    rcount_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'rcount', False)
    # Processing the call keyword arguments (line 113)
    # Getting the type of 'ret' (line 114)
    ret_20835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'ret', False)
    keyword_20836 = ret_20835
    str_20837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'str', 'unsafe')
    keyword_20838 = str_20837
    # Getting the type of 'False' (line 114)
    False_20839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 62), 'False', False)
    keyword_20840 = False_20839
    kwargs_20841 = {'subok': keyword_20840, 'casting': keyword_20838, 'out': keyword_20836}
    # Getting the type of 'um' (line 113)
    um_20831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'um', False)
    # Obtaining the member 'true_divide' of a type (line 113)
    true_divide_20832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 14), um_20831, 'true_divide')
    # Calling true_divide(args, kwargs) (line 113)
    true_divide_call_result_20842 = invoke(stypy.reporting.localization.Localization(__file__, 113, 14), true_divide_20832, *[ret_20833, rcount_20834], **kwargs_20841)
    
    # Assigning a type to the variable 'ret' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'ret', true_divide_call_result_20842)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 115)
    str_20843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'str', 'dtype')
    # Getting the type of 'ret' (line 115)
    ret_20844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'ret')
    
    (may_be_20845, more_types_in_union_20846) = may_provide_member(str_20843, ret_20844)

    if may_be_20845:

        if more_types_in_union_20846:
            # Runtime conditional SSA (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'ret' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'ret', remove_not_member_provider_from_union(ret_20844, 'dtype'))
        
        # Assigning a Call to a Name (line 116):
        
        # Call to type(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'ret' (line 116)
        ret_20850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'ret', False)
        # Getting the type of 'rcount' (line 116)
        rcount_20851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 35), 'rcount', False)
        # Applying the binary operator 'div' (line 116)
        result_div_20852 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 29), 'div', ret_20850, rcount_20851)
        
        # Processing the call keyword arguments (line 116)
        kwargs_20853 = {}
        # Getting the type of 'ret' (line 116)
        ret_20847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'ret', False)
        # Obtaining the member 'dtype' of a type (line 116)
        dtype_20848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 14), ret_20847, 'dtype')
        # Obtaining the member 'type' of a type (line 116)
        type_20849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 14), dtype_20848, 'type')
        # Calling type(args, kwargs) (line 116)
        type_call_result_20854 = invoke(stypy.reporting.localization.Localization(__file__, 116, 14), type_20849, *[result_div_20852], **kwargs_20853)
        
        # Assigning a type to the variable 'ret' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'ret', type_call_result_20854)

        if more_types_in_union_20846:
            # Runtime conditional SSA for else branch (line 115)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20845) or more_types_in_union_20846):
        # Assigning a type to the variable 'ret' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'ret', remove_member_provider_from_union(ret_20844, 'dtype'))
        
        # Assigning a BinOp to a Name (line 118):
        # Getting the type of 'ret' (line 118)
        ret_20855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ret')
        # Getting the type of 'rcount' (line 118)
        rcount_20856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'rcount')
        # Applying the binary operator 'div' (line 118)
        result_div_20857 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 14), 'div', ret_20855, rcount_20856)
        
        # Assigning a type to the variable 'ret' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'ret', result_div_20857)

        if (may_be_20845 and more_types_in_union_20846):
            # SSA join for if statement (line 115)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 120)
    ret_20858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type', ret_20858)
    
    # ################# End of '_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_var' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_20859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20859)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_var'
    return stypy_return_type_20859

# Assigning a type to the variable '_var' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), '_var', _var)

@norecursion
def _std(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 122)
    None_20860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'None')
    # Getting the type of 'None' (line 122)
    None_20861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'None')
    # Getting the type of 'None' (line 122)
    None_20862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'None')
    int_20863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 50), 'int')
    # Getting the type of 'False' (line 122)
    False_20864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 62), 'False')
    defaults = [None_20860, None_20861, None_20862, int_20863, False_20864]
    # Create a new context for function '_std'
    module_type_store = module_type_store.open_function_context('_std', 122, 0, False)
    
    # Passed parameters checking function
    _std.stypy_localization = localization
    _std.stypy_type_of_self = None
    _std.stypy_type_store = module_type_store
    _std.stypy_function_name = '_std'
    _std.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    _std.stypy_varargs_param_name = None
    _std.stypy_kwargs_param_name = None
    _std.stypy_call_defaults = defaults
    _std.stypy_call_varargs = varargs
    _std.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_std', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_std', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_std(...)' code ##################

    
    # Assigning a Call to a Name (line 123):
    
    # Call to _var(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'a' (line 123)
    a_20866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'a', False)
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'axis' (line 123)
    axis_20867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'axis', False)
    keyword_20868 = axis_20867
    # Getting the type of 'dtype' (line 123)
    dtype_20869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'dtype', False)
    keyword_20870 = dtype_20869
    # Getting the type of 'out' (line 123)
    out_20871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'out', False)
    keyword_20872 = out_20871
    # Getting the type of 'ddof' (line 123)
    ddof_20873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 56), 'ddof', False)
    keyword_20874 = ddof_20873
    # Getting the type of 'keepdims' (line 124)
    keepdims_20875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'keepdims', False)
    keyword_20876 = keepdims_20875
    kwargs_20877 = {'dtype': keyword_20870, 'out': keyword_20872, 'ddof': keyword_20874, 'keepdims': keyword_20876, 'axis': keyword_20868}
    # Getting the type of '_var' (line 123)
    _var_20865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), '_var', False)
    # Calling _var(args, kwargs) (line 123)
    _var_call_result_20878 = invoke(stypy.reporting.localization.Localization(__file__, 123, 10), _var_20865, *[a_20866], **kwargs_20877)
    
    # Assigning a type to the variable 'ret' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'ret', _var_call_result_20878)
    
    
    # Call to isinstance(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'ret' (line 126)
    ret_20880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'ret', False)
    # Getting the type of 'mu' (line 126)
    mu_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'mu', False)
    # Obtaining the member 'ndarray' of a type (line 126)
    ndarray_20882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), mu_20881, 'ndarray')
    # Processing the call keyword arguments (line 126)
    kwargs_20883 = {}
    # Getting the type of 'isinstance' (line 126)
    isinstance_20879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 126)
    isinstance_call_result_20884 = invoke(stypy.reporting.localization.Localization(__file__, 126, 7), isinstance_20879, *[ret_20880, ndarray_20882], **kwargs_20883)
    
    # Testing the type of an if condition (line 126)
    if_condition_20885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), isinstance_call_result_20884)
    # Assigning a type to the variable 'if_condition_20885' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_20885', if_condition_20885)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 127):
    
    # Call to sqrt(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'ret' (line 127)
    ret_20888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'ret', False)
    # Processing the call keyword arguments (line 127)
    # Getting the type of 'ret' (line 127)
    ret_20889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'ret', False)
    keyword_20890 = ret_20889
    kwargs_20891 = {'out': keyword_20890}
    # Getting the type of 'um' (line 127)
    um_20886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'um', False)
    # Obtaining the member 'sqrt' of a type (line 127)
    sqrt_20887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 14), um_20886, 'sqrt')
    # Calling sqrt(args, kwargs) (line 127)
    sqrt_call_result_20892 = invoke(stypy.reporting.localization.Localization(__file__, 127, 14), sqrt_20887, *[ret_20888], **kwargs_20891)
    
    # Assigning a type to the variable 'ret' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'ret', sqrt_call_result_20892)
    # SSA branch for the else part of an if statement (line 126)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 128)
    str_20893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'str', 'dtype')
    # Getting the type of 'ret' (line 128)
    ret_20894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'ret')
    
    (may_be_20895, more_types_in_union_20896) = may_provide_member(str_20893, ret_20894)

    if may_be_20895:

        if more_types_in_union_20896:
            # Runtime conditional SSA (line 128)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'ret' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 9), 'ret', remove_not_member_provider_from_union(ret_20894, 'dtype'))
        
        # Assigning a Call to a Name (line 129):
        
        # Call to type(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to sqrt(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'ret' (line 129)
        ret_20902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'ret', False)
        # Processing the call keyword arguments (line 129)
        kwargs_20903 = {}
        # Getting the type of 'um' (line 129)
        um_20900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 29), 'um', False)
        # Obtaining the member 'sqrt' of a type (line 129)
        sqrt_20901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 29), um_20900, 'sqrt')
        # Calling sqrt(args, kwargs) (line 129)
        sqrt_call_result_20904 = invoke(stypy.reporting.localization.Localization(__file__, 129, 29), sqrt_20901, *[ret_20902], **kwargs_20903)
        
        # Processing the call keyword arguments (line 129)
        kwargs_20905 = {}
        # Getting the type of 'ret' (line 129)
        ret_20897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 14), 'ret', False)
        # Obtaining the member 'dtype' of a type (line 129)
        dtype_20898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 14), ret_20897, 'dtype')
        # Obtaining the member 'type' of a type (line 129)
        type_20899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 14), dtype_20898, 'type')
        # Calling type(args, kwargs) (line 129)
        type_call_result_20906 = invoke(stypy.reporting.localization.Localization(__file__, 129, 14), type_20899, *[sqrt_call_result_20904], **kwargs_20905)
        
        # Assigning a type to the variable 'ret' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'ret', type_call_result_20906)

        if more_types_in_union_20896:
            # Runtime conditional SSA for else branch (line 128)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20895) or more_types_in_union_20896):
        # Assigning a type to the variable 'ret' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 9), 'ret', remove_member_provider_from_union(ret_20894, 'dtype'))
        
        # Assigning a Call to a Name (line 131):
        
        # Call to sqrt(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'ret' (line 131)
        ret_20909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'ret', False)
        # Processing the call keyword arguments (line 131)
        kwargs_20910 = {}
        # Getting the type of 'um' (line 131)
        um_20907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 14), 'um', False)
        # Obtaining the member 'sqrt' of a type (line 131)
        sqrt_20908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 14), um_20907, 'sqrt')
        # Calling sqrt(args, kwargs) (line 131)
        sqrt_call_result_20911 = invoke(stypy.reporting.localization.Localization(__file__, 131, 14), sqrt_20908, *[ret_20909], **kwargs_20910)
        
        # Assigning a type to the variable 'ret' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'ret', sqrt_call_result_20911)

        if (may_be_20895 and more_types_in_union_20896):
            # SSA join for if statement (line 128)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 133)
    ret_20912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', ret_20912)
    
    # ################# End of '_std(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_std' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_20913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20913)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_std'
    return stypy_return_type_20913

# Assigning a type to the variable '_std' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), '_std', _std)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
