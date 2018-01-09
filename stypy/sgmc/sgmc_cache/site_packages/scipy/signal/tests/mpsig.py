
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Some signal functions implemented using mpmath.
3: '''
4: 
5: from __future__ import division
6: 
7: try:
8:     import mpmath
9: except ImportError:
10:     mpmath = None
11: 
12: 
13: def _prod(seq):
14:     '''Returns the product of the elements in the sequence `seq`.'''
15:     p = 1
16:     for elem in seq:
17:         p *= elem
18:     return p
19: 
20: 
21: def _relative_degree(z, p):
22:     '''
23:     Return relative degree of transfer function from zeros and poles.
24: 
25:     This is simply len(p) - len(z), which must be nonnegative.
26:     A ValueError is raised if len(p) < len(z).
27:     '''
28:     degree = len(p) - len(z)
29:     if degree < 0:
30:         raise ValueError("Improper transfer function. "
31:                          "Must have at least as many poles as zeros.")
32:     return degree
33: 
34: 
35: def _zpkbilinear(z, p, k, fs):
36:     '''Bilinear transformation to convert a filter from analog to digital.'''
37: 
38:     degree = _relative_degree(z, p)
39: 
40:     fs2 = 2*fs
41: 
42:     # Bilinear transform the poles and zeros
43:     z_z = [(fs2 + z1) / (fs2 - z1) for z1 in z]
44:     p_z = [(fs2 + p1) / (fs2 - p1) for p1 in p]
45: 
46:     # Any zeros that were at infinity get moved to the Nyquist frequency
47:     z_z.extend([-1] * degree)
48: 
49:     # Compensate for gain change
50:     numer = _prod(fs2 - z1 for z1 in z)
51:     denom = _prod(fs2 - p1 for p1 in p)
52:     k_z = k * numer / denom
53: 
54:     return z_z, p_z, k_z.real
55: 
56: 
57: def _zpklp2lp(z, p, k, wo=1):
58:     '''Transform a lowpass filter to a different cutoff frequency.'''
59: 
60:     degree = _relative_degree(z, p)
61: 
62:     # Scale all points radially from origin to shift cutoff frequency
63:     z_lp = [wo * z1 for z1 in z]
64:     p_lp = [wo * p1 for p1 in p]
65: 
66:     # Each shifted pole decreases gain by wo, each shifted zero increases it.
67:     # Cancel out the net change to keep overall gain the same
68:     k_lp = k * wo**degree
69: 
70:     return z_lp, p_lp, k_lp
71: 
72: 
73: def _butter_analog_poles(n):
74:     '''
75:     Poles of an analog Butterworth lowpass filter.
76: 
77:     This is the same calculation as scipy.signal.buttap(n) or
78:     scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,
79:     and only the poles are returned.
80:     '''
81:     poles = []
82:     for k in range(-n+1, n, 2):
83:         poles.append(-mpmath.exp(1j*mpmath.pi*k/(2*n)))
84:     return poles
85: 
86: 
87: def butter_lp(n, Wn):
88:     '''
89:     Lowpass Butterworth digital filter design.
90: 
91:     This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),
92:     but it uses mpmath, and the results are returned in lists instead of numpy
93:     arrays.
94:     '''
95:     zeros = []
96:     poles = _butter_analog_poles(n)
97:     k = 1
98:     fs = 2
99:     warped = 2 * fs * mpmath.tan(mpmath.pi * Wn / fs)
100:     z, p, k = _zpklp2lp(zeros, poles, k, wo=warped)
101:     z, p, k = _zpkbilinear(z, p, k, fs=fs)
102:     return z, p, k
103: 
104: 
105: def zpkfreqz(z, p, k, worN=None):
106:     '''
107:     Frequency response of a filter in zpk format, using mpmath.
108: 
109:     This is the same calculation as scipy.signal.freqz, but the input is in
110:     zpk format, the calculation is performed using mpath, and the results are
111:     returned in lists instead of numpy arrays.
112:     '''
113:     if worN is None or isinstance(worN, int):
114:         N = worN or 512
115:         ws = [mpmath.pi * mpmath.mpf(j) / N for j in range(N)]
116:     else:
117:         ws = worN
118: 
119:     h = []
120:     for wk in ws:
121:         zm1 = mpmath.exp(1j * wk)
122:         numer = _prod([zm1 - t for t in z])
123:         denom = _prod([zm1 - t for t in p])
124:         hk = k * numer / denom
125:         h.append(hk)
126:     return ws, h
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_288899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nSome signal functions implemented using mpmath.\n')


# SSA begins for try-except statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'import mpmath' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_288900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'mpmath')

if (type(import_288900) is not StypyTypeError):

    if (import_288900 != 'pyd_module'):
        __import__(import_288900)
        sys_modules_288901 = sys.modules[import_288900]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'mpmath', sys_modules_288901.module_type_store, module_type_store)
    else:
        import mpmath

        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'mpmath', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'mpmath', import_288900)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# SSA branch for the except part of a try statement (line 7)
# SSA branch for the except 'ImportError' branch of a try statement (line 7)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 10):

# Assigning a Name to a Name (line 10):
# Getting the type of 'None' (line 10)
None_288902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'None')
# Assigning a type to the variable 'mpmath' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'mpmath', None_288902)
# SSA join for try-except statement (line 7)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_prod'
    module_type_store = module_type_store.open_function_context('_prod', 13, 0, False)
    
    # Passed parameters checking function
    _prod.stypy_localization = localization
    _prod.stypy_type_of_self = None
    _prod.stypy_type_store = module_type_store
    _prod.stypy_function_name = '_prod'
    _prod.stypy_param_names_list = ['seq']
    _prod.stypy_varargs_param_name = None
    _prod.stypy_kwargs_param_name = None
    _prod.stypy_call_defaults = defaults
    _prod.stypy_call_varargs = varargs
    _prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prod', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prod', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prod(...)' code ##################

    str_288903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Returns the product of the elements in the sequence `seq`.')
    
    # Assigning a Num to a Name (line 15):
    
    # Assigning a Num to a Name (line 15):
    int_288904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
    # Assigning a type to the variable 'p' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'p', int_288904)
    
    # Getting the type of 'seq' (line 16)
    seq_288905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'seq')
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), seq_288905)
    # Getting the type of the for loop variable (line 16)
    for_loop_var_288906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), seq_288905)
    # Assigning a type to the variable 'elem' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'elem', for_loop_var_288906)
    # SSA begins for a for statement (line 16)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'p' (line 17)
    p_288907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'p')
    # Getting the type of 'elem' (line 17)
    elem_288908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'elem')
    # Applying the binary operator '*=' (line 17)
    result_imul_288909 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 8), '*=', p_288907, elem_288908)
    # Assigning a type to the variable 'p' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'p', result_imul_288909)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'p' (line 18)
    p_288910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'p')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', p_288910)
    
    # ################# End of '_prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prod' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_288911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288911)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prod'
    return stypy_return_type_288911

# Assigning a type to the variable '_prod' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_prod', _prod)

@norecursion
def _relative_degree(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_relative_degree'
    module_type_store = module_type_store.open_function_context('_relative_degree', 21, 0, False)
    
    # Passed parameters checking function
    _relative_degree.stypy_localization = localization
    _relative_degree.stypy_type_of_self = None
    _relative_degree.stypy_type_store = module_type_store
    _relative_degree.stypy_function_name = '_relative_degree'
    _relative_degree.stypy_param_names_list = ['z', 'p']
    _relative_degree.stypy_varargs_param_name = None
    _relative_degree.stypy_kwargs_param_name = None
    _relative_degree.stypy_call_defaults = defaults
    _relative_degree.stypy_call_varargs = varargs
    _relative_degree.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_relative_degree', ['z', 'p'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_relative_degree', localization, ['z', 'p'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_relative_degree(...)' code ##################

    str_288912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n    Return relative degree of transfer function from zeros and poles.\n\n    This is simply len(p) - len(z), which must be nonnegative.\n    A ValueError is raised if len(p) < len(z).\n    ')
    
    # Assigning a BinOp to a Name (line 28):
    
    # Assigning a BinOp to a Name (line 28):
    
    # Call to len(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'p' (line 28)
    p_288914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'p', False)
    # Processing the call keyword arguments (line 28)
    kwargs_288915 = {}
    # Getting the type of 'len' (line 28)
    len_288913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'len', False)
    # Calling len(args, kwargs) (line 28)
    len_call_result_288916 = invoke(stypy.reporting.localization.Localization(__file__, 28, 13), len_288913, *[p_288914], **kwargs_288915)
    
    
    # Call to len(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'z' (line 28)
    z_288918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'z', False)
    # Processing the call keyword arguments (line 28)
    kwargs_288919 = {}
    # Getting the type of 'len' (line 28)
    len_288917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'len', False)
    # Calling len(args, kwargs) (line 28)
    len_call_result_288920 = invoke(stypy.reporting.localization.Localization(__file__, 28, 22), len_288917, *[z_288918], **kwargs_288919)
    
    # Applying the binary operator '-' (line 28)
    result_sub_288921 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 13), '-', len_call_result_288916, len_call_result_288920)
    
    # Assigning a type to the variable 'degree' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'degree', result_sub_288921)
    
    
    # Getting the type of 'degree' (line 29)
    degree_288922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'degree')
    int_288923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'int')
    # Applying the binary operator '<' (line 29)
    result_lt_288924 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), '<', degree_288922, int_288923)
    
    # Testing the type of an if condition (line 29)
    if_condition_288925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_lt_288924)
    # Assigning a type to the variable 'if_condition_288925' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_288925', if_condition_288925)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 30)
    # Processing the call arguments (line 30)
    str_288927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', 'Improper transfer function. Must have at least as many poles as zeros.')
    # Processing the call keyword arguments (line 30)
    kwargs_288928 = {}
    # Getting the type of 'ValueError' (line 30)
    ValueError_288926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 30)
    ValueError_call_result_288929 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), ValueError_288926, *[str_288927], **kwargs_288928)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 30, 8), ValueError_call_result_288929, 'raise parameter', BaseException)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'degree' (line 32)
    degree_288930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'degree')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', degree_288930)
    
    # ################# End of '_relative_degree(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_relative_degree' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_288931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_relative_degree'
    return stypy_return_type_288931

# Assigning a type to the variable '_relative_degree' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '_relative_degree', _relative_degree)

@norecursion
def _zpkbilinear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_zpkbilinear'
    module_type_store = module_type_store.open_function_context('_zpkbilinear', 35, 0, False)
    
    # Passed parameters checking function
    _zpkbilinear.stypy_localization = localization
    _zpkbilinear.stypy_type_of_self = None
    _zpkbilinear.stypy_type_store = module_type_store
    _zpkbilinear.stypy_function_name = '_zpkbilinear'
    _zpkbilinear.stypy_param_names_list = ['z', 'p', 'k', 'fs']
    _zpkbilinear.stypy_varargs_param_name = None
    _zpkbilinear.stypy_kwargs_param_name = None
    _zpkbilinear.stypy_call_defaults = defaults
    _zpkbilinear.stypy_call_varargs = varargs
    _zpkbilinear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zpkbilinear', ['z', 'p', 'k', 'fs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zpkbilinear', localization, ['z', 'p', 'k', 'fs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zpkbilinear(...)' code ##################

    str_288932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'Bilinear transformation to convert a filter from analog to digital.')
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to _relative_degree(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'z' (line 38)
    z_288934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'z', False)
    # Getting the type of 'p' (line 38)
    p_288935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'p', False)
    # Processing the call keyword arguments (line 38)
    kwargs_288936 = {}
    # Getting the type of '_relative_degree' (line 38)
    _relative_degree_288933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), '_relative_degree', False)
    # Calling _relative_degree(args, kwargs) (line 38)
    _relative_degree_call_result_288937 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), _relative_degree_288933, *[z_288934, p_288935], **kwargs_288936)
    
    # Assigning a type to the variable 'degree' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'degree', _relative_degree_call_result_288937)
    
    # Assigning a BinOp to a Name (line 40):
    
    # Assigning a BinOp to a Name (line 40):
    int_288938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'int')
    # Getting the type of 'fs' (line 40)
    fs_288939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'fs')
    # Applying the binary operator '*' (line 40)
    result_mul_288940 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 10), '*', int_288938, fs_288939)
    
    # Assigning a type to the variable 'fs2' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'fs2', result_mul_288940)
    
    # Assigning a ListComp to a Name (line 43):
    
    # Assigning a ListComp to a Name (line 43):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'z' (line 43)
    z_288948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 45), 'z')
    comprehension_288949 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 11), z_288948)
    # Assigning a type to the variable 'z1' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'z1', comprehension_288949)
    # Getting the type of 'fs2' (line 43)
    fs2_288941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'fs2')
    # Getting the type of 'z1' (line 43)
    z1_288942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'z1')
    # Applying the binary operator '+' (line 43)
    result_add_288943 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), '+', fs2_288941, z1_288942)
    
    # Getting the type of 'fs2' (line 43)
    fs2_288944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'fs2')
    # Getting the type of 'z1' (line 43)
    z1_288945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'z1')
    # Applying the binary operator '-' (line 43)
    result_sub_288946 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 25), '-', fs2_288944, z1_288945)
    
    # Applying the binary operator 'div' (line 43)
    result_div_288947 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'div', result_add_288943, result_sub_288946)
    
    list_288950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 11), list_288950, result_div_288947)
    # Assigning a type to the variable 'z_z' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'z_z', list_288950)
    
    # Assigning a ListComp to a Name (line 44):
    
    # Assigning a ListComp to a Name (line 44):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'p' (line 44)
    p_288958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'p')
    comprehension_288959 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 11), p_288958)
    # Assigning a type to the variable 'p1' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'p1', comprehension_288959)
    # Getting the type of 'fs2' (line 44)
    fs2_288951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'fs2')
    # Getting the type of 'p1' (line 44)
    p1_288952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'p1')
    # Applying the binary operator '+' (line 44)
    result_add_288953 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+', fs2_288951, p1_288952)
    
    # Getting the type of 'fs2' (line 44)
    fs2_288954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'fs2')
    # Getting the type of 'p1' (line 44)
    p1_288955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'p1')
    # Applying the binary operator '-' (line 44)
    result_sub_288956 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 25), '-', fs2_288954, p1_288955)
    
    # Applying the binary operator 'div' (line 44)
    result_div_288957 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'div', result_add_288953, result_sub_288956)
    
    list_288960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 11), list_288960, result_div_288957)
    # Assigning a type to the variable 'p_z' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'p_z', list_288960)
    
    # Call to extend(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_288963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    int_288964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 15), list_288963, int_288964)
    
    # Getting the type of 'degree' (line 47)
    degree_288965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'degree', False)
    # Applying the binary operator '*' (line 47)
    result_mul_288966 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 15), '*', list_288963, degree_288965)
    
    # Processing the call keyword arguments (line 47)
    kwargs_288967 = {}
    # Getting the type of 'z_z' (line 47)
    z_z_288961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'z_z', False)
    # Obtaining the member 'extend' of a type (line 47)
    extend_288962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), z_z_288961, 'extend')
    # Calling extend(args, kwargs) (line 47)
    extend_call_result_288968 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), extend_288962, *[result_mul_288966], **kwargs_288967)
    
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to _prod(...): (line 50)
    # Processing the call arguments (line 50)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 50, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'z' (line 50)
    z_288973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 37), 'z', False)
    comprehension_288974 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), z_288973)
    # Assigning a type to the variable 'z1' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'z1', comprehension_288974)
    # Getting the type of 'fs2' (line 50)
    fs2_288970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'fs2', False)
    # Getting the type of 'z1' (line 50)
    z1_288971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'z1', False)
    # Applying the binary operator '-' (line 50)
    result_sub_288972 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 18), '-', fs2_288970, z1_288971)
    
    list_288975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), list_288975, result_sub_288972)
    # Processing the call keyword arguments (line 50)
    kwargs_288976 = {}
    # Getting the type of '_prod' (line 50)
    _prod_288969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), '_prod', False)
    # Calling _prod(args, kwargs) (line 50)
    _prod_call_result_288977 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), _prod_288969, *[list_288975], **kwargs_288976)
    
    # Assigning a type to the variable 'numer' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'numer', _prod_call_result_288977)
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to _prod(...): (line 51)
    # Processing the call arguments (line 51)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 51, 18, True)
    # Calculating comprehension expression
    # Getting the type of 'p' (line 51)
    p_288982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'p', False)
    comprehension_288983 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 18), p_288982)
    # Assigning a type to the variable 'p1' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'p1', comprehension_288983)
    # Getting the type of 'fs2' (line 51)
    fs2_288979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'fs2', False)
    # Getting the type of 'p1' (line 51)
    p1_288980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'p1', False)
    # Applying the binary operator '-' (line 51)
    result_sub_288981 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 18), '-', fs2_288979, p1_288980)
    
    list_288984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 18), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 18), list_288984, result_sub_288981)
    # Processing the call keyword arguments (line 51)
    kwargs_288985 = {}
    # Getting the type of '_prod' (line 51)
    _prod_288978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), '_prod', False)
    # Calling _prod(args, kwargs) (line 51)
    _prod_call_result_288986 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), _prod_288978, *[list_288984], **kwargs_288985)
    
    # Assigning a type to the variable 'denom' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'denom', _prod_call_result_288986)
    
    # Assigning a BinOp to a Name (line 52):
    
    # Assigning a BinOp to a Name (line 52):
    # Getting the type of 'k' (line 52)
    k_288987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'k')
    # Getting the type of 'numer' (line 52)
    numer_288988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'numer')
    # Applying the binary operator '*' (line 52)
    result_mul_288989 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 10), '*', k_288987, numer_288988)
    
    # Getting the type of 'denom' (line 52)
    denom_288990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'denom')
    # Applying the binary operator 'div' (line 52)
    result_div_288991 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), 'div', result_mul_288989, denom_288990)
    
    # Assigning a type to the variable 'k_z' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'k_z', result_div_288991)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_288992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'z_z' (line 54)
    z_z_288993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'z_z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 11), tuple_288992, z_z_288993)
    # Adding element type (line 54)
    # Getting the type of 'p_z' (line 54)
    p_z_288994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'p_z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 11), tuple_288992, p_z_288994)
    # Adding element type (line 54)
    # Getting the type of 'k_z' (line 54)
    k_z_288995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'k_z')
    # Obtaining the member 'real' of a type (line 54)
    real_288996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), k_z_288995, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 11), tuple_288992, real_288996)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', tuple_288992)
    
    # ################# End of '_zpkbilinear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zpkbilinear' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_288997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288997)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zpkbilinear'
    return stypy_return_type_288997

# Assigning a type to the variable '_zpkbilinear' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '_zpkbilinear', _zpkbilinear)

@norecursion
def _zpklp2lp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_288998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'int')
    defaults = [int_288998]
    # Create a new context for function '_zpklp2lp'
    module_type_store = module_type_store.open_function_context('_zpklp2lp', 57, 0, False)
    
    # Passed parameters checking function
    _zpklp2lp.stypy_localization = localization
    _zpklp2lp.stypy_type_of_self = None
    _zpklp2lp.stypy_type_store = module_type_store
    _zpklp2lp.stypy_function_name = '_zpklp2lp'
    _zpklp2lp.stypy_param_names_list = ['z', 'p', 'k', 'wo']
    _zpklp2lp.stypy_varargs_param_name = None
    _zpklp2lp.stypy_kwargs_param_name = None
    _zpklp2lp.stypy_call_defaults = defaults
    _zpklp2lp.stypy_call_varargs = varargs
    _zpklp2lp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_zpklp2lp', ['z', 'p', 'k', 'wo'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_zpklp2lp', localization, ['z', 'p', 'k', 'wo'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_zpklp2lp(...)' code ##################

    str_288999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'str', 'Transform a lowpass filter to a different cutoff frequency.')
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to _relative_degree(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'z' (line 60)
    z_289001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 30), 'z', False)
    # Getting the type of 'p' (line 60)
    p_289002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'p', False)
    # Processing the call keyword arguments (line 60)
    kwargs_289003 = {}
    # Getting the type of '_relative_degree' (line 60)
    _relative_degree_289000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), '_relative_degree', False)
    # Calling _relative_degree(args, kwargs) (line 60)
    _relative_degree_call_result_289004 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), _relative_degree_289000, *[z_289001, p_289002], **kwargs_289003)
    
    # Assigning a type to the variable 'degree' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'degree', _relative_degree_call_result_289004)
    
    # Assigning a ListComp to a Name (line 63):
    
    # Assigning a ListComp to a Name (line 63):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'z' (line 63)
    z_289008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'z')
    comprehension_289009 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 12), z_289008)
    # Assigning a type to the variable 'z1' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'z1', comprehension_289009)
    # Getting the type of 'wo' (line 63)
    wo_289005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'wo')
    # Getting the type of 'z1' (line 63)
    z1_289006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'z1')
    # Applying the binary operator '*' (line 63)
    result_mul_289007 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 12), '*', wo_289005, z1_289006)
    
    list_289010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 12), list_289010, result_mul_289007)
    # Assigning a type to the variable 'z_lp' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'z_lp', list_289010)
    
    # Assigning a ListComp to a Name (line 64):
    
    # Assigning a ListComp to a Name (line 64):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'p' (line 64)
    p_289014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'p')
    comprehension_289015 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 12), p_289014)
    # Assigning a type to the variable 'p1' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'p1', comprehension_289015)
    # Getting the type of 'wo' (line 64)
    wo_289011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'wo')
    # Getting the type of 'p1' (line 64)
    p1_289012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'p1')
    # Applying the binary operator '*' (line 64)
    result_mul_289013 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '*', wo_289011, p1_289012)
    
    list_289016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 12), list_289016, result_mul_289013)
    # Assigning a type to the variable 'p_lp' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'p_lp', list_289016)
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    # Getting the type of 'k' (line 68)
    k_289017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'k')
    # Getting the type of 'wo' (line 68)
    wo_289018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'wo')
    # Getting the type of 'degree' (line 68)
    degree_289019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'degree')
    # Applying the binary operator '**' (line 68)
    result_pow_289020 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '**', wo_289018, degree_289019)
    
    # Applying the binary operator '*' (line 68)
    result_mul_289021 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 11), '*', k_289017, result_pow_289020)
    
    # Assigning a type to the variable 'k_lp' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'k_lp', result_mul_289021)
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_289022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'z_lp' (line 70)
    z_lp_289023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'z_lp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_289022, z_lp_289023)
    # Adding element type (line 70)
    # Getting the type of 'p_lp' (line 70)
    p_lp_289024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'p_lp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_289022, p_lp_289024)
    # Adding element type (line 70)
    # Getting the type of 'k_lp' (line 70)
    k_lp_289025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'k_lp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_289022, k_lp_289025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', tuple_289022)
    
    # ################# End of '_zpklp2lp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_zpklp2lp' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_289026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_zpklp2lp'
    return stypy_return_type_289026

# Assigning a type to the variable '_zpklp2lp' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_zpklp2lp', _zpklp2lp)

@norecursion
def _butter_analog_poles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_butter_analog_poles'
    module_type_store = module_type_store.open_function_context('_butter_analog_poles', 73, 0, False)
    
    # Passed parameters checking function
    _butter_analog_poles.stypy_localization = localization
    _butter_analog_poles.stypy_type_of_self = None
    _butter_analog_poles.stypy_type_store = module_type_store
    _butter_analog_poles.stypy_function_name = '_butter_analog_poles'
    _butter_analog_poles.stypy_param_names_list = ['n']
    _butter_analog_poles.stypy_varargs_param_name = None
    _butter_analog_poles.stypy_kwargs_param_name = None
    _butter_analog_poles.stypy_call_defaults = defaults
    _butter_analog_poles.stypy_call_varargs = varargs
    _butter_analog_poles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_butter_analog_poles', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_butter_analog_poles', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_butter_analog_poles(...)' code ##################

    str_289027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, (-1)), 'str', "\n    Poles of an analog Butterworth lowpass filter.\n\n    This is the same calculation as scipy.signal.buttap(n) or\n    scipy.signal.butter(n, 1, analog=True, output='zpk'), but mpmath is used,\n    and only the poles are returned.\n    ")
    
    # Assigning a List to a Name (line 81):
    
    # Assigning a List to a Name (line 81):
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_289028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    
    # Assigning a type to the variable 'poles' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'poles', list_289028)
    
    
    # Call to range(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Getting the type of 'n' (line 82)
    n_289030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'n', False)
    # Applying the 'usub' unary operator (line 82)
    result___neg___289031 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), 'usub', n_289030)
    
    int_289032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 22), 'int')
    # Applying the binary operator '+' (line 82)
    result_add_289033 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '+', result___neg___289031, int_289032)
    
    # Getting the type of 'n' (line 82)
    n_289034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'n', False)
    int_289035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 28), 'int')
    # Processing the call keyword arguments (line 82)
    kwargs_289036 = {}
    # Getting the type of 'range' (line 82)
    range_289029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'range', False)
    # Calling range(args, kwargs) (line 82)
    range_call_result_289037 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), range_289029, *[result_add_289033, n_289034, int_289035], **kwargs_289036)
    
    # Testing the type of a for loop iterable (line 82)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 4), range_call_result_289037)
    # Getting the type of the for loop variable (line 82)
    for_loop_var_289038 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 4), range_call_result_289037)
    # Assigning a type to the variable 'k' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'k', for_loop_var_289038)
    # SSA begins for a for statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 83)
    # Processing the call arguments (line 83)
    
    
    # Call to exp(...): (line 83)
    # Processing the call arguments (line 83)
    complex_289043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'complex')
    # Getting the type of 'mpmath' (line 83)
    mpmath_289044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'mpmath', False)
    # Obtaining the member 'pi' of a type (line 83)
    pi_289045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 36), mpmath_289044, 'pi')
    # Applying the binary operator '*' (line 83)
    result_mul_289046 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 33), '*', complex_289043, pi_289045)
    
    # Getting the type of 'k' (line 83)
    k_289047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'k', False)
    # Applying the binary operator '*' (line 83)
    result_mul_289048 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 45), '*', result_mul_289046, k_289047)
    
    int_289049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 49), 'int')
    # Getting the type of 'n' (line 83)
    n_289050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'n', False)
    # Applying the binary operator '*' (line 83)
    result_mul_289051 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 49), '*', int_289049, n_289050)
    
    # Applying the binary operator 'div' (line 83)
    result_div_289052 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 47), 'div', result_mul_289048, result_mul_289051)
    
    # Processing the call keyword arguments (line 83)
    kwargs_289053 = {}
    # Getting the type of 'mpmath' (line 83)
    mpmath_289041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'mpmath', False)
    # Obtaining the member 'exp' of a type (line 83)
    exp_289042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), mpmath_289041, 'exp')
    # Calling exp(args, kwargs) (line 83)
    exp_call_result_289054 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), exp_289042, *[result_div_289052], **kwargs_289053)
    
    # Applying the 'usub' unary operator (line 83)
    result___neg___289055 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 21), 'usub', exp_call_result_289054)
    
    # Processing the call keyword arguments (line 83)
    kwargs_289056 = {}
    # Getting the type of 'poles' (line 83)
    poles_289039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'poles', False)
    # Obtaining the member 'append' of a type (line 83)
    append_289040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), poles_289039, 'append')
    # Calling append(args, kwargs) (line 83)
    append_call_result_289057 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), append_289040, *[result___neg___289055], **kwargs_289056)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'poles' (line 84)
    poles_289058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'poles')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', poles_289058)
    
    # ################# End of '_butter_analog_poles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_butter_analog_poles' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_289059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289059)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_butter_analog_poles'
    return stypy_return_type_289059

# Assigning a type to the variable '_butter_analog_poles' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_butter_analog_poles', _butter_analog_poles)

@norecursion
def butter_lp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'butter_lp'
    module_type_store = module_type_store.open_function_context('butter_lp', 87, 0, False)
    
    # Passed parameters checking function
    butter_lp.stypy_localization = localization
    butter_lp.stypy_type_of_self = None
    butter_lp.stypy_type_store = module_type_store
    butter_lp.stypy_function_name = 'butter_lp'
    butter_lp.stypy_param_names_list = ['n', 'Wn']
    butter_lp.stypy_varargs_param_name = None
    butter_lp.stypy_kwargs_param_name = None
    butter_lp.stypy_call_defaults = defaults
    butter_lp.stypy_call_varargs = varargs
    butter_lp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'butter_lp', ['n', 'Wn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'butter_lp', localization, ['n', 'Wn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'butter_lp(...)' code ##################

    str_289060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', "\n    Lowpass Butterworth digital filter design.\n\n    This computes the same result as scipy.signal.butter(n, Wn, output='zpk'),\n    but it uses mpmath, and the results are returned in lists instead of numpy\n    arrays.\n    ")
    
    # Assigning a List to a Name (line 95):
    
    # Assigning a List to a Name (line 95):
    
    # Obtaining an instance of the builtin type 'list' (line 95)
    list_289061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 95)
    
    # Assigning a type to the variable 'zeros' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'zeros', list_289061)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to _butter_analog_poles(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'n' (line 96)
    n_289063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'n', False)
    # Processing the call keyword arguments (line 96)
    kwargs_289064 = {}
    # Getting the type of '_butter_analog_poles' (line 96)
    _butter_analog_poles_289062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), '_butter_analog_poles', False)
    # Calling _butter_analog_poles(args, kwargs) (line 96)
    _butter_analog_poles_call_result_289065 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), _butter_analog_poles_289062, *[n_289063], **kwargs_289064)
    
    # Assigning a type to the variable 'poles' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'poles', _butter_analog_poles_call_result_289065)
    
    # Assigning a Num to a Name (line 97):
    
    # Assigning a Num to a Name (line 97):
    int_289066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'int')
    # Assigning a type to the variable 'k' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'k', int_289066)
    
    # Assigning a Num to a Name (line 98):
    
    # Assigning a Num to a Name (line 98):
    int_289067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 9), 'int')
    # Assigning a type to the variable 'fs' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'fs', int_289067)
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    int_289068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 13), 'int')
    # Getting the type of 'fs' (line 99)
    fs_289069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'fs')
    # Applying the binary operator '*' (line 99)
    result_mul_289070 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 13), '*', int_289068, fs_289069)
    
    
    # Call to tan(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'mpmath' (line 99)
    mpmath_289073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'mpmath', False)
    # Obtaining the member 'pi' of a type (line 99)
    pi_289074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 33), mpmath_289073, 'pi')
    # Getting the type of 'Wn' (line 99)
    Wn_289075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'Wn', False)
    # Applying the binary operator '*' (line 99)
    result_mul_289076 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 33), '*', pi_289074, Wn_289075)
    
    # Getting the type of 'fs' (line 99)
    fs_289077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 50), 'fs', False)
    # Applying the binary operator 'div' (line 99)
    result_div_289078 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 48), 'div', result_mul_289076, fs_289077)
    
    # Processing the call keyword arguments (line 99)
    kwargs_289079 = {}
    # Getting the type of 'mpmath' (line 99)
    mpmath_289071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'mpmath', False)
    # Obtaining the member 'tan' of a type (line 99)
    tan_289072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), mpmath_289071, 'tan')
    # Calling tan(args, kwargs) (line 99)
    tan_call_result_289080 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), tan_289072, *[result_div_289078], **kwargs_289079)
    
    # Applying the binary operator '*' (line 99)
    result_mul_289081 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 20), '*', result_mul_289070, tan_call_result_289080)
    
    # Assigning a type to the variable 'warped' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'warped', result_mul_289081)
    
    # Assigning a Call to a Tuple (line 100):
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_289082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'int')
    
    # Call to _zpklp2lp(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'zeros' (line 100)
    zeros_289084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'zeros', False)
    # Getting the type of 'poles' (line 100)
    poles_289085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'poles', False)
    # Getting the type of 'k' (line 100)
    k_289086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'k', False)
    # Processing the call keyword arguments (line 100)
    # Getting the type of 'warped' (line 100)
    warped_289087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'warped', False)
    keyword_289088 = warped_289087
    kwargs_289089 = {'wo': keyword_289088}
    # Getting the type of '_zpklp2lp' (line 100)
    _zpklp2lp_289083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_zpklp2lp', False)
    # Calling _zpklp2lp(args, kwargs) (line 100)
    _zpklp2lp_call_result_289090 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), _zpklp2lp_289083, *[zeros_289084, poles_289085, k_289086], **kwargs_289089)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___289091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), _zpklp2lp_call_result_289090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_289092 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___289091, int_289082)
    
    # Assigning a type to the variable 'tuple_var_assignment_288893' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288893', subscript_call_result_289092)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_289093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'int')
    
    # Call to _zpklp2lp(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'zeros' (line 100)
    zeros_289095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'zeros', False)
    # Getting the type of 'poles' (line 100)
    poles_289096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'poles', False)
    # Getting the type of 'k' (line 100)
    k_289097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'k', False)
    # Processing the call keyword arguments (line 100)
    # Getting the type of 'warped' (line 100)
    warped_289098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'warped', False)
    keyword_289099 = warped_289098
    kwargs_289100 = {'wo': keyword_289099}
    # Getting the type of '_zpklp2lp' (line 100)
    _zpklp2lp_289094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_zpklp2lp', False)
    # Calling _zpklp2lp(args, kwargs) (line 100)
    _zpklp2lp_call_result_289101 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), _zpklp2lp_289094, *[zeros_289095, poles_289096, k_289097], **kwargs_289100)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___289102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), _zpklp2lp_call_result_289101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_289103 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___289102, int_289093)
    
    # Assigning a type to the variable 'tuple_var_assignment_288894' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288894', subscript_call_result_289103)
    
    # Assigning a Subscript to a Name (line 100):
    
    # Obtaining the type of the subscript
    int_289104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 4), 'int')
    
    # Call to _zpklp2lp(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'zeros' (line 100)
    zeros_289106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'zeros', False)
    # Getting the type of 'poles' (line 100)
    poles_289107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'poles', False)
    # Getting the type of 'k' (line 100)
    k_289108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'k', False)
    # Processing the call keyword arguments (line 100)
    # Getting the type of 'warped' (line 100)
    warped_289109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'warped', False)
    keyword_289110 = warped_289109
    kwargs_289111 = {'wo': keyword_289110}
    # Getting the type of '_zpklp2lp' (line 100)
    _zpklp2lp_289105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), '_zpklp2lp', False)
    # Calling _zpklp2lp(args, kwargs) (line 100)
    _zpklp2lp_call_result_289112 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), _zpklp2lp_289105, *[zeros_289106, poles_289107, k_289108], **kwargs_289111)
    
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___289113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), _zpklp2lp_call_result_289112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_289114 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), getitem___289113, int_289104)
    
    # Assigning a type to the variable 'tuple_var_assignment_288895' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288895', subscript_call_result_289114)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_288893' (line 100)
    tuple_var_assignment_288893_289115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288893')
    # Assigning a type to the variable 'z' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'z', tuple_var_assignment_288893_289115)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_288894' (line 100)
    tuple_var_assignment_288894_289116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288894')
    # Assigning a type to the variable 'p' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'p', tuple_var_assignment_288894_289116)
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'tuple_var_assignment_288895' (line 100)
    tuple_var_assignment_288895_289117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'tuple_var_assignment_288895')
    # Assigning a type to the variable 'k' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 10), 'k', tuple_var_assignment_288895_289117)
    
    # Assigning a Call to a Tuple (line 101):
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_289118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'int')
    
    # Call to _zpkbilinear(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'z' (line 101)
    z_289120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'z', False)
    # Getting the type of 'p' (line 101)
    p_289121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'p', False)
    # Getting the type of 'k' (line 101)
    k_289122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'k', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'fs' (line 101)
    fs_289123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'fs', False)
    keyword_289124 = fs_289123
    kwargs_289125 = {'fs': keyword_289124}
    # Getting the type of '_zpkbilinear' (line 101)
    _zpkbilinear_289119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), '_zpkbilinear', False)
    # Calling _zpkbilinear(args, kwargs) (line 101)
    _zpkbilinear_call_result_289126 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), _zpkbilinear_289119, *[z_289120, p_289121, k_289122], **kwargs_289125)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___289127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), _zpkbilinear_call_result_289126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_289128 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), getitem___289127, int_289118)
    
    # Assigning a type to the variable 'tuple_var_assignment_288896' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288896', subscript_call_result_289128)
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_289129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'int')
    
    # Call to _zpkbilinear(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'z' (line 101)
    z_289131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'z', False)
    # Getting the type of 'p' (line 101)
    p_289132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'p', False)
    # Getting the type of 'k' (line 101)
    k_289133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'k', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'fs' (line 101)
    fs_289134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'fs', False)
    keyword_289135 = fs_289134
    kwargs_289136 = {'fs': keyword_289135}
    # Getting the type of '_zpkbilinear' (line 101)
    _zpkbilinear_289130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), '_zpkbilinear', False)
    # Calling _zpkbilinear(args, kwargs) (line 101)
    _zpkbilinear_call_result_289137 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), _zpkbilinear_289130, *[z_289131, p_289132, k_289133], **kwargs_289136)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___289138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), _zpkbilinear_call_result_289137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_289139 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), getitem___289138, int_289129)
    
    # Assigning a type to the variable 'tuple_var_assignment_288897' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288897', subscript_call_result_289139)
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_289140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'int')
    
    # Call to _zpkbilinear(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'z' (line 101)
    z_289142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'z', False)
    # Getting the type of 'p' (line 101)
    p_289143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'p', False)
    # Getting the type of 'k' (line 101)
    k_289144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'k', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'fs' (line 101)
    fs_289145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'fs', False)
    keyword_289146 = fs_289145
    kwargs_289147 = {'fs': keyword_289146}
    # Getting the type of '_zpkbilinear' (line 101)
    _zpkbilinear_289141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), '_zpkbilinear', False)
    # Calling _zpkbilinear(args, kwargs) (line 101)
    _zpkbilinear_call_result_289148 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), _zpkbilinear_289141, *[z_289142, p_289143, k_289144], **kwargs_289147)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___289149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), _zpkbilinear_call_result_289148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_289150 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), getitem___289149, int_289140)
    
    # Assigning a type to the variable 'tuple_var_assignment_288898' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288898', subscript_call_result_289150)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_288896' (line 101)
    tuple_var_assignment_288896_289151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288896')
    # Assigning a type to the variable 'z' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'z', tuple_var_assignment_288896_289151)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_288897' (line 101)
    tuple_var_assignment_288897_289152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288897')
    # Assigning a type to the variable 'p' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'p', tuple_var_assignment_288897_289152)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_288898' (line 101)
    tuple_var_assignment_288898_289153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288898')
    # Assigning a type to the variable 'k' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'k', tuple_var_assignment_288898_289153)
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_289154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'z' (line 102)
    z_289155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), tuple_289154, z_289155)
    # Adding element type (line 102)
    # Getting the type of 'p' (line 102)
    p_289156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), tuple_289154, p_289156)
    # Adding element type (line 102)
    # Getting the type of 'k' (line 102)
    k_289157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 11), tuple_289154, k_289157)
    
    # Assigning a type to the variable 'stypy_return_type' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type', tuple_289154)
    
    # ################# End of 'butter_lp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'butter_lp' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_289158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289158)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'butter_lp'
    return stypy_return_type_289158

# Assigning a type to the variable 'butter_lp' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'butter_lp', butter_lp)

@norecursion
def zpkfreqz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 105)
    None_289159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'None')
    defaults = [None_289159]
    # Create a new context for function 'zpkfreqz'
    module_type_store = module_type_store.open_function_context('zpkfreqz', 105, 0, False)
    
    # Passed parameters checking function
    zpkfreqz.stypy_localization = localization
    zpkfreqz.stypy_type_of_self = None
    zpkfreqz.stypy_type_store = module_type_store
    zpkfreqz.stypy_function_name = 'zpkfreqz'
    zpkfreqz.stypy_param_names_list = ['z', 'p', 'k', 'worN']
    zpkfreqz.stypy_varargs_param_name = None
    zpkfreqz.stypy_kwargs_param_name = None
    zpkfreqz.stypy_call_defaults = defaults
    zpkfreqz.stypy_call_varargs = varargs
    zpkfreqz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zpkfreqz', ['z', 'p', 'k', 'worN'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zpkfreqz', localization, ['z', 'p', 'k', 'worN'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zpkfreqz(...)' code ##################

    str_289160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', '\n    Frequency response of a filter in zpk format, using mpmath.\n\n    This is the same calculation as scipy.signal.freqz, but the input is in\n    zpk format, the calculation is performed using mpath, and the results are\n    returned in lists instead of numpy arrays.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'worN' (line 113)
    worN_289161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'worN')
    # Getting the type of 'None' (line 113)
    None_289162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'None')
    # Applying the binary operator 'is' (line 113)
    result_is__289163 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), 'is', worN_289161, None_289162)
    
    
    # Call to isinstance(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'worN' (line 113)
    worN_289165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'worN', False)
    # Getting the type of 'int' (line 113)
    int_289166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'int', False)
    # Processing the call keyword arguments (line 113)
    kwargs_289167 = {}
    # Getting the type of 'isinstance' (line 113)
    isinstance_289164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 113)
    isinstance_call_result_289168 = invoke(stypy.reporting.localization.Localization(__file__, 113, 23), isinstance_289164, *[worN_289165, int_289166], **kwargs_289167)
    
    # Applying the binary operator 'or' (line 113)
    result_or_keyword_289169 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), 'or', result_is__289163, isinstance_call_result_289168)
    
    # Testing the type of an if condition (line 113)
    if_condition_289170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), result_or_keyword_289169)
    # Assigning a type to the variable 'if_condition_289170' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_289170', if_condition_289170)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 114):
    
    # Assigning a BoolOp to a Name (line 114):
    
    # Evaluating a boolean operation
    # Getting the type of 'worN' (line 114)
    worN_289171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'worN')
    int_289172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'int')
    # Applying the binary operator 'or' (line 114)
    result_or_keyword_289173 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 12), 'or', worN_289171, int_289172)
    
    # Assigning a type to the variable 'N' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'N', result_or_keyword_289173)
    
    # Assigning a ListComp to a Name (line 115):
    
    # Assigning a ListComp to a Name (line 115):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'N' (line 115)
    N_289185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'N', False)
    # Processing the call keyword arguments (line 115)
    kwargs_289186 = {}
    # Getting the type of 'range' (line 115)
    range_289184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 53), 'range', False)
    # Calling range(args, kwargs) (line 115)
    range_call_result_289187 = invoke(stypy.reporting.localization.Localization(__file__, 115, 53), range_289184, *[N_289185], **kwargs_289186)
    
    comprehension_289188 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 14), range_call_result_289187)
    # Assigning a type to the variable 'j' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'j', comprehension_289188)
    # Getting the type of 'mpmath' (line 115)
    mpmath_289174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'mpmath')
    # Obtaining the member 'pi' of a type (line 115)
    pi_289175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 14), mpmath_289174, 'pi')
    
    # Call to mpf(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'j' (line 115)
    j_289178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'j', False)
    # Processing the call keyword arguments (line 115)
    kwargs_289179 = {}
    # Getting the type of 'mpmath' (line 115)
    mpmath_289176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'mpmath', False)
    # Obtaining the member 'mpf' of a type (line 115)
    mpf_289177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 26), mpmath_289176, 'mpf')
    # Calling mpf(args, kwargs) (line 115)
    mpf_call_result_289180 = invoke(stypy.reporting.localization.Localization(__file__, 115, 26), mpf_289177, *[j_289178], **kwargs_289179)
    
    # Applying the binary operator '*' (line 115)
    result_mul_289181 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 14), '*', pi_289175, mpf_call_result_289180)
    
    # Getting the type of 'N' (line 115)
    N_289182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'N')
    # Applying the binary operator 'div' (line 115)
    result_div_289183 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 40), 'div', result_mul_289181, N_289182)
    
    list_289189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 14), list_289189, result_div_289183)
    # Assigning a type to the variable 'ws' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'ws', list_289189)
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 117):
    
    # Assigning a Name to a Name (line 117):
    # Getting the type of 'worN' (line 117)
    worN_289190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'worN')
    # Assigning a type to the variable 'ws' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'ws', worN_289190)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 119):
    
    # Assigning a List to a Name (line 119):
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_289191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    
    # Assigning a type to the variable 'h' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'h', list_289191)
    
    # Getting the type of 'ws' (line 120)
    ws_289192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'ws')
    # Testing the type of a for loop iterable (line 120)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 120, 4), ws_289192)
    # Getting the type of the for loop variable (line 120)
    for_loop_var_289193 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 120, 4), ws_289192)
    # Assigning a type to the variable 'wk' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'wk', for_loop_var_289193)
    # SSA begins for a for statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to exp(...): (line 121)
    # Processing the call arguments (line 121)
    complex_289196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 25), 'complex')
    # Getting the type of 'wk' (line 121)
    wk_289197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'wk', False)
    # Applying the binary operator '*' (line 121)
    result_mul_289198 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 25), '*', complex_289196, wk_289197)
    
    # Processing the call keyword arguments (line 121)
    kwargs_289199 = {}
    # Getting the type of 'mpmath' (line 121)
    mpmath_289194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 14), 'mpmath', False)
    # Obtaining the member 'exp' of a type (line 121)
    exp_289195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 14), mpmath_289194, 'exp')
    # Calling exp(args, kwargs) (line 121)
    exp_call_result_289200 = invoke(stypy.reporting.localization.Localization(__file__, 121, 14), exp_289195, *[result_mul_289198], **kwargs_289199)
    
    # Assigning a type to the variable 'zm1' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'zm1', exp_call_result_289200)
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to _prod(...): (line 122)
    # Processing the call arguments (line 122)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'z' (line 122)
    z_289205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'z', False)
    comprehension_289206 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 23), z_289205)
    # Assigning a type to the variable 't' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 't', comprehension_289206)
    # Getting the type of 'zm1' (line 122)
    zm1_289202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'zm1', False)
    # Getting the type of 't' (line 122)
    t_289203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 't', False)
    # Applying the binary operator '-' (line 122)
    result_sub_289204 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 23), '-', zm1_289202, t_289203)
    
    list_289207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 23), list_289207, result_sub_289204)
    # Processing the call keyword arguments (line 122)
    kwargs_289208 = {}
    # Getting the type of '_prod' (line 122)
    _prod_289201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), '_prod', False)
    # Calling _prod(args, kwargs) (line 122)
    _prod_call_result_289209 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), _prod_289201, *[list_289207], **kwargs_289208)
    
    # Assigning a type to the variable 'numer' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'numer', _prod_call_result_289209)
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to _prod(...): (line 123)
    # Processing the call arguments (line 123)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'p' (line 123)
    p_289214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 40), 'p', False)
    comprehension_289215 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 23), p_289214)
    # Assigning a type to the variable 't' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 't', comprehension_289215)
    # Getting the type of 'zm1' (line 123)
    zm1_289211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'zm1', False)
    # Getting the type of 't' (line 123)
    t_289212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 't', False)
    # Applying the binary operator '-' (line 123)
    result_sub_289213 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 23), '-', zm1_289211, t_289212)
    
    list_289216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 23), list_289216, result_sub_289213)
    # Processing the call keyword arguments (line 123)
    kwargs_289217 = {}
    # Getting the type of '_prod' (line 123)
    _prod_289210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), '_prod', False)
    # Calling _prod(args, kwargs) (line 123)
    _prod_call_result_289218 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), _prod_289210, *[list_289216], **kwargs_289217)
    
    # Assigning a type to the variable 'denom' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'denom', _prod_call_result_289218)
    
    # Assigning a BinOp to a Name (line 124):
    
    # Assigning a BinOp to a Name (line 124):
    # Getting the type of 'k' (line 124)
    k_289219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'k')
    # Getting the type of 'numer' (line 124)
    numer_289220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'numer')
    # Applying the binary operator '*' (line 124)
    result_mul_289221 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '*', k_289219, numer_289220)
    
    # Getting the type of 'denom' (line 124)
    denom_289222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'denom')
    # Applying the binary operator 'div' (line 124)
    result_div_289223 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), 'div', result_mul_289221, denom_289222)
    
    # Assigning a type to the variable 'hk' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'hk', result_div_289223)
    
    # Call to append(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'hk' (line 125)
    hk_289226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'hk', False)
    # Processing the call keyword arguments (line 125)
    kwargs_289227 = {}
    # Getting the type of 'h' (line 125)
    h_289224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'h', False)
    # Obtaining the member 'append' of a type (line 125)
    append_289225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), h_289224, 'append')
    # Calling append(args, kwargs) (line 125)
    append_call_result_289228 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), append_289225, *[hk_289226], **kwargs_289227)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 126)
    tuple_289229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 126)
    # Adding element type (line 126)
    # Getting the type of 'ws' (line 126)
    ws_289230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'ws')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 11), tuple_289229, ws_289230)
    # Adding element type (line 126)
    # Getting the type of 'h' (line 126)
    h_289231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 11), tuple_289229, h_289231)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', tuple_289229)
    
    # ################# End of 'zpkfreqz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zpkfreqz' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_289232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289232)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zpkfreqz'
    return stypy_return_type_289232

# Assigning a type to the variable 'zpkfreqz' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'zpkfreqz', zpkfreqz)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
