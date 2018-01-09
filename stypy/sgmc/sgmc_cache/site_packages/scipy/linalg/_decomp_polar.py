
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy.linalg import svd
5: 
6: 
7: __all__ = ['polar']
8: 
9: 
10: def polar(a, side="right"):
11:     '''
12:     Compute the polar decomposition.
13: 
14:     Returns the factors of the polar decomposition [1]_ `u` and `p` such
15:     that ``a = up`` (if `side` is "right") or ``a = pu`` (if `side` is
16:     "left"), where `p` is positive semidefinite.  Depending on the shape
17:     of `a`, either the rows or columns of `u` are orthonormal.  When `a`
18:     is a square array, `u` is a square unitary array.  When `a` is not
19:     square, the "canonical polar decomposition" [2]_ is computed.
20: 
21:     Parameters
22:     ----------
23:     a : (m, n) array_like
24:         The array to be factored.
25:     side : {'left', 'right'}, optional
26:         Determines whether a right or left polar decomposition is computed.
27:         If `side` is "right", then ``a = up``.  If `side` is "left",  then
28:         ``a = pu``.  The default is "right".
29: 
30:     Returns
31:     -------
32:     u : (m, n) ndarray
33:         If `a` is square, then `u` is unitary.  If m > n, then the columns
34:         of `a` are orthonormal, and if m < n, then the rows of `u` are
35:         orthonormal.
36:     p : ndarray
37:         `p` is Hermitian positive semidefinite.  If `a` is nonsingular, `p`
38:         is positive definite.  The shape of `p` is (n, n) or (m, m), depending
39:         on whether `side` is "right" or "left", respectively.
40: 
41:     References
42:     ----------
43:     .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge
44:            University Press, 1985.
45:     .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",
46:            SIAM, 2008.
47: 
48:     Examples
49:     --------
50:     >>> from scipy.linalg import polar
51:     >>> a = np.array([[1, -1], [2, 4]])
52:     >>> u, p = polar(a)
53:     >>> u
54:     array([[ 0.85749293, -0.51449576],
55:            [ 0.51449576,  0.85749293]])
56:     >>> p
57:     array([[ 1.88648444,  1.2004901 ],
58:            [ 1.2004901 ,  3.94446746]])
59: 
60:     A non-square example, with m < n:
61: 
62:     >>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])
63:     >>> u, p = polar(b)
64:     >>> u
65:     array([[-0.21196618, -0.42393237,  0.88054056],
66:            [ 0.39378971,  0.78757942,  0.4739708 ]])
67:     >>> p
68:     array([[ 0.48470147,  0.96940295,  1.15122648],
69:            [ 0.96940295,  1.9388059 ,  2.30245295],
70:            [ 1.15122648,  2.30245295,  3.65696431]])
71:     >>> u.dot(p)   # Verify the decomposition.
72:     array([[ 0.5,  1. ,  2. ],
73:            [ 1.5,  3. ,  4. ]])
74:     >>> u.dot(u.T)   # The rows of u are orthonormal.
75:     array([[  1.00000000e+00,  -2.07353665e-17],
76:            [ -2.07353665e-17,   1.00000000e+00]])
77: 
78:     Another non-square example, with m > n:
79: 
80:     >>> c = b.T
81:     >>> u, p = polar(c)
82:     >>> u
83:     array([[-0.21196618,  0.39378971],
84:            [-0.42393237,  0.78757942],
85:            [ 0.88054056,  0.4739708 ]])
86:     >>> p
87:     array([[ 1.23116567,  1.93241587],
88:            [ 1.93241587,  4.84930602]])
89:     >>> u.dot(p)   # Verify the decomposition.
90:     array([[ 0.5,  1.5],
91:            [ 1. ,  3. ],
92:            [ 2. ,  4. ]])
93:     >>> u.T.dot(u)  # The columns of u are orthonormal.
94:     array([[  1.00000000e+00,  -1.26363763e-16],
95:            [ -1.26363763e-16,   1.00000000e+00]])
96: 
97:     '''
98:     if side not in ['right', 'left']:
99:         raise ValueError("`side` must be either 'right' or 'left'")
100:     a = np.asarray(a)
101:     if a.ndim != 2:
102:         raise ValueError("`a` must be a 2-D array.")
103: 
104:     w, s, vh = svd(a, full_matrices=False)
105:     u = w.dot(vh)
106:     if side == 'right':
107:         # a = up
108:         p = (vh.T.conj() * s).dot(vh)
109:     else:
110:         # a = pu
111:         p = (w * s).dot(w.T.conj())
112:     return u, p
113: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_25713) is not StypyTypeError):

    if (import_25713 != 'pyd_module'):
        __import__(import_25713)
        sys_modules_25714 = sys.modules[import_25713]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_25714.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_25713)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.linalg import svd' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg')

if (type(import_25715) is not StypyTypeError):

    if (import_25715 != 'pyd_module'):
        __import__(import_25715)
        sys_modules_25716 = sys.modules[import_25715]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', sys_modules_25716.module_type_store, module_type_store, ['svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_25716, sys_modules_25716.module_type_store, module_type_store)
    else:
        from scipy.linalg import svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', None, module_type_store, ['svd'], [svd])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', import_25715)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['polar']
module_type_store.set_exportable_members(['polar'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_25717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_25718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'polar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_25717, str_25718)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_25717)

@norecursion
def polar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_25719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'right')
    defaults = [str_25719]
    # Create a new context for function 'polar'
    module_type_store = module_type_store.open_function_context('polar', 10, 0, False)
    
    # Passed parameters checking function
    polar.stypy_localization = localization
    polar.stypy_type_of_self = None
    polar.stypy_type_store = module_type_store
    polar.stypy_function_name = 'polar'
    polar.stypy_param_names_list = ['a', 'side']
    polar.stypy_varargs_param_name = None
    polar.stypy_kwargs_param_name = None
    polar.stypy_call_defaults = defaults
    polar.stypy_call_varargs = varargs
    polar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polar', ['a', 'side'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polar', localization, ['a', 'side'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polar(...)' code ##################

    str_25720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    Compute the polar decomposition.\n\n    Returns the factors of the polar decomposition [1]_ `u` and `p` such\n    that ``a = up`` (if `side` is "right") or ``a = pu`` (if `side` is\n    "left"), where `p` is positive semidefinite.  Depending on the shape\n    of `a`, either the rows or columns of `u` are orthonormal.  When `a`\n    is a square array, `u` is a square unitary array.  When `a` is not\n    square, the "canonical polar decomposition" [2]_ is computed.\n\n    Parameters\n    ----------\n    a : (m, n) array_like\n        The array to be factored.\n    side : {\'left\', \'right\'}, optional\n        Determines whether a right or left polar decomposition is computed.\n        If `side` is "right", then ``a = up``.  If `side` is "left",  then\n        ``a = pu``.  The default is "right".\n\n    Returns\n    -------\n    u : (m, n) ndarray\n        If `a` is square, then `u` is unitary.  If m > n, then the columns\n        of `a` are orthonormal, and if m < n, then the rows of `u` are\n        orthonormal.\n    p : ndarray\n        `p` is Hermitian positive semidefinite.  If `a` is nonsingular, `p`\n        is positive definite.  The shape of `p` is (n, n) or (m, m), depending\n        on whether `side` is "right" or "left", respectively.\n\n    References\n    ----------\n    .. [1] R. A. Horn and C. R. Johnson, "Matrix Analysis", Cambridge\n           University Press, 1985.\n    .. [2] N. J. Higham, "Functions of Matrices: Theory and Computation",\n           SIAM, 2008.\n\n    Examples\n    --------\n    >>> from scipy.linalg import polar\n    >>> a = np.array([[1, -1], [2, 4]])\n    >>> u, p = polar(a)\n    >>> u\n    array([[ 0.85749293, -0.51449576],\n           [ 0.51449576,  0.85749293]])\n    >>> p\n    array([[ 1.88648444,  1.2004901 ],\n           [ 1.2004901 ,  3.94446746]])\n\n    A non-square example, with m < n:\n\n    >>> b = np.array([[0.5, 1, 2], [1.5, 3, 4]])\n    >>> u, p = polar(b)\n    >>> u\n    array([[-0.21196618, -0.42393237,  0.88054056],\n           [ 0.39378971,  0.78757942,  0.4739708 ]])\n    >>> p\n    array([[ 0.48470147,  0.96940295,  1.15122648],\n           [ 0.96940295,  1.9388059 ,  2.30245295],\n           [ 1.15122648,  2.30245295,  3.65696431]])\n    >>> u.dot(p)   # Verify the decomposition.\n    array([[ 0.5,  1. ,  2. ],\n           [ 1.5,  3. ,  4. ]])\n    >>> u.dot(u.T)   # The rows of u are orthonormal.\n    array([[  1.00000000e+00,  -2.07353665e-17],\n           [ -2.07353665e-17,   1.00000000e+00]])\n\n    Another non-square example, with m > n:\n\n    >>> c = b.T\n    >>> u, p = polar(c)\n    >>> u\n    array([[-0.21196618,  0.39378971],\n           [-0.42393237,  0.78757942],\n           [ 0.88054056,  0.4739708 ]])\n    >>> p\n    array([[ 1.23116567,  1.93241587],\n           [ 1.93241587,  4.84930602]])\n    >>> u.dot(p)   # Verify the decomposition.\n    array([[ 0.5,  1.5],\n           [ 1. ,  3. ],\n           [ 2. ,  4. ]])\n    >>> u.T.dot(u)  # The columns of u are orthonormal.\n    array([[  1.00000000e+00,  -1.26363763e-16],\n           [ -1.26363763e-16,   1.00000000e+00]])\n\n    ')
    
    
    # Getting the type of 'side' (line 98)
    side_25721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'side')
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_25722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    str_25723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 20), 'str', 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_25722, str_25723)
    # Adding element type (line 98)
    str_25724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'str', 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 19), list_25722, str_25724)
    
    # Applying the binary operator 'notin' (line 98)
    result_contains_25725 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 7), 'notin', side_25721, list_25722)
    
    # Testing the type of an if condition (line 98)
    if_condition_25726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), result_contains_25725)
    # Assigning a type to the variable 'if_condition_25726' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_25726', if_condition_25726)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 99)
    # Processing the call arguments (line 99)
    str_25728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'str', "`side` must be either 'right' or 'left'")
    # Processing the call keyword arguments (line 99)
    kwargs_25729 = {}
    # Getting the type of 'ValueError' (line 99)
    ValueError_25727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 99)
    ValueError_call_result_25730 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), ValueError_25727, *[str_25728], **kwargs_25729)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 8), ValueError_call_result_25730, 'raise parameter', BaseException)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to asarray(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'a' (line 100)
    a_25733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'a', False)
    # Processing the call keyword arguments (line 100)
    kwargs_25734 = {}
    # Getting the type of 'np' (line 100)
    np_25731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 100)
    asarray_25732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), np_25731, 'asarray')
    # Calling asarray(args, kwargs) (line 100)
    asarray_call_result_25735 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), asarray_25732, *[a_25733], **kwargs_25734)
    
    # Assigning a type to the variable 'a' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'a', asarray_call_result_25735)
    
    
    # Getting the type of 'a' (line 101)
    a_25736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 101)
    ndim_25737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 7), a_25736, 'ndim')
    int_25738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'int')
    # Applying the binary operator '!=' (line 101)
    result_ne_25739 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '!=', ndim_25737, int_25738)
    
    # Testing the type of an if condition (line 101)
    if_condition_25740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_ne_25739)
    # Assigning a type to the variable 'if_condition_25740' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_25740', if_condition_25740)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 102)
    # Processing the call arguments (line 102)
    str_25742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'str', '`a` must be a 2-D array.')
    # Processing the call keyword arguments (line 102)
    kwargs_25743 = {}
    # Getting the type of 'ValueError' (line 102)
    ValueError_25741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 102)
    ValueError_call_result_25744 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), ValueError_25741, *[str_25742], **kwargs_25743)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 8), ValueError_call_result_25744, 'raise parameter', BaseException)
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 104):
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    int_25745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'int')
    
    # Call to svd(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'a' (line 104)
    a_25747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'a', False)
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'False' (line 104)
    False_25748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'False', False)
    keyword_25749 = False_25748
    kwargs_25750 = {'full_matrices': keyword_25749}
    # Getting the type of 'svd' (line 104)
    svd_25746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 104)
    svd_call_result_25751 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), svd_25746, *[a_25747], **kwargs_25750)
    
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___25752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), svd_call_result_25751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_25753 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), getitem___25752, int_25745)
    
    # Assigning a type to the variable 'tuple_var_assignment_25710' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25710', subscript_call_result_25753)
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    int_25754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'int')
    
    # Call to svd(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'a' (line 104)
    a_25756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'a', False)
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'False' (line 104)
    False_25757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'False', False)
    keyword_25758 = False_25757
    kwargs_25759 = {'full_matrices': keyword_25758}
    # Getting the type of 'svd' (line 104)
    svd_25755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 104)
    svd_call_result_25760 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), svd_25755, *[a_25756], **kwargs_25759)
    
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___25761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), svd_call_result_25760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_25762 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), getitem___25761, int_25754)
    
    # Assigning a type to the variable 'tuple_var_assignment_25711' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25711', subscript_call_result_25762)
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    int_25763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 4), 'int')
    
    # Call to svd(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'a' (line 104)
    a_25765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'a', False)
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'False' (line 104)
    False_25766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'False', False)
    keyword_25767 = False_25766
    kwargs_25768 = {'full_matrices': keyword_25767}
    # Getting the type of 'svd' (line 104)
    svd_25764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 104)
    svd_call_result_25769 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), svd_25764, *[a_25765], **kwargs_25768)
    
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___25770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), svd_call_result_25769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_25771 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), getitem___25770, int_25763)
    
    # Assigning a type to the variable 'tuple_var_assignment_25712' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25712', subscript_call_result_25771)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'tuple_var_assignment_25710' (line 104)
    tuple_var_assignment_25710_25772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25710')
    # Assigning a type to the variable 'w' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'w', tuple_var_assignment_25710_25772)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'tuple_var_assignment_25711' (line 104)
    tuple_var_assignment_25711_25773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25711')
    # Assigning a type to the variable 's' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 's', tuple_var_assignment_25711_25773)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'tuple_var_assignment_25712' (line 104)
    tuple_var_assignment_25712_25774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_var_assignment_25712')
    # Assigning a type to the variable 'vh' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 10), 'vh', tuple_var_assignment_25712_25774)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to dot(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'vh' (line 105)
    vh_25777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'vh', False)
    # Processing the call keyword arguments (line 105)
    kwargs_25778 = {}
    # Getting the type of 'w' (line 105)
    w_25775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'w', False)
    # Obtaining the member 'dot' of a type (line 105)
    dot_25776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), w_25775, 'dot')
    # Calling dot(args, kwargs) (line 105)
    dot_call_result_25779 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), dot_25776, *[vh_25777], **kwargs_25778)
    
    # Assigning a type to the variable 'u' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'u', dot_call_result_25779)
    
    
    # Getting the type of 'side' (line 106)
    side_25780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 7), 'side')
    str_25781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'str', 'right')
    # Applying the binary operator '==' (line 106)
    result_eq_25782 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), '==', side_25780, str_25781)
    
    # Testing the type of an if condition (line 106)
    if_condition_25783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 4), result_eq_25782)
    # Assigning a type to the variable 'if_condition_25783' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'if_condition_25783', if_condition_25783)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to dot(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'vh' (line 108)
    vh_25792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'vh', False)
    # Processing the call keyword arguments (line 108)
    kwargs_25793 = {}
    
    # Call to conj(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_25787 = {}
    # Getting the type of 'vh' (line 108)
    vh_25784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'vh', False)
    # Obtaining the member 'T' of a type (line 108)
    T_25785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), vh_25784, 'T')
    # Obtaining the member 'conj' of a type (line 108)
    conj_25786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), T_25785, 'conj')
    # Calling conj(args, kwargs) (line 108)
    conj_call_result_25788 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), conj_25786, *[], **kwargs_25787)
    
    # Getting the type of 's' (line 108)
    s_25789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 's', False)
    # Applying the binary operator '*' (line 108)
    result_mul_25790 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 13), '*', conj_call_result_25788, s_25789)
    
    # Obtaining the member 'dot' of a type (line 108)
    dot_25791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), result_mul_25790, 'dot')
    # Calling dot(args, kwargs) (line 108)
    dot_call_result_25794 = invoke(stypy.reporting.localization.Localization(__file__, 108, 13), dot_25791, *[vh_25792], **kwargs_25793)
    
    # Assigning a type to the variable 'p' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'p', dot_call_result_25794)
    # SSA branch for the else part of an if statement (line 106)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to dot(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to conj(...): (line 111)
    # Processing the call keyword arguments (line 111)
    kwargs_25802 = {}
    # Getting the type of 'w' (line 111)
    w_25799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'w', False)
    # Obtaining the member 'T' of a type (line 111)
    T_25800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), w_25799, 'T')
    # Obtaining the member 'conj' of a type (line 111)
    conj_25801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), T_25800, 'conj')
    # Calling conj(args, kwargs) (line 111)
    conj_call_result_25803 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), conj_25801, *[], **kwargs_25802)
    
    # Processing the call keyword arguments (line 111)
    kwargs_25804 = {}
    # Getting the type of 'w' (line 111)
    w_25795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 13), 'w', False)
    # Getting the type of 's' (line 111)
    s_25796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 's', False)
    # Applying the binary operator '*' (line 111)
    result_mul_25797 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 13), '*', w_25795, s_25796)
    
    # Obtaining the member 'dot' of a type (line 111)
    dot_25798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 13), result_mul_25797, 'dot')
    # Calling dot(args, kwargs) (line 111)
    dot_call_result_25805 = invoke(stypy.reporting.localization.Localization(__file__, 111, 13), dot_25798, *[conj_call_result_25803], **kwargs_25804)
    
    # Assigning a type to the variable 'p' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'p', dot_call_result_25805)
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_25806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'u' (line 112)
    u_25807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 11), tuple_25806, u_25807)
    # Adding element type (line 112)
    # Getting the type of 'p' (line 112)
    p_25808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 11), tuple_25806, p_25808)
    
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', tuple_25806)
    
    # ################# End of 'polar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polar' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_25809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polar'
    return stypy_return_type_25809

# Assigning a type to the variable 'polar' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'polar', polar)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
