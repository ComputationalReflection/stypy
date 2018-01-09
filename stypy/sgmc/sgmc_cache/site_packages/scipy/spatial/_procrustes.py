
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This module provides functions to perform full Procrustes analysis.
3: 
4: This code was originally written by Justin Kucynski and ported over from
5: scikit-bio by Yoshiki Vazquez-Baeza.
6: '''
7: 
8: from __future__ import absolute_import, division, print_function
9: 
10: import numpy as np
11: from scipy.linalg import orthogonal_procrustes
12: 
13: 
14: __all__ = ['procrustes']
15: 
16: 
17: def procrustes(data1, data2):
18:     r'''Procrustes analysis, a similarity test for two data sets.
19: 
20:     Each input matrix is a set of points or vectors (the rows of the matrix).
21:     The dimension of the space is the number of columns of each matrix. Given
22:     two identically sized matrices, procrustes standardizes both such that:
23: 
24:     - :math:`tr(AA^{T}) = 1`.
25: 
26:     - Both sets of points are centered around the origin.
27: 
28:     Procrustes ([1]_, [2]_) then applies the optimal transform to the second
29:     matrix (including scaling/dilation, rotations, and reflections) to minimize
30:     :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
31:     pointwise differences between the two input datasets.
32: 
33:     This function was not designed to handle datasets with different numbers of
34:     datapoints (rows).  If two data sets have different dimensionality
35:     (different number of columns), simply add columns of zeros to the smaller
36:     of the two.
37: 
38:     Parameters
39:     ----------
40:     data1 : array_like
41:         Matrix, n rows represent points in k (columns) space `data1` is the
42:         reference data, after it is standardised, the data from `data2` will be
43:         transformed to fit the pattern in `data1` (must have >1 unique points).
44:     data2 : array_like
45:         n rows of data in k space to be fit to `data1`.  Must be the  same
46:         shape ``(numrows, numcols)`` as data1 (must have >1 unique points).
47: 
48:     Returns
49:     -------
50:     mtx1 : array_like
51:         A standardized version of `data1`.
52:     mtx2 : array_like
53:         The orientation of `data2` that best fits `data1`. Centered, but not
54:         necessarily :math:`tr(AA^{T}) = 1`.
55:     disparity : float
56:         :math:`M^{2}` as defined above.
57: 
58:     Raises
59:     ------
60:     ValueError
61:         If the input arrays are not two-dimensional.
62:         If the shape of the input arrays is different.
63:         If the input arrays have zero columns or zero rows.
64: 
65:     See Also
66:     --------
67:     scipy.linalg.orthogonal_procrustes
68:     scipy.spatial.distance.directed_hausdorff : Another similarity test
69:       for two data sets
70: 
71:     Notes
72:     -----
73:     - The disparity should not depend on the order of the input matrices, but
74:       the output matrices will, as only the first output matrix is guaranteed
75:       to be scaled such that :math:`tr(AA^{T}) = 1`.
76: 
77:     - Duplicate data points are generally ok, duplicating a data point will
78:       increase its effect on the procrustes fit.
79: 
80:     - The disparity scales as the number of points per input matrix.
81: 
82:     References
83:     ----------
84:     .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
85:     .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".
86: 
87:     Examples
88:     --------
89:     >>> from scipy.spatial import procrustes
90: 
91:     The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
92:     ``a`` here:
93: 
94:     >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
95:     >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
96:     >>> mtx1, mtx2, disparity = procrustes(a, b)
97:     >>> round(disparity)
98:     0.0
99: 
100:     '''
101:     mtx1 = np.array(data1, dtype=np.double, copy=True)
102:     mtx2 = np.array(data2, dtype=np.double, copy=True)
103: 
104:     if mtx1.ndim != 2 or mtx2.ndim != 2:
105:         raise ValueError("Input matrices must be two-dimensional")
106:     if mtx1.shape != mtx2.shape:
107:         raise ValueError("Input matrices must be of same shape")
108:     if mtx1.size == 0:
109:         raise ValueError("Input matrices must be >0 rows and >0 cols")
110: 
111:     # translate all the data to the origin
112:     mtx1 -= np.mean(mtx1, 0)
113:     mtx2 -= np.mean(mtx2, 0)
114: 
115:     norm1 = np.linalg.norm(mtx1)
116:     norm2 = np.linalg.norm(mtx2)
117: 
118:     if norm1 == 0 or norm2 == 0:
119:         raise ValueError("Input matrices must contain >1 unique points")
120: 
121:     # change scaling of data (in rows) such that trace(mtx*mtx') = 1
122:     mtx1 /= norm1
123:     mtx2 /= norm2
124: 
125:     # transform mtx2 to minimize disparity
126:     R, s = orthogonal_procrustes(mtx1, mtx2)
127:     mtx2 = np.dot(mtx2, R.T) * s
128: 
129:     # measure the dissimilarity between the two datasets
130:     disparity = np.sum(np.square(mtx1 - mtx2))
131: 
132:     return mtx1, mtx2, disparity
133: 
134: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_470794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nThis module provides functions to perform full Procrustes analysis.\n\nThis code was originally written by Justin Kucynski and ported over from\nscikit-bio by Yoshiki Vazquez-Baeza.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470795 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_470795) is not StypyTypeError):

    if (import_470795 != 'pyd_module'):
        __import__(import_470795)
        sys_modules_470796 = sys.modules[import_470795]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_470796.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_470795)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg import orthogonal_procrustes' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/spatial/')
import_470797 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_470797) is not StypyTypeError):

    if (import_470797 != 'pyd_module'):
        __import__(import_470797)
        sys_modules_470798 = sys.modules[import_470797]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_470798.module_type_store, module_type_store, ['orthogonal_procrustes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_470798, sys_modules_470798.module_type_store, module_type_store)
    else:
        from scipy.linalg import orthogonal_procrustes

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', None, module_type_store, ['orthogonal_procrustes'], [orthogonal_procrustes])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_470797)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/spatial/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['procrustes']
module_type_store.set_exportable_members(['procrustes'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_470799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_470800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'procrustes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_470799, str_470800)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_470799)

@norecursion
def procrustes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'procrustes'
    module_type_store = module_type_store.open_function_context('procrustes', 17, 0, False)
    
    # Passed parameters checking function
    procrustes.stypy_localization = localization
    procrustes.stypy_type_of_self = None
    procrustes.stypy_type_store = module_type_store
    procrustes.stypy_function_name = 'procrustes'
    procrustes.stypy_param_names_list = ['data1', 'data2']
    procrustes.stypy_varargs_param_name = None
    procrustes.stypy_kwargs_param_name = None
    procrustes.stypy_call_defaults = defaults
    procrustes.stypy_call_varargs = varargs
    procrustes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'procrustes', ['data1', 'data2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'procrustes', localization, ['data1', 'data2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'procrustes(...)' code ##################

    str_470801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', 'Procrustes analysis, a similarity test for two data sets.\n\n    Each input matrix is a set of points or vectors (the rows of the matrix).\n    The dimension of the space is the number of columns of each matrix. Given\n    two identically sized matrices, procrustes standardizes both such that:\n\n    - :math:`tr(AA^{T}) = 1`.\n\n    - Both sets of points are centered around the origin.\n\n    Procrustes ([1]_, [2]_) then applies the optimal transform to the second\n    matrix (including scaling/dilation, rotations, and reflections) to minimize\n    :math:`M^{2}=\\sum(data1-data2)^{2}`, or the sum of the squares of the\n    pointwise differences between the two input datasets.\n\n    This function was not designed to handle datasets with different numbers of\n    datapoints (rows).  If two data sets have different dimensionality\n    (different number of columns), simply add columns of zeros to the smaller\n    of the two.\n\n    Parameters\n    ----------\n    data1 : array_like\n        Matrix, n rows represent points in k (columns) space `data1` is the\n        reference data, after it is standardised, the data from `data2` will be\n        transformed to fit the pattern in `data1` (must have >1 unique points).\n    data2 : array_like\n        n rows of data in k space to be fit to `data1`.  Must be the  same\n        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).\n\n    Returns\n    -------\n    mtx1 : array_like\n        A standardized version of `data1`.\n    mtx2 : array_like\n        The orientation of `data2` that best fits `data1`. Centered, but not\n        necessarily :math:`tr(AA^{T}) = 1`.\n    disparity : float\n        :math:`M^{2}` as defined above.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays are not two-dimensional.\n        If the shape of the input arrays is different.\n        If the input arrays have zero columns or zero rows.\n\n    See Also\n    --------\n    scipy.linalg.orthogonal_procrustes\n    scipy.spatial.distance.directed_hausdorff : Another similarity test\n      for two data sets\n\n    Notes\n    -----\n    - The disparity should not depend on the order of the input matrices, but\n      the output matrices will, as only the first output matrix is guaranteed\n      to be scaled such that :math:`tr(AA^{T}) = 1`.\n\n    - Duplicate data points are generally ok, duplicating a data point will\n      increase its effect on the procrustes fit.\n\n    - The disparity scales as the number of points per input matrix.\n\n    References\n    ----------\n    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".\n    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".\n\n    Examples\n    --------\n    >>> from scipy.spatial import procrustes\n\n    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of\n    ``a`` here:\n\n    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], \'d\')\n    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], \'d\')\n    >>> mtx1, mtx2, disparity = procrustes(a, b)\n    >>> round(disparity)\n    0.0\n\n    ')
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to array(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'data1' (line 101)
    data1_470804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'data1', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'np' (line 101)
    np_470805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'np', False)
    # Obtaining the member 'double' of a type (line 101)
    double_470806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 33), np_470805, 'double')
    keyword_470807 = double_470806
    # Getting the type of 'True' (line 101)
    True_470808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'True', False)
    keyword_470809 = True_470808
    kwargs_470810 = {'dtype': keyword_470807, 'copy': keyword_470809}
    # Getting the type of 'np' (line 101)
    np_470802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 101)
    array_470803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 11), np_470802, 'array')
    # Calling array(args, kwargs) (line 101)
    array_call_result_470811 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), array_470803, *[data1_470804], **kwargs_470810)
    
    # Assigning a type to the variable 'mtx1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'mtx1', array_call_result_470811)
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to array(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'data2' (line 102)
    data2_470814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'data2', False)
    # Processing the call keyword arguments (line 102)
    # Getting the type of 'np' (line 102)
    np_470815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'np', False)
    # Obtaining the member 'double' of a type (line 102)
    double_470816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 33), np_470815, 'double')
    keyword_470817 = double_470816
    # Getting the type of 'True' (line 102)
    True_470818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'True', False)
    keyword_470819 = True_470818
    kwargs_470820 = {'dtype': keyword_470817, 'copy': keyword_470819}
    # Getting the type of 'np' (line 102)
    np_470812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 102)
    array_470813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), np_470812, 'array')
    # Calling array(args, kwargs) (line 102)
    array_call_result_470821 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), array_470813, *[data2_470814], **kwargs_470820)
    
    # Assigning a type to the variable 'mtx2' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'mtx2', array_call_result_470821)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mtx1' (line 104)
    mtx1_470822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'mtx1')
    # Obtaining the member 'ndim' of a type (line 104)
    ndim_470823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 7), mtx1_470822, 'ndim')
    int_470824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 20), 'int')
    # Applying the binary operator '!=' (line 104)
    result_ne_470825 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), '!=', ndim_470823, int_470824)
    
    
    # Getting the type of 'mtx2' (line 104)
    mtx2_470826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'mtx2')
    # Obtaining the member 'ndim' of a type (line 104)
    ndim_470827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), mtx2_470826, 'ndim')
    int_470828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 38), 'int')
    # Applying the binary operator '!=' (line 104)
    result_ne_470829 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 25), '!=', ndim_470827, int_470828)
    
    # Applying the binary operator 'or' (line 104)
    result_or_keyword_470830 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), 'or', result_ne_470825, result_ne_470829)
    
    # Testing the type of an if condition (line 104)
    if_condition_470831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_or_keyword_470830)
    # Assigning a type to the variable 'if_condition_470831' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_470831', if_condition_470831)
    # SSA begins for if statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 105)
    # Processing the call arguments (line 105)
    str_470833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'str', 'Input matrices must be two-dimensional')
    # Processing the call keyword arguments (line 105)
    kwargs_470834 = {}
    # Getting the type of 'ValueError' (line 105)
    ValueError_470832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 105)
    ValueError_call_result_470835 = invoke(stypy.reporting.localization.Localization(__file__, 105, 14), ValueError_470832, *[str_470833], **kwargs_470834)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 105, 8), ValueError_call_result_470835, 'raise parameter', BaseException)
    # SSA join for if statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mtx1' (line 106)
    mtx1_470836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 7), 'mtx1')
    # Obtaining the member 'shape' of a type (line 106)
    shape_470837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 7), mtx1_470836, 'shape')
    # Getting the type of 'mtx2' (line 106)
    mtx2_470838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'mtx2')
    # Obtaining the member 'shape' of a type (line 106)
    shape_470839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 21), mtx2_470838, 'shape')
    # Applying the binary operator '!=' (line 106)
    result_ne_470840 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 7), '!=', shape_470837, shape_470839)
    
    # Testing the type of an if condition (line 106)
    if_condition_470841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 4), result_ne_470840)
    # Assigning a type to the variable 'if_condition_470841' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'if_condition_470841', if_condition_470841)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 107)
    # Processing the call arguments (line 107)
    str_470843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'str', 'Input matrices must be of same shape')
    # Processing the call keyword arguments (line 107)
    kwargs_470844 = {}
    # Getting the type of 'ValueError' (line 107)
    ValueError_470842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 107)
    ValueError_call_result_470845 = invoke(stypy.reporting.localization.Localization(__file__, 107, 14), ValueError_470842, *[str_470843], **kwargs_470844)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 8), ValueError_call_result_470845, 'raise parameter', BaseException)
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mtx1' (line 108)
    mtx1_470846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'mtx1')
    # Obtaining the member 'size' of a type (line 108)
    size_470847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 7), mtx1_470846, 'size')
    int_470848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'int')
    # Applying the binary operator '==' (line 108)
    result_eq_470849 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), '==', size_470847, int_470848)
    
    # Testing the type of an if condition (line 108)
    if_condition_470850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_eq_470849)
    # Assigning a type to the variable 'if_condition_470850' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_470850', if_condition_470850)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 109)
    # Processing the call arguments (line 109)
    str_470852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'str', 'Input matrices must be >0 rows and >0 cols')
    # Processing the call keyword arguments (line 109)
    kwargs_470853 = {}
    # Getting the type of 'ValueError' (line 109)
    ValueError_470851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 109)
    ValueError_call_result_470854 = invoke(stypy.reporting.localization.Localization(__file__, 109, 14), ValueError_470851, *[str_470852], **kwargs_470853)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 109, 8), ValueError_call_result_470854, 'raise parameter', BaseException)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mtx1' (line 112)
    mtx1_470855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'mtx1')
    
    # Call to mean(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'mtx1' (line 112)
    mtx1_470858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'mtx1', False)
    int_470859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 26), 'int')
    # Processing the call keyword arguments (line 112)
    kwargs_470860 = {}
    # Getting the type of 'np' (line 112)
    np_470856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'np', False)
    # Obtaining the member 'mean' of a type (line 112)
    mean_470857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), np_470856, 'mean')
    # Calling mean(args, kwargs) (line 112)
    mean_call_result_470861 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), mean_470857, *[mtx1_470858, int_470859], **kwargs_470860)
    
    # Applying the binary operator '-=' (line 112)
    result_isub_470862 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 4), '-=', mtx1_470855, mean_call_result_470861)
    # Assigning a type to the variable 'mtx1' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'mtx1', result_isub_470862)
    
    
    # Getting the type of 'mtx2' (line 113)
    mtx2_470863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'mtx2')
    
    # Call to mean(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'mtx2' (line 113)
    mtx2_470866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'mtx2', False)
    int_470867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 26), 'int')
    # Processing the call keyword arguments (line 113)
    kwargs_470868 = {}
    # Getting the type of 'np' (line 113)
    np_470864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'np', False)
    # Obtaining the member 'mean' of a type (line 113)
    mean_470865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), np_470864, 'mean')
    # Calling mean(args, kwargs) (line 113)
    mean_call_result_470869 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), mean_470865, *[mtx2_470866, int_470867], **kwargs_470868)
    
    # Applying the binary operator '-=' (line 113)
    result_isub_470870 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 4), '-=', mtx2_470863, mean_call_result_470869)
    # Assigning a type to the variable 'mtx2' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'mtx2', result_isub_470870)
    
    
    # Assigning a Call to a Name (line 115):
    
    # Assigning a Call to a Name (line 115):
    
    # Call to norm(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'mtx1' (line 115)
    mtx1_470874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'mtx1', False)
    # Processing the call keyword arguments (line 115)
    kwargs_470875 = {}
    # Getting the type of 'np' (line 115)
    np_470871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 115)
    linalg_470872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), np_470871, 'linalg')
    # Obtaining the member 'norm' of a type (line 115)
    norm_470873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), linalg_470872, 'norm')
    # Calling norm(args, kwargs) (line 115)
    norm_call_result_470876 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), norm_470873, *[mtx1_470874], **kwargs_470875)
    
    # Assigning a type to the variable 'norm1' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'norm1', norm_call_result_470876)
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to norm(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'mtx2' (line 116)
    mtx2_470880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'mtx2', False)
    # Processing the call keyword arguments (line 116)
    kwargs_470881 = {}
    # Getting the type of 'np' (line 116)
    np_470877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'np', False)
    # Obtaining the member 'linalg' of a type (line 116)
    linalg_470878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), np_470877, 'linalg')
    # Obtaining the member 'norm' of a type (line 116)
    norm_470879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), linalg_470878, 'norm')
    # Calling norm(args, kwargs) (line 116)
    norm_call_result_470882 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), norm_470879, *[mtx2_470880], **kwargs_470881)
    
    # Assigning a type to the variable 'norm2' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'norm2', norm_call_result_470882)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'norm1' (line 118)
    norm1_470883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'norm1')
    int_470884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'int')
    # Applying the binary operator '==' (line 118)
    result_eq_470885 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), '==', norm1_470883, int_470884)
    
    
    # Getting the type of 'norm2' (line 118)
    norm2_470886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'norm2')
    int_470887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 30), 'int')
    # Applying the binary operator '==' (line 118)
    result_eq_470888 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 21), '==', norm2_470886, int_470887)
    
    # Applying the binary operator 'or' (line 118)
    result_or_keyword_470889 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), 'or', result_eq_470885, result_eq_470888)
    
    # Testing the type of an if condition (line 118)
    if_condition_470890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_or_keyword_470889)
    # Assigning a type to the variable 'if_condition_470890' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_470890', if_condition_470890)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 119)
    # Processing the call arguments (line 119)
    str_470892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'str', 'Input matrices must contain >1 unique points')
    # Processing the call keyword arguments (line 119)
    kwargs_470893 = {}
    # Getting the type of 'ValueError' (line 119)
    ValueError_470891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 119)
    ValueError_call_result_470894 = invoke(stypy.reporting.localization.Localization(__file__, 119, 14), ValueError_470891, *[str_470892], **kwargs_470893)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 8), ValueError_call_result_470894, 'raise parameter', BaseException)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'mtx1' (line 122)
    mtx1_470895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'mtx1')
    # Getting the type of 'norm1' (line 122)
    norm1_470896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'norm1')
    # Applying the binary operator 'div=' (line 122)
    result_div_470897 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 4), 'div=', mtx1_470895, norm1_470896)
    # Assigning a type to the variable 'mtx1' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'mtx1', result_div_470897)
    
    
    # Getting the type of 'mtx2' (line 123)
    mtx2_470898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'mtx2')
    # Getting the type of 'norm2' (line 123)
    norm2_470899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'norm2')
    # Applying the binary operator 'div=' (line 123)
    result_div_470900 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 4), 'div=', mtx2_470898, norm2_470899)
    # Assigning a type to the variable 'mtx2' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'mtx2', result_div_470900)
    
    
    # Assigning a Call to a Tuple (line 126):
    
    # Assigning a Subscript to a Name (line 126):
    
    # Obtaining the type of the subscript
    int_470901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'mtx1' (line 126)
    mtx1_470903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'mtx1', False)
    # Getting the type of 'mtx2' (line 126)
    mtx2_470904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'mtx2', False)
    # Processing the call keyword arguments (line 126)
    kwargs_470905 = {}
    # Getting the type of 'orthogonal_procrustes' (line 126)
    orthogonal_procrustes_470902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 126)
    orthogonal_procrustes_call_result_470906 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), orthogonal_procrustes_470902, *[mtx1_470903, mtx2_470904], **kwargs_470905)
    
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___470907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 4), orthogonal_procrustes_call_result_470906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_470908 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), getitem___470907, int_470901)
    
    # Assigning a type to the variable 'tuple_var_assignment_470792' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'tuple_var_assignment_470792', subscript_call_result_470908)
    
    # Assigning a Subscript to a Name (line 126):
    
    # Obtaining the type of the subscript
    int_470909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 4), 'int')
    
    # Call to orthogonal_procrustes(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'mtx1' (line 126)
    mtx1_470911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'mtx1', False)
    # Getting the type of 'mtx2' (line 126)
    mtx2_470912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'mtx2', False)
    # Processing the call keyword arguments (line 126)
    kwargs_470913 = {}
    # Getting the type of 'orthogonal_procrustes' (line 126)
    orthogonal_procrustes_470910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'orthogonal_procrustes', False)
    # Calling orthogonal_procrustes(args, kwargs) (line 126)
    orthogonal_procrustes_call_result_470914 = invoke(stypy.reporting.localization.Localization(__file__, 126, 11), orthogonal_procrustes_470910, *[mtx1_470911, mtx2_470912], **kwargs_470913)
    
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___470915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 4), orthogonal_procrustes_call_result_470914, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_470916 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), getitem___470915, int_470909)
    
    # Assigning a type to the variable 'tuple_var_assignment_470793' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'tuple_var_assignment_470793', subscript_call_result_470916)
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'tuple_var_assignment_470792' (line 126)
    tuple_var_assignment_470792_470917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'tuple_var_assignment_470792')
    # Assigning a type to the variable 'R' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'R', tuple_var_assignment_470792_470917)
    
    # Assigning a Name to a Name (line 126):
    # Getting the type of 'tuple_var_assignment_470793' (line 126)
    tuple_var_assignment_470793_470918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'tuple_var_assignment_470793')
    # Assigning a type to the variable 's' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 's', tuple_var_assignment_470793_470918)
    
    # Assigning a BinOp to a Name (line 127):
    
    # Assigning a BinOp to a Name (line 127):
    
    # Call to dot(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'mtx2' (line 127)
    mtx2_470921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'mtx2', False)
    # Getting the type of 'R' (line 127)
    R_470922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'R', False)
    # Obtaining the member 'T' of a type (line 127)
    T_470923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), R_470922, 'T')
    # Processing the call keyword arguments (line 127)
    kwargs_470924 = {}
    # Getting the type of 'np' (line 127)
    np_470919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'np', False)
    # Obtaining the member 'dot' of a type (line 127)
    dot_470920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 11), np_470919, 'dot')
    # Calling dot(args, kwargs) (line 127)
    dot_call_result_470925 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), dot_470920, *[mtx2_470921, T_470923], **kwargs_470924)
    
    # Getting the type of 's' (line 127)
    s_470926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 's')
    # Applying the binary operator '*' (line 127)
    result_mul_470927 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 11), '*', dot_call_result_470925, s_470926)
    
    # Assigning a type to the variable 'mtx2' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'mtx2', result_mul_470927)
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to sum(...): (line 130)
    # Processing the call arguments (line 130)
    
    # Call to square(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'mtx1' (line 130)
    mtx1_470932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'mtx1', False)
    # Getting the type of 'mtx2' (line 130)
    mtx2_470933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'mtx2', False)
    # Applying the binary operator '-' (line 130)
    result_sub_470934 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 33), '-', mtx1_470932, mtx2_470933)
    
    # Processing the call keyword arguments (line 130)
    kwargs_470935 = {}
    # Getting the type of 'np' (line 130)
    np_470930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 130)
    square_470931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 23), np_470930, 'square')
    # Calling square(args, kwargs) (line 130)
    square_call_result_470936 = invoke(stypy.reporting.localization.Localization(__file__, 130, 23), square_470931, *[result_sub_470934], **kwargs_470935)
    
    # Processing the call keyword arguments (line 130)
    kwargs_470937 = {}
    # Getting the type of 'np' (line 130)
    np_470928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'np', False)
    # Obtaining the member 'sum' of a type (line 130)
    sum_470929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), np_470928, 'sum')
    # Calling sum(args, kwargs) (line 130)
    sum_call_result_470938 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), sum_470929, *[square_call_result_470936], **kwargs_470937)
    
    # Assigning a type to the variable 'disparity' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'disparity', sum_call_result_470938)
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_470939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    # Getting the type of 'mtx1' (line 132)
    mtx1_470940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'mtx1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 11), tuple_470939, mtx1_470940)
    # Adding element type (line 132)
    # Getting the type of 'mtx2' (line 132)
    mtx2_470941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'mtx2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 11), tuple_470939, mtx2_470941)
    # Adding element type (line 132)
    # Getting the type of 'disparity' (line 132)
    disparity_470942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'disparity')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 11), tuple_470939, disparity_470942)
    
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type', tuple_470939)
    
    # ################# End of 'procrustes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'procrustes' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_470943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_470943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'procrustes'
    return stypy_return_type_470943

# Assigning a type to the variable 'procrustes' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'procrustes', procrustes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
