
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Solve the orthogonal Procrustes problem.
3: 
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import numpy as np
8: from .decomp_svd import svd
9: 
10: 
11: __all__ = ['orthogonal_procrustes']
12: 
13: 
14: def orthogonal_procrustes(A, B, check_finite=True):
15:     '''
16:     Compute the matrix solution of the orthogonal Procrustes problem.
17: 
18:     Given matrices A and B of equal shape, find an orthogonal matrix R
19:     that most closely maps A to B [1]_.
20:     Note that unlike higher level Procrustes analyses of spatial data,
21:     this function only uses orthogonal transformations like rotations
22:     and reflections, and it does not use scaling or translation.
23: 
24:     Parameters
25:     ----------
26:     A : (M, N) array_like
27:         Matrix to be mapped.
28:     B : (M, N) array_like
29:         Target matrix.
30:     check_finite : bool, optional
31:         Whether to check that the input matrices contain only finite numbers.
32:         Disabling may give a performance gain, but may result in problems
33:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
34: 
35:     Returns
36:     -------
37:     R : (N, N) ndarray
38:         The matrix solution of the orthogonal Procrustes problem.
39:         Minimizes the Frobenius norm of dot(A, R) - B, subject to
40:         dot(R.T, R) == I.
41:     scale : float
42:         Sum of the singular values of ``dot(A.T, B)``.
43: 
44:     Raises
45:     ------
46:     ValueError
47:         If the input arrays are incompatibly shaped.
48:         This may also be raised if matrix A or B contains an inf or nan
49:         and check_finite is True, or if the matrix product AB contains
50:         an inf or nan.
51: 
52:     Notes
53:     -----
54:     .. versionadded:: 0.15.0
55: 
56:     References
57:     ----------
58:     .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal
59:            Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.
60: 
61:     '''
62:     if check_finite:
63:         A = np.asarray_chkfinite(A)
64:         B = np.asarray_chkfinite(B)
65:     else:
66:         A = np.asanyarray(A)
67:         B = np.asanyarray(B)
68:     if A.ndim != 2:
69:         raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
70:     if A.shape != B.shape:
71:         raise ValueError('the shapes of A and B differ (%s vs %s)' % (
72:             A.shape, B.shape))
73:     # Be clever with transposes, with the intention to save memory.
74:     u, w, vt = svd(B.T.dot(A).T)
75:     R = u.dot(vt)
76:     scale = w.sum()
77:     return R, scale
78: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nSolve the orthogonal Procrustes problem.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35683 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_35683) is not StypyTypeError):

    if (import_35683 != 'pyd_module'):
        __import__(import_35683)
        sys_modules_35684 = sys.modules[import_35683]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_35684.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_35683)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg.decomp_svd import svd' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_35685 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.decomp_svd')

if (type(import_35685) is not StypyTypeError):

    if (import_35685 != 'pyd_module'):
        __import__(import_35685)
        sys_modules_35686 = sys.modules[import_35685]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.decomp_svd', sys_modules_35686.module_type_store, module_type_store, ['svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_35686, sys_modules_35686.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp_svd import svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.decomp_svd', None, module_type_store, ['svd'], [svd])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp_svd' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.decomp_svd', import_35685)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 11):

# Assigning a List to a Name (line 11):
__all__ = ['orthogonal_procrustes']
module_type_store.set_exportable_members(['orthogonal_procrustes'])

# Obtaining an instance of the builtin type 'list' (line 11)
list_35687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_35688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'orthogonal_procrustes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_35687, str_35688)

# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', list_35687)

@norecursion
def orthogonal_procrustes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 14)
    True_35689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 45), 'True')
    defaults = [True_35689]
    # Create a new context for function 'orthogonal_procrustes'
    module_type_store = module_type_store.open_function_context('orthogonal_procrustes', 14, 0, False)
    
    # Passed parameters checking function
    orthogonal_procrustes.stypy_localization = localization
    orthogonal_procrustes.stypy_type_of_self = None
    orthogonal_procrustes.stypy_type_store = module_type_store
    orthogonal_procrustes.stypy_function_name = 'orthogonal_procrustes'
    orthogonal_procrustes.stypy_param_names_list = ['A', 'B', 'check_finite']
    orthogonal_procrustes.stypy_varargs_param_name = None
    orthogonal_procrustes.stypy_kwargs_param_name = None
    orthogonal_procrustes.stypy_call_defaults = defaults
    orthogonal_procrustes.stypy_call_varargs = varargs
    orthogonal_procrustes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'orthogonal_procrustes', ['A', 'B', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'orthogonal_procrustes', localization, ['A', 'B', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'orthogonal_procrustes(...)' code ##################

    str_35690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    Compute the matrix solution of the orthogonal Procrustes problem.\n\n    Given matrices A and B of equal shape, find an orthogonal matrix R\n    that most closely maps A to B [1]_.\n    Note that unlike higher level Procrustes analyses of spatial data,\n    this function only uses orthogonal transformations like rotations\n    and reflections, and it does not use scaling or translation.\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        Matrix to be mapped.\n    B : (M, N) array_like\n        Target matrix.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    R : (N, N) ndarray\n        The matrix solution of the orthogonal Procrustes problem.\n        Minimizes the Frobenius norm of dot(A, R) - B, subject to\n        dot(R.T, R) == I.\n    scale : float\n        Sum of the singular values of ``dot(A.T, B)``.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays are incompatibly shaped.\n        This may also be raised if matrix A or B contains an inf or nan\n        and check_finite is True, or if the matrix product AB contains\n        an inf or nan.\n\n    Notes\n    -----\n    .. versionadded:: 0.15.0\n\n    References\n    ----------\n    .. [1] Peter H. Schonemann, "A generalized solution of the orthogonal\n           Procrustes problem", Psychometrica -- Vol. 31, No. 1, March, 1996.\n\n    ')
    
    # Getting the type of 'check_finite' (line 62)
    check_finite_35691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 7), 'check_finite')
    # Testing the type of an if condition (line 62)
    if_condition_35692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), check_finite_35691)
    # Assigning a type to the variable 'if_condition_35692' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'if_condition_35692', if_condition_35692)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to asarray_chkfinite(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'A' (line 63)
    A_35695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'A', False)
    # Processing the call keyword arguments (line 63)
    kwargs_35696 = {}
    # Getting the type of 'np' (line 63)
    np_35693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 63)
    asarray_chkfinite_35694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), np_35693, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 63)
    asarray_chkfinite_call_result_35697 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), asarray_chkfinite_35694, *[A_35695], **kwargs_35696)
    
    # Assigning a type to the variable 'A' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'A', asarray_chkfinite_call_result_35697)
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to asarray_chkfinite(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'B' (line 64)
    B_35700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'B', False)
    # Processing the call keyword arguments (line 64)
    kwargs_35701 = {}
    # Getting the type of 'np' (line 64)
    np_35698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 64)
    asarray_chkfinite_35699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), np_35698, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 64)
    asarray_chkfinite_call_result_35702 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), asarray_chkfinite_35699, *[B_35700], **kwargs_35701)
    
    # Assigning a type to the variable 'B' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'B', asarray_chkfinite_call_result_35702)
    # SSA branch for the else part of an if statement (line 62)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to asanyarray(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'A' (line 66)
    A_35705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'A', False)
    # Processing the call keyword arguments (line 66)
    kwargs_35706 = {}
    # Getting the type of 'np' (line 66)
    np_35703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 66)
    asanyarray_35704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), np_35703, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 66)
    asanyarray_call_result_35707 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), asanyarray_35704, *[A_35705], **kwargs_35706)
    
    # Assigning a type to the variable 'A' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'A', asanyarray_call_result_35707)
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to asanyarray(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'B' (line 67)
    B_35710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'B', False)
    # Processing the call keyword arguments (line 67)
    kwargs_35711 = {}
    # Getting the type of 'np' (line 67)
    np_35708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 67)
    asanyarray_35709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), np_35708, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 67)
    asanyarray_call_result_35712 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), asanyarray_35709, *[B_35710], **kwargs_35711)
    
    # Assigning a type to the variable 'B' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'B', asanyarray_call_result_35712)
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 68)
    A_35713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'A')
    # Obtaining the member 'ndim' of a type (line 68)
    ndim_35714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 7), A_35713, 'ndim')
    int_35715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'int')
    # Applying the binary operator '!=' (line 68)
    result_ne_35716 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '!=', ndim_35714, int_35715)
    
    # Testing the type of an if condition (line 68)
    if_condition_35717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_ne_35716)
    # Assigning a type to the variable 'if_condition_35717' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_35717', if_condition_35717)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 69)
    # Processing the call arguments (line 69)
    str_35719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'str', 'expected ndim to be 2, but observed %s')
    # Getting the type of 'A' (line 69)
    A_35720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 68), 'A', False)
    # Obtaining the member 'ndim' of a type (line 69)
    ndim_35721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 68), A_35720, 'ndim')
    # Applying the binary operator '%' (line 69)
    result_mod_35722 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '%', str_35719, ndim_35721)
    
    # Processing the call keyword arguments (line 69)
    kwargs_35723 = {}
    # Getting the type of 'ValueError' (line 69)
    ValueError_35718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 69)
    ValueError_call_result_35724 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), ValueError_35718, *[result_mod_35722], **kwargs_35723)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 8), ValueError_call_result_35724, 'raise parameter', BaseException)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 70)
    A_35725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 7), 'A')
    # Obtaining the member 'shape' of a type (line 70)
    shape_35726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 7), A_35725, 'shape')
    # Getting the type of 'B' (line 70)
    B_35727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'B')
    # Obtaining the member 'shape' of a type (line 70)
    shape_35728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 18), B_35727, 'shape')
    # Applying the binary operator '!=' (line 70)
    result_ne_35729 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 7), '!=', shape_35726, shape_35728)
    
    # Testing the type of an if condition (line 70)
    if_condition_35730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), result_ne_35729)
    # Assigning a type to the variable 'if_condition_35730' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_35730', if_condition_35730)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 71)
    # Processing the call arguments (line 71)
    str_35732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'str', 'the shapes of A and B differ (%s vs %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_35733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'A' (line 72)
    A_35734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'A', False)
    # Obtaining the member 'shape' of a type (line 72)
    shape_35735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), A_35734, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), tuple_35733, shape_35735)
    # Adding element type (line 72)
    # Getting the type of 'B' (line 72)
    B_35736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'B', False)
    # Obtaining the member 'shape' of a type (line 72)
    shape_35737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), B_35736, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), tuple_35733, shape_35737)
    
    # Applying the binary operator '%' (line 71)
    result_mod_35738 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 25), '%', str_35732, tuple_35733)
    
    # Processing the call keyword arguments (line 71)
    kwargs_35739 = {}
    # Getting the type of 'ValueError' (line 71)
    ValueError_35731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 71)
    ValueError_call_result_35740 = invoke(stypy.reporting.localization.Localization(__file__, 71, 14), ValueError_35731, *[result_mod_35738], **kwargs_35739)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 8), ValueError_call_result_35740, 'raise parameter', BaseException)
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 74):
    
    # Assigning a Subscript to a Name (line 74):
    
    # Obtaining the type of the subscript
    int_35741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')
    
    # Call to svd(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to dot(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'A' (line 74)
    A_35746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'A', False)
    # Processing the call keyword arguments (line 74)
    kwargs_35747 = {}
    # Getting the type of 'B' (line 74)
    B_35743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'B', False)
    # Obtaining the member 'T' of a type (line 74)
    T_35744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), B_35743, 'T')
    # Obtaining the member 'dot' of a type (line 74)
    dot_35745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), T_35744, 'dot')
    # Calling dot(args, kwargs) (line 74)
    dot_call_result_35748 = invoke(stypy.reporting.localization.Localization(__file__, 74, 19), dot_35745, *[A_35746], **kwargs_35747)
    
    # Obtaining the member 'T' of a type (line 74)
    T_35749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), dot_call_result_35748, 'T')
    # Processing the call keyword arguments (line 74)
    kwargs_35750 = {}
    # Getting the type of 'svd' (line 74)
    svd_35742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 74)
    svd_call_result_35751 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), svd_35742, *[T_35749], **kwargs_35750)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___35752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), svd_call_result_35751, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_35753 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), getitem___35752, int_35741)
    
    # Assigning a type to the variable 'tuple_var_assignment_35679' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35679', subscript_call_result_35753)
    
    # Assigning a Subscript to a Name (line 74):
    
    # Obtaining the type of the subscript
    int_35754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')
    
    # Call to svd(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to dot(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'A' (line 74)
    A_35759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'A', False)
    # Processing the call keyword arguments (line 74)
    kwargs_35760 = {}
    # Getting the type of 'B' (line 74)
    B_35756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'B', False)
    # Obtaining the member 'T' of a type (line 74)
    T_35757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), B_35756, 'T')
    # Obtaining the member 'dot' of a type (line 74)
    dot_35758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), T_35757, 'dot')
    # Calling dot(args, kwargs) (line 74)
    dot_call_result_35761 = invoke(stypy.reporting.localization.Localization(__file__, 74, 19), dot_35758, *[A_35759], **kwargs_35760)
    
    # Obtaining the member 'T' of a type (line 74)
    T_35762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), dot_call_result_35761, 'T')
    # Processing the call keyword arguments (line 74)
    kwargs_35763 = {}
    # Getting the type of 'svd' (line 74)
    svd_35755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 74)
    svd_call_result_35764 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), svd_35755, *[T_35762], **kwargs_35763)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___35765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), svd_call_result_35764, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_35766 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), getitem___35765, int_35754)
    
    # Assigning a type to the variable 'tuple_var_assignment_35680' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35680', subscript_call_result_35766)
    
    # Assigning a Subscript to a Name (line 74):
    
    # Obtaining the type of the subscript
    int_35767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'int')
    
    # Call to svd(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Call to dot(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'A' (line 74)
    A_35772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'A', False)
    # Processing the call keyword arguments (line 74)
    kwargs_35773 = {}
    # Getting the type of 'B' (line 74)
    B_35769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'B', False)
    # Obtaining the member 'T' of a type (line 74)
    T_35770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), B_35769, 'T')
    # Obtaining the member 'dot' of a type (line 74)
    dot_35771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), T_35770, 'dot')
    # Calling dot(args, kwargs) (line 74)
    dot_call_result_35774 = invoke(stypy.reporting.localization.Localization(__file__, 74, 19), dot_35771, *[A_35772], **kwargs_35773)
    
    # Obtaining the member 'T' of a type (line 74)
    T_35775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), dot_call_result_35774, 'T')
    # Processing the call keyword arguments (line 74)
    kwargs_35776 = {}
    # Getting the type of 'svd' (line 74)
    svd_35768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 74)
    svd_call_result_35777 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), svd_35768, *[T_35775], **kwargs_35776)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___35778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), svd_call_result_35777, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_35779 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), getitem___35778, int_35767)
    
    # Assigning a type to the variable 'tuple_var_assignment_35681' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35681', subscript_call_result_35779)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_var_assignment_35679' (line 74)
    tuple_var_assignment_35679_35780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35679')
    # Assigning a type to the variable 'u' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'u', tuple_var_assignment_35679_35780)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_var_assignment_35680' (line 74)
    tuple_var_assignment_35680_35781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35680')
    # Assigning a type to the variable 'w' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 7), 'w', tuple_var_assignment_35680_35781)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_var_assignment_35681' (line 74)
    tuple_var_assignment_35681_35782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_var_assignment_35681')
    # Assigning a type to the variable 'vt' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 10), 'vt', tuple_var_assignment_35681_35782)
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to dot(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'vt' (line 75)
    vt_35785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'vt', False)
    # Processing the call keyword arguments (line 75)
    kwargs_35786 = {}
    # Getting the type of 'u' (line 75)
    u_35783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'u', False)
    # Obtaining the member 'dot' of a type (line 75)
    dot_35784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), u_35783, 'dot')
    # Calling dot(args, kwargs) (line 75)
    dot_call_result_35787 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), dot_35784, *[vt_35785], **kwargs_35786)
    
    # Assigning a type to the variable 'R' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'R', dot_call_result_35787)
    
    # Assigning a Call to a Name (line 76):
    
    # Assigning a Call to a Name (line 76):
    
    # Call to sum(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_35790 = {}
    # Getting the type of 'w' (line 76)
    w_35788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'w', False)
    # Obtaining the member 'sum' of a type (line 76)
    sum_35789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), w_35788, 'sum')
    # Calling sum(args, kwargs) (line 76)
    sum_call_result_35791 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), sum_35789, *[], **kwargs_35790)
    
    # Assigning a type to the variable 'scale' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'scale', sum_call_result_35791)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_35792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'R' (line 77)
    R_35793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 11), tuple_35792, R_35793)
    # Adding element type (line 77)
    # Getting the type of 'scale' (line 77)
    scale_35794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'scale')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 11), tuple_35792, scale_35794)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', tuple_35792)
    
    # ################# End of 'orthogonal_procrustes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'orthogonal_procrustes' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_35795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'orthogonal_procrustes'
    return stypy_return_type_35795

# Assigning a type to the variable 'orthogonal_procrustes' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'orthogonal_procrustes', orthogonal_procrustes)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
