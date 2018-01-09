
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from . import _nnls
4: from numpy import asarray_chkfinite, zeros, double
5: 
6: __all__ = ['nnls']
7: 
8: 
9: def nnls(A, b):
10:     '''
11:     Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper
12:     for a FORTRAN non-negative least squares solver.
13: 
14:     Parameters
15:     ----------
16:     A : ndarray
17:         Matrix ``A`` as shown above.
18:     b : ndarray
19:         Right-hand side vector.
20: 
21:     Returns
22:     -------
23:     x : ndarray
24:         Solution vector.
25:     rnorm : float
26:         The residual, ``|| Ax-b ||_2``.
27: 
28:     Notes
29:     -----
30:     The FORTRAN code was published in the book below. The algorithm
31:     is an active set method. It solves the KKT (Karush-Kuhn-Tucker)
32:     conditions for the non-negative least squares problem.
33: 
34:     References
35:     ----------
36:     Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM
37: 
38:     '''
39: 
40:     A, b = map(asarray_chkfinite, (A, b))
41: 
42:     if len(A.shape) != 2:
43:         raise ValueError("expected matrix")
44:     if len(b.shape) != 1:
45:         raise ValueError("expected vector")
46: 
47:     m, n = A.shape
48: 
49:     if m != b.shape[0]:
50:         raise ValueError("incompatible dimensions")
51: 
52:     w = zeros((n,), dtype=double)
53:     zz = zeros((m,), dtype=double)
54:     index = zeros((n,), dtype=int)
55: 
56:     x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index)
57:     if mode != 1:
58:         raise RuntimeError("too many iterations")
59: 
60:     return x, rnorm
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.optimize import _nnls' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_172837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.optimize')

if (type(import_172837) is not StypyTypeError):

    if (import_172837 != 'pyd_module'):
        __import__(import_172837)
        sys_modules_172838 = sys.modules[import_172837]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.optimize', sys_modules_172838.module_type_store, module_type_store, ['_nnls'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_172838, sys_modules_172838.module_type_store, module_type_store)
    else:
        from scipy.optimize import _nnls

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.optimize', None, module_type_store, ['_nnls'], [_nnls])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.optimize', import_172837)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import asarray_chkfinite, zeros, double' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/')
import_172839 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_172839) is not StypyTypeError):

    if (import_172839 != 'pyd_module'):
        __import__(import_172839)
        sys_modules_172840 = sys.modules[import_172839]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_172840.module_type_store, module_type_store, ['asarray_chkfinite', 'zeros', 'double'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_172840, sys_modules_172840.module_type_store, module_type_store)
    else:
        from numpy import asarray_chkfinite, zeros, double

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['asarray_chkfinite', 'zeros', 'double'], [asarray_chkfinite, zeros, double])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_172839)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/')


# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['nnls']
module_type_store.set_exportable_members(['nnls'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_172841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_172842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'nnls')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_172841, str_172842)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_172841)

@norecursion
def nnls(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nnls'
    module_type_store = module_type_store.open_function_context('nnls', 9, 0, False)
    
    # Passed parameters checking function
    nnls.stypy_localization = localization
    nnls.stypy_type_of_self = None
    nnls.stypy_type_store = module_type_store
    nnls.stypy_function_name = 'nnls'
    nnls.stypy_param_names_list = ['A', 'b']
    nnls.stypy_varargs_param_name = None
    nnls.stypy_kwargs_param_name = None
    nnls.stypy_call_defaults = defaults
    nnls.stypy_call_varargs = varargs
    nnls.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nnls', ['A', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nnls', localization, ['A', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nnls(...)' code ##################

    str_172843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\n    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper\n    for a FORTRAN non-negative least squares solver.\n\n    Parameters\n    ----------\n    A : ndarray\n        Matrix ``A`` as shown above.\n    b : ndarray\n        Right-hand side vector.\n\n    Returns\n    -------\n    x : ndarray\n        Solution vector.\n    rnorm : float\n        The residual, ``|| Ax-b ||_2``.\n\n    Notes\n    -----\n    The FORTRAN code was published in the book below. The algorithm\n    is an active set method. It solves the KKT (Karush-Kuhn-Tucker)\n    conditions for the non-negative least squares problem.\n\n    References\n    ----------\n    Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM\n\n    ')
    
    # Assigning a Call to a Tuple (line 40):
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    int_172844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'int')
    
    # Call to map(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'asarray_chkfinite' (line 40)
    asarray_chkfinite_172846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'asarray_chkfinite', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_172847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'A' (line 40)
    A_172848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), tuple_172847, A_172848)
    # Adding element type (line 40)
    # Getting the type of 'b' (line 40)
    b_172849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), tuple_172847, b_172849)
    
    # Processing the call keyword arguments (line 40)
    kwargs_172850 = {}
    # Getting the type of 'map' (line 40)
    map_172845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'map', False)
    # Calling map(args, kwargs) (line 40)
    map_call_result_172851 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), map_172845, *[asarray_chkfinite_172846, tuple_172847], **kwargs_172850)
    
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___172852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), map_call_result_172851, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_172853 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), getitem___172852, int_172844)
    
    # Assigning a type to the variable 'tuple_var_assignment_172830' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_172830', subscript_call_result_172853)
    
    # Assigning a Subscript to a Name (line 40):
    
    # Obtaining the type of the subscript
    int_172854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'int')
    
    # Call to map(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'asarray_chkfinite' (line 40)
    asarray_chkfinite_172856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'asarray_chkfinite', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_172857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    # Getting the type of 'A' (line 40)
    A_172858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), tuple_172857, A_172858)
    # Adding element type (line 40)
    # Getting the type of 'b' (line 40)
    b_172859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), tuple_172857, b_172859)
    
    # Processing the call keyword arguments (line 40)
    kwargs_172860 = {}
    # Getting the type of 'map' (line 40)
    map_172855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'map', False)
    # Calling map(args, kwargs) (line 40)
    map_call_result_172861 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), map_172855, *[asarray_chkfinite_172856, tuple_172857], **kwargs_172860)
    
    # Obtaining the member '__getitem__' of a type (line 40)
    getitem___172862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), map_call_result_172861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
    subscript_call_result_172863 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), getitem___172862, int_172854)
    
    # Assigning a type to the variable 'tuple_var_assignment_172831' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_172831', subscript_call_result_172863)
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'tuple_var_assignment_172830' (line 40)
    tuple_var_assignment_172830_172864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_172830')
    # Assigning a type to the variable 'A' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'A', tuple_var_assignment_172830_172864)
    
    # Assigning a Name to a Name (line 40):
    # Getting the type of 'tuple_var_assignment_172831' (line 40)
    tuple_var_assignment_172831_172865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'tuple_var_assignment_172831')
    # Assigning a type to the variable 'b' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'b', tuple_var_assignment_172831_172865)
    
    
    
    # Call to len(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'A' (line 42)
    A_172867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 42)
    shape_172868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), A_172867, 'shape')
    # Processing the call keyword arguments (line 42)
    kwargs_172869 = {}
    # Getting the type of 'len' (line 42)
    len_172866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'len', False)
    # Calling len(args, kwargs) (line 42)
    len_call_result_172870 = invoke(stypy.reporting.localization.Localization(__file__, 42, 7), len_172866, *[shape_172868], **kwargs_172869)
    
    int_172871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
    # Applying the binary operator '!=' (line 42)
    result_ne_172872 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 7), '!=', len_call_result_172870, int_172871)
    
    # Testing the type of an if condition (line 42)
    if_condition_172873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), result_ne_172872)
    # Assigning a type to the variable 'if_condition_172873' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_172873', if_condition_172873)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 43)
    # Processing the call arguments (line 43)
    str_172875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 43)
    kwargs_172876 = {}
    # Getting the type of 'ValueError' (line 43)
    ValueError_172874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 43)
    ValueError_call_result_172877 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), ValueError_172874, *[str_172875], **kwargs_172876)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 8), ValueError_call_result_172877, 'raise parameter', BaseException)
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'b' (line 44)
    b_172879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'b', False)
    # Obtaining the member 'shape' of a type (line 44)
    shape_172880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), b_172879, 'shape')
    # Processing the call keyword arguments (line 44)
    kwargs_172881 = {}
    # Getting the type of 'len' (line 44)
    len_172878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'len', False)
    # Calling len(args, kwargs) (line 44)
    len_call_result_172882 = invoke(stypy.reporting.localization.Localization(__file__, 44, 7), len_172878, *[shape_172880], **kwargs_172881)
    
    int_172883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'int')
    # Applying the binary operator '!=' (line 44)
    result_ne_172884 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), '!=', len_call_result_172882, int_172883)
    
    # Testing the type of an if condition (line 44)
    if_condition_172885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_ne_172884)
    # Assigning a type to the variable 'if_condition_172885' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_172885', if_condition_172885)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 45)
    # Processing the call arguments (line 45)
    str_172887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'str', 'expected vector')
    # Processing the call keyword arguments (line 45)
    kwargs_172888 = {}
    # Getting the type of 'ValueError' (line 45)
    ValueError_172886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 45)
    ValueError_call_result_172889 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), ValueError_172886, *[str_172887], **kwargs_172888)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 45, 8), ValueError_call_result_172889, 'raise parameter', BaseException)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 47):
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_172890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'int')
    # Getting the type of 'A' (line 47)
    A_172891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'A')
    # Obtaining the member 'shape' of a type (line 47)
    shape_172892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), A_172891, 'shape')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___172893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), shape_172892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_172894 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), getitem___172893, int_172890)
    
    # Assigning a type to the variable 'tuple_var_assignment_172832' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_var_assignment_172832', subscript_call_result_172894)
    
    # Assigning a Subscript to a Name (line 47):
    
    # Obtaining the type of the subscript
    int_172895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'int')
    # Getting the type of 'A' (line 47)
    A_172896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'A')
    # Obtaining the member 'shape' of a type (line 47)
    shape_172897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), A_172896, 'shape')
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___172898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), shape_172897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_172899 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), getitem___172898, int_172895)
    
    # Assigning a type to the variable 'tuple_var_assignment_172833' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_var_assignment_172833', subscript_call_result_172899)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_var_assignment_172832' (line 47)
    tuple_var_assignment_172832_172900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_var_assignment_172832')
    # Assigning a type to the variable 'm' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'm', tuple_var_assignment_172832_172900)
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'tuple_var_assignment_172833' (line 47)
    tuple_var_assignment_172833_172901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'tuple_var_assignment_172833')
    # Assigning a type to the variable 'n' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'n', tuple_var_assignment_172833_172901)
    
    
    # Getting the type of 'm' (line 49)
    m_172902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'm')
    
    # Obtaining the type of the subscript
    int_172903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
    # Getting the type of 'b' (line 49)
    b_172904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'b')
    # Obtaining the member 'shape' of a type (line 49)
    shape_172905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), b_172904, 'shape')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___172906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), shape_172905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_172907 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), getitem___172906, int_172903)
    
    # Applying the binary operator '!=' (line 49)
    result_ne_172908 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), '!=', m_172902, subscript_call_result_172907)
    
    # Testing the type of an if condition (line 49)
    if_condition_172909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_ne_172908)
    # Assigning a type to the variable 'if_condition_172909' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_172909', if_condition_172909)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 50)
    # Processing the call arguments (line 50)
    str_172911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 25), 'str', 'incompatible dimensions')
    # Processing the call keyword arguments (line 50)
    kwargs_172912 = {}
    # Getting the type of 'ValueError' (line 50)
    ValueError_172910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 50)
    ValueError_call_result_172913 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), ValueError_172910, *[str_172911], **kwargs_172912)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 50, 8), ValueError_call_result_172913, 'raise parameter', BaseException)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to zeros(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining an instance of the builtin type 'tuple' (line 52)
    tuple_172915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 52)
    # Adding element type (line 52)
    # Getting the type of 'n' (line 52)
    n_172916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 15), tuple_172915, n_172916)
    
    # Processing the call keyword arguments (line 52)
    # Getting the type of 'double' (line 52)
    double_172917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'double', False)
    keyword_172918 = double_172917
    kwargs_172919 = {'dtype': keyword_172918}
    # Getting the type of 'zeros' (line 52)
    zeros_172914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 52)
    zeros_call_result_172920 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), zeros_172914, *[tuple_172915], **kwargs_172919)
    
    # Assigning a type to the variable 'w' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'w', zeros_call_result_172920)
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to zeros(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 53)
    tuple_172922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'm' (line 53)
    m_172923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), tuple_172922, m_172923)
    
    # Processing the call keyword arguments (line 53)
    # Getting the type of 'double' (line 53)
    double_172924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'double', False)
    keyword_172925 = double_172924
    kwargs_172926 = {'dtype': keyword_172925}
    # Getting the type of 'zeros' (line 53)
    zeros_172921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'zeros', False)
    # Calling zeros(args, kwargs) (line 53)
    zeros_call_result_172927 = invoke(stypy.reporting.localization.Localization(__file__, 53, 9), zeros_172921, *[tuple_172922], **kwargs_172926)
    
    # Assigning a type to the variable 'zz' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'zz', zeros_call_result_172927)
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to zeros(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_172929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'n' (line 54)
    n_172930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 19), tuple_172929, n_172930)
    
    # Processing the call keyword arguments (line 54)
    # Getting the type of 'int' (line 54)
    int_172931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 30), 'int', False)
    keyword_172932 = int_172931
    kwargs_172933 = {'dtype': keyword_172932}
    # Getting the type of 'zeros' (line 54)
    zeros_172928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'zeros', False)
    # Calling zeros(args, kwargs) (line 54)
    zeros_call_result_172934 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), zeros_172928, *[tuple_172929], **kwargs_172933)
    
    # Assigning a type to the variable 'index' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'index', zeros_call_result_172934)
    
    # Assigning a Call to a Tuple (line 56):
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_172935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to nnls(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A' (line 56)
    A_172938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'A', False)
    # Getting the type of 'm' (line 56)
    m_172939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'm', False)
    # Getting the type of 'n' (line 56)
    n_172940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'n', False)
    # Getting the type of 'b' (line 56)
    b_172941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'b', False)
    # Getting the type of 'w' (line 56)
    w_172942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'w', False)
    # Getting the type of 'zz' (line 56)
    zz_172943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'zz', False)
    # Getting the type of 'index' (line 56)
    index_172944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 51), 'index', False)
    # Processing the call keyword arguments (line 56)
    kwargs_172945 = {}
    # Getting the type of '_nnls' (line 56)
    _nnls_172936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), '_nnls', False)
    # Obtaining the member 'nnls' of a type (line 56)
    nnls_172937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), _nnls_172936, 'nnls')
    # Calling nnls(args, kwargs) (line 56)
    nnls_call_result_172946 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), nnls_172937, *[A_172938, m_172939, n_172940, b_172941, w_172942, zz_172943, index_172944], **kwargs_172945)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___172947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), nnls_call_result_172946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_172948 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___172947, int_172935)
    
    # Assigning a type to the variable 'tuple_var_assignment_172834' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172834', subscript_call_result_172948)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_172949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to nnls(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A' (line 56)
    A_172952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'A', False)
    # Getting the type of 'm' (line 56)
    m_172953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'm', False)
    # Getting the type of 'n' (line 56)
    n_172954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'n', False)
    # Getting the type of 'b' (line 56)
    b_172955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'b', False)
    # Getting the type of 'w' (line 56)
    w_172956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'w', False)
    # Getting the type of 'zz' (line 56)
    zz_172957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'zz', False)
    # Getting the type of 'index' (line 56)
    index_172958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 51), 'index', False)
    # Processing the call keyword arguments (line 56)
    kwargs_172959 = {}
    # Getting the type of '_nnls' (line 56)
    _nnls_172950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), '_nnls', False)
    # Obtaining the member 'nnls' of a type (line 56)
    nnls_172951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), _nnls_172950, 'nnls')
    # Calling nnls(args, kwargs) (line 56)
    nnls_call_result_172960 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), nnls_172951, *[A_172952, m_172953, n_172954, b_172955, w_172956, zz_172957, index_172958], **kwargs_172959)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___172961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), nnls_call_result_172960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_172962 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___172961, int_172949)
    
    # Assigning a type to the variable 'tuple_var_assignment_172835' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172835', subscript_call_result_172962)
    
    # Assigning a Subscript to a Name (line 56):
    
    # Obtaining the type of the subscript
    int_172963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'int')
    
    # Call to nnls(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'A' (line 56)
    A_172966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 32), 'A', False)
    # Getting the type of 'm' (line 56)
    m_172967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'm', False)
    # Getting the type of 'n' (line 56)
    n_172968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 38), 'n', False)
    # Getting the type of 'b' (line 56)
    b_172969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'b', False)
    # Getting the type of 'w' (line 56)
    w_172970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 44), 'w', False)
    # Getting the type of 'zz' (line 56)
    zz_172971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 47), 'zz', False)
    # Getting the type of 'index' (line 56)
    index_172972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 51), 'index', False)
    # Processing the call keyword arguments (line 56)
    kwargs_172973 = {}
    # Getting the type of '_nnls' (line 56)
    _nnls_172964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), '_nnls', False)
    # Obtaining the member 'nnls' of a type (line 56)
    nnls_172965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), _nnls_172964, 'nnls')
    # Calling nnls(args, kwargs) (line 56)
    nnls_call_result_172974 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), nnls_172965, *[A_172966, m_172967, n_172968, b_172969, w_172970, zz_172971, index_172972], **kwargs_172973)
    
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___172975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), nnls_call_result_172974, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_172976 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), getitem___172975, int_172963)
    
    # Assigning a type to the variable 'tuple_var_assignment_172836' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172836', subscript_call_result_172976)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_172834' (line 56)
    tuple_var_assignment_172834_172977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172834')
    # Assigning a type to the variable 'x' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'x', tuple_var_assignment_172834_172977)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_172835' (line 56)
    tuple_var_assignment_172835_172978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172835')
    # Assigning a type to the variable 'rnorm' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'rnorm', tuple_var_assignment_172835_172978)
    
    # Assigning a Name to a Name (line 56):
    # Getting the type of 'tuple_var_assignment_172836' (line 56)
    tuple_var_assignment_172836_172979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'tuple_var_assignment_172836')
    # Assigning a type to the variable 'mode' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'mode', tuple_var_assignment_172836_172979)
    
    
    # Getting the type of 'mode' (line 57)
    mode_172980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'mode')
    int_172981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'int')
    # Applying the binary operator '!=' (line 57)
    result_ne_172982 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '!=', mode_172980, int_172981)
    
    # Testing the type of an if condition (line 57)
    if_condition_172983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_ne_172982)
    # Assigning a type to the variable 'if_condition_172983' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_172983', if_condition_172983)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 58)
    # Processing the call arguments (line 58)
    str_172985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'str', 'too many iterations')
    # Processing the call keyword arguments (line 58)
    kwargs_172986 = {}
    # Getting the type of 'RuntimeError' (line 58)
    RuntimeError_172984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 58)
    RuntimeError_call_result_172987 = invoke(stypy.reporting.localization.Localization(__file__, 58, 14), RuntimeError_172984, *[str_172985], **kwargs_172986)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 58, 8), RuntimeError_call_result_172987, 'raise parameter', BaseException)
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_172988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'x' (line 60)
    x_172989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), tuple_172988, x_172989)
    # Adding element type (line 60)
    # Getting the type of 'rnorm' (line 60)
    rnorm_172990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'rnorm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 11), tuple_172988, rnorm_172990)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', tuple_172988)
    
    # ################# End of 'nnls(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nnls' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_172991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172991)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nnls'
    return stypy_return_type_172991

# Assigning a type to the variable 'nnls' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'nnls', nnls)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
