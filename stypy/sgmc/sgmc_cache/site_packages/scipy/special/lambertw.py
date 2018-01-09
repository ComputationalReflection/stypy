
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from ._ufuncs import _lambertw
4: 
5: 
6: def lambertw(z, k=0, tol=1e-8):
7:     r'''
8:     lambertw(z, k=0, tol=1e-8)
9: 
10:     Lambert W function.
11: 
12:     The Lambert W function `W(z)` is defined as the inverse function
13:     of ``w * exp(w)``. In other words, the value of ``W(z)`` is
14:     such that ``z = W(z) * exp(W(z))`` for any complex number
15:     ``z``.
16: 
17:     The Lambert W function is a multivalued function with infinitely
18:     many branches. Each branch gives a separate solution of the
19:     equation ``z = w exp(w)``. Here, the branches are indexed by the
20:     integer `k`.
21: 
22:     Parameters
23:     ----------
24:     z : array_like
25:         Input argument.
26:     k : int, optional
27:         Branch index.
28:     tol : float, optional
29:         Evaluation tolerance.
30: 
31:     Returns
32:     -------
33:     w : array
34:         `w` will have the same shape as `z`.
35: 
36:     Notes
37:     -----
38:     All branches are supported by `lambertw`:
39: 
40:     * ``lambertw(z)`` gives the principal solution (branch 0)
41:     * ``lambertw(z, k)`` gives the solution on branch `k`
42: 
43:     The Lambert W function has two partially real branches: the
44:     principal branch (`k = 0`) is real for real ``z > -1/e``, and the
45:     ``k = -1`` branch is real for ``-1/e < z < 0``. All branches except
46:     ``k = 0`` have a logarithmic singularity at ``z = 0``.
47: 
48:     **Possible issues**
49: 
50:     The evaluation can become inaccurate very close to the branch point
51:     at ``-1/e``. In some corner cases, `lambertw` might currently
52:     fail to converge, or can end up on the wrong branch.
53: 
54:     **Algorithm**
55: 
56:     Halley's iteration is used to invert ``w * exp(w)``, using a first-order
57:     asymptotic approximation (O(log(w)) or `O(w)`) as the initial estimate.
58: 
59:     The definition, implementation and choice of branches is based on [2]_.
60: 
61:     See Also
62:     --------
63:     wrightomega : the Wright Omega function
64: 
65:     References
66:     ----------
67:     .. [1] http://en.wikipedia.org/wiki/Lambert_W_function
68:     .. [2] Corless et al, "On the Lambert W function", Adv. Comp. Math. 5
69:        (1996) 329-359.
70:        http://www.apmaths.uwo.ca/~djeffrey/Offprints/W-adv-cm.pdf
71: 
72:     Examples
73:     --------
74:     The Lambert W function is the inverse of ``w exp(w)``:
75: 
76:     >>> from scipy.special import lambertw
77:     >>> w = lambertw(1)
78:     >>> w
79:     (0.56714329040978384+0j)
80:     >>> w * np.exp(w)
81:     (1.0+0j)
82: 
83:     Any branch gives a valid inverse:
84: 
85:     >>> w = lambertw(1, k=3)
86:     >>> w
87:     (-2.8535817554090377+17.113535539412148j)
88:     >>> w*np.exp(w)
89:     (1.0000000000000002+1.609823385706477e-15j)
90: 
91:     **Applications to equation-solving**
92: 
93:     The Lambert W function may be used to solve various kinds of
94:     equations, such as finding the value of the infinite power
95:     tower :math:`z^{z^{z^{\ldots}}}`:
96: 
97:     >>> def tower(z, n):
98:     ...     if n == 0:
99:     ...         return z
100:     ...     return z ** tower(z, n-1)
101:     ...
102:     >>> tower(0.5, 100)
103:     0.641185744504986
104:     >>> -lambertw(-np.log(0.5)) / np.log(0.5)
105:     (0.64118574450498589+0j)
106:     '''
107:     return _lambertw(z, k, tol)
108: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.special._ufuncs import _lambertw' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_498381 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs')

if (type(import_498381) is not StypyTypeError):

    if (import_498381 != 'pyd_module'):
        __import__(import_498381)
        sys_modules_498382 = sys.modules[import_498381]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', sys_modules_498382.module_type_store, module_type_store, ['_lambertw'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_498382, sys_modules_498382.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _lambertw

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', None, module_type_store, ['_lambertw'], [_lambertw])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special._ufuncs', import_498381)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


@norecursion
def lambertw(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_498383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
    float_498384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'float')
    defaults = [int_498383, float_498384]
    # Create a new context for function 'lambertw'
    module_type_store = module_type_store.open_function_context('lambertw', 6, 0, False)
    
    # Passed parameters checking function
    lambertw.stypy_localization = localization
    lambertw.stypy_type_of_self = None
    lambertw.stypy_type_store = module_type_store
    lambertw.stypy_function_name = 'lambertw'
    lambertw.stypy_param_names_list = ['z', 'k', 'tol']
    lambertw.stypy_varargs_param_name = None
    lambertw.stypy_kwargs_param_name = None
    lambertw.stypy_call_defaults = defaults
    lambertw.stypy_call_varargs = varargs
    lambertw.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lambertw', ['z', 'k', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lambertw', localization, ['z', 'k', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lambertw(...)' code ##################

    str_498385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', '\n    lambertw(z, k=0, tol=1e-8)\n\n    Lambert W function.\n\n    The Lambert W function `W(z)` is defined as the inverse function\n    of ``w * exp(w)``. In other words, the value of ``W(z)`` is\n    such that ``z = W(z) * exp(W(z))`` for any complex number\n    ``z``.\n\n    The Lambert W function is a multivalued function with infinitely\n    many branches. Each branch gives a separate solution of the\n    equation ``z = w exp(w)``. Here, the branches are indexed by the\n    integer `k`.\n\n    Parameters\n    ----------\n    z : array_like\n        Input argument.\n    k : int, optional\n        Branch index.\n    tol : float, optional\n        Evaluation tolerance.\n\n    Returns\n    -------\n    w : array\n        `w` will have the same shape as `z`.\n\n    Notes\n    -----\n    All branches are supported by `lambertw`:\n\n    * ``lambertw(z)`` gives the principal solution (branch 0)\n    * ``lambertw(z, k)`` gives the solution on branch `k`\n\n    The Lambert W function has two partially real branches: the\n    principal branch (`k = 0`) is real for real ``z > -1/e``, and the\n    ``k = -1`` branch is real for ``-1/e < z < 0``. All branches except\n    ``k = 0`` have a logarithmic singularity at ``z = 0``.\n\n    **Possible issues**\n\n    The evaluation can become inaccurate very close to the branch point\n    at ``-1/e``. In some corner cases, `lambertw` might currently\n    fail to converge, or can end up on the wrong branch.\n\n    **Algorithm**\n\n    Halley\'s iteration is used to invert ``w * exp(w)``, using a first-order\n    asymptotic approximation (O(log(w)) or `O(w)`) as the initial estimate.\n\n    The definition, implementation and choice of branches is based on [2]_.\n\n    See Also\n    --------\n    wrightomega : the Wright Omega function\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Lambert_W_function\n    .. [2] Corless et al, "On the Lambert W function", Adv. Comp. Math. 5\n       (1996) 329-359.\n       http://www.apmaths.uwo.ca/~djeffrey/Offprints/W-adv-cm.pdf\n\n    Examples\n    --------\n    The Lambert W function is the inverse of ``w exp(w)``:\n\n    >>> from scipy.special import lambertw\n    >>> w = lambertw(1)\n    >>> w\n    (0.56714329040978384+0j)\n    >>> w * np.exp(w)\n    (1.0+0j)\n\n    Any branch gives a valid inverse:\n\n    >>> w = lambertw(1, k=3)\n    >>> w\n    (-2.8535817554090377+17.113535539412148j)\n    >>> w*np.exp(w)\n    (1.0000000000000002+1.609823385706477e-15j)\n\n    **Applications to equation-solving**\n\n    The Lambert W function may be used to solve various kinds of\n    equations, such as finding the value of the infinite power\n    tower :math:`z^{z^{z^{\\ldots}}}`:\n\n    >>> def tower(z, n):\n    ...     if n == 0:\n    ...         return z\n    ...     return z ** tower(z, n-1)\n    ...\n    >>> tower(0.5, 100)\n    0.641185744504986\n    >>> -lambertw(-np.log(0.5)) / np.log(0.5)\n    (0.64118574450498589+0j)\n    ')
    
    # Call to _lambertw(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'z' (line 107)
    z_498387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'z', False)
    # Getting the type of 'k' (line 107)
    k_498388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'k', False)
    # Getting the type of 'tol' (line 107)
    tol_498389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'tol', False)
    # Processing the call keyword arguments (line 107)
    kwargs_498390 = {}
    # Getting the type of '_lambertw' (line 107)
    _lambertw_498386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), '_lambertw', False)
    # Calling _lambertw(args, kwargs) (line 107)
    _lambertw_call_result_498391 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), _lambertw_498386, *[z_498387, k_498388, tol_498389], **kwargs_498390)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', _lambertw_call_result_498391)
    
    # ################# End of 'lambertw(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lambertw' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_498392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_498392)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lambertw'
    return stypy_return_type_498392

# Assigning a type to the variable 'lambertw' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'lambertw', lambertw)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
