
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =============================================
3: Integration and ODEs (:mod:`scipy.integrate`)
4: =============================================
5: 
6: .. currentmodule:: scipy.integrate
7: 
8: Integrating functions, given function object
9: ============================================
10: 
11: .. autosummary::
12:    :toctree: generated/
13: 
14:    quad          -- General purpose integration
15:    dblquad       -- General purpose double integration
16:    tplquad       -- General purpose triple integration
17:    nquad         -- General purpose n-dimensional integration
18:    fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n
19:    quadrature    -- Integrate with given tolerance using Gaussian quadrature
20:    romberg       -- Integrate func using Romberg integration
21:    quad_explain  -- Print information for use of quad
22:    newton_cotes  -- Weights and error coefficient for Newton-Cotes integration
23:    IntegrationWarning -- Warning on issues during integration
24: 
25: Integrating functions, given fixed samples
26: ==========================================
27: 
28: .. autosummary::
29:    :toctree: generated/
30: 
31:    trapz         -- Use trapezoidal rule to compute integral.
32:    cumtrapz      -- Use trapezoidal rule to cumulatively compute integral.
33:    simps         -- Use Simpson's rule to compute integral from samples.
34:    romb          -- Use Romberg Integration to compute integral from
35:                  -- (2**k + 1) evenly-spaced samples.
36: 
37: .. seealso::
38: 
39:    :mod:`scipy.special` for orthogonal polynomials (special) for Gaussian
40:    quadrature roots and weights for other weighting factors and regions.
41: 
42: Solving initial value problems for ODE systems
43: ==============================================
44: 
45: The solvers are implemented as individual classes which can be used directly
46: (low-level usage) or through a convenience function.
47: 
48: .. autosummary::
49:    :toctree: generated/
50: 
51:    solve_ivp     -- Convenient function for ODE integration.
52:    RK23          -- Explicit Runge-Kutta solver of order 3(2).
53:    RK45          -- Explicit Runge-Kutta solver of order 5(4).
54:    Radau         -- Implicit Runge-Kutta solver of order 5.
55:    BDF           -- Implicit multi-step variable order (1 to 5) solver.
56:    LSODA         -- LSODA solver from ODEPACK Fortran package.
57:    OdeSolver     -- Base class for ODE solvers.
58:    DenseOutput   -- Local interpolant for computing a dense output.
59:    OdeSolution   -- Class which represents a continuous ODE solution.
60: 
61: 
62: Old API
63: -------
64: 
65: These are the routines developed earlier for scipy. They wrap older solvers
66: implemented in Fortran (mostly ODEPACK). While the interface to them is not
67: particularly convenient and certain features are missing compared to the new
68: API, the solvers themselves are of good quality and work fast as compiled
69: Fortran code. In some cases it might be worth using this old API.
70: 
71: .. autosummary::
72:    :toctree: generated/
73: 
74:    odeint        -- General integration of ordinary differential equations.
75:    ode           -- Integrate ODE using VODE and ZVODE routines.
76:    complex_ode   -- Convert a complex-valued ODE to real-valued and integrate.
77: 
78: 
79: Solving boundary value problems for ODE systems
80: ===============================================
81: 
82: .. autosummary::
83:    :toctree: generated/
84: 
85:    solve_bvp     -- Solve a boundary value problem for a system of ODEs.
86: '''
87: from __future__ import division, print_function, absolute_import
88: 
89: from .quadrature import *
90: from .odepack import *
91: from .quadpack import *
92: from ._ode import *
93: from ._bvp import solve_bvp
94: from ._ivp import (solve_ivp, OdeSolution, DenseOutput,
95:                    OdeSolver, RK23, RK45, Radau, BDF, LSODA)
96: 
97: __all__ = [s for s in dir() if not s.startswith('_')]
98: 
99: from scipy._lib._testutils import PytestTester
100: test = PytestTester(__name__)
101: del PytestTester
102: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_37968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', "\n=============================================\nIntegration and ODEs (:mod:`scipy.integrate`)\n=============================================\n\n.. currentmodule:: scipy.integrate\n\nIntegrating functions, given function object\n============================================\n\n.. autosummary::\n   :toctree: generated/\n\n   quad          -- General purpose integration\n   dblquad       -- General purpose double integration\n   tplquad       -- General purpose triple integration\n   nquad         -- General purpose n-dimensional integration\n   fixed_quad    -- Integrate func(x) using Gaussian quadrature of order n\n   quadrature    -- Integrate with given tolerance using Gaussian quadrature\n   romberg       -- Integrate func using Romberg integration\n   quad_explain  -- Print information for use of quad\n   newton_cotes  -- Weights and error coefficient for Newton-Cotes integration\n   IntegrationWarning -- Warning on issues during integration\n\nIntegrating functions, given fixed samples\n==========================================\n\n.. autosummary::\n   :toctree: generated/\n\n   trapz         -- Use trapezoidal rule to compute integral.\n   cumtrapz      -- Use trapezoidal rule to cumulatively compute integral.\n   simps         -- Use Simpson's rule to compute integral from samples.\n   romb          -- Use Romberg Integration to compute integral from\n                 -- (2**k + 1) evenly-spaced samples.\n\n.. seealso::\n\n   :mod:`scipy.special` for orthogonal polynomials (special) for Gaussian\n   quadrature roots and weights for other weighting factors and regions.\n\nSolving initial value problems for ODE systems\n==============================================\n\nThe solvers are implemented as individual classes which can be used directly\n(low-level usage) or through a convenience function.\n\n.. autosummary::\n   :toctree: generated/\n\n   solve_ivp     -- Convenient function for ODE integration.\n   RK23          -- Explicit Runge-Kutta solver of order 3(2).\n   RK45          -- Explicit Runge-Kutta solver of order 5(4).\n   Radau         -- Implicit Runge-Kutta solver of order 5.\n   BDF           -- Implicit multi-step variable order (1 to 5) solver.\n   LSODA         -- LSODA solver from ODEPACK Fortran package.\n   OdeSolver     -- Base class for ODE solvers.\n   DenseOutput   -- Local interpolant for computing a dense output.\n   OdeSolution   -- Class which represents a continuous ODE solution.\n\n\nOld API\n-------\n\nThese are the routines developed earlier for scipy. They wrap older solvers\nimplemented in Fortran (mostly ODEPACK). While the interface to them is not\nparticularly convenient and certain features are missing compared to the new\nAPI, the solvers themselves are of good quality and work fast as compiled\nFortran code. In some cases it might be worth using this old API.\n\n.. autosummary::\n   :toctree: generated/\n\n   odeint        -- General integration of ordinary differential equations.\n   ode           -- Integrate ODE using VODE and ZVODE routines.\n   complex_ode   -- Convert a complex-valued ODE to real-valued and integrate.\n\n\nSolving boundary value problems for ODE systems\n===============================================\n\n.. autosummary::\n   :toctree: generated/\n\n   solve_bvp     -- Solve a boundary value problem for a system of ODEs.\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 89, 0))

# 'from scipy.integrate.quadrature import ' statement (line 89)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37969 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy.integrate.quadrature')

if (type(import_37969) is not StypyTypeError):

    if (import_37969 != 'pyd_module'):
        __import__(import_37969)
        sys_modules_37970 = sys.modules[import_37969]
        import_from_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy.integrate.quadrature', sys_modules_37970.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 89, 0), __file__, sys_modules_37970, sys_modules_37970.module_type_store, module_type_store)
    else:
        from scipy.integrate.quadrature import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy.integrate.quadrature', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.integrate.quadrature' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'scipy.integrate.quadrature', import_37969)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 90, 0))

# 'from scipy.integrate.odepack import ' statement (line 90)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37971 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'scipy.integrate.odepack')

if (type(import_37971) is not StypyTypeError):

    if (import_37971 != 'pyd_module'):
        __import__(import_37971)
        sys_modules_37972 = sys.modules[import_37971]
        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'scipy.integrate.odepack', sys_modules_37972.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 90, 0), __file__, sys_modules_37972, sys_modules_37972.module_type_store, module_type_store)
    else:
        from scipy.integrate.odepack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 90, 0), 'scipy.integrate.odepack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.integrate.odepack' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'scipy.integrate.odepack', import_37971)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 0))

# 'from scipy.integrate.quadpack import ' statement (line 91)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37973 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.integrate.quadpack')

if (type(import_37973) is not StypyTypeError):

    if (import_37973 != 'pyd_module'):
        __import__(import_37973)
        sys_modules_37974 = sys.modules[import_37973]
        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.integrate.quadpack', sys_modules_37974.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 91, 0), __file__, sys_modules_37974, sys_modules_37974.module_type_store, module_type_store)
    else:
        from scipy.integrate.quadpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.integrate.quadpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.integrate.quadpack' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'scipy.integrate.quadpack', import_37973)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 0))

# 'from scipy.integrate._ode import ' statement (line 92)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37975 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate._ode')

if (type(import_37975) is not StypyTypeError):

    if (import_37975 != 'pyd_module'):
        __import__(import_37975)
        sys_modules_37976 = sys.modules[import_37975]
        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate._ode', sys_modules_37976.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 92, 0), __file__, sys_modules_37976, sys_modules_37976.module_type_store, module_type_store)
    else:
        from scipy.integrate._ode import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate._ode', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.integrate._ode' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'scipy.integrate._ode', import_37975)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 93, 0))

# 'from scipy.integrate._bvp import solve_bvp' statement (line 93)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37977 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate._bvp')

if (type(import_37977) is not StypyTypeError):

    if (import_37977 != 'pyd_module'):
        __import__(import_37977)
        sys_modules_37978 = sys.modules[import_37977]
        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate._bvp', sys_modules_37978.module_type_store, module_type_store, ['solve_bvp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 93, 0), __file__, sys_modules_37978, sys_modules_37978.module_type_store, module_type_store)
    else:
        from scipy.integrate._bvp import solve_bvp

        import_from_module(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate._bvp', None, module_type_store, ['solve_bvp'], [solve_bvp])

else:
    # Assigning a type to the variable 'scipy.integrate._bvp' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'scipy.integrate._bvp', import_37977)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 94, 0))

# 'from scipy.integrate._ivp import solve_ivp, OdeSolution, DenseOutput, OdeSolver, RK23, RK45, Radau, BDF, LSODA' statement (line 94)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37979 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate._ivp')

if (type(import_37979) is not StypyTypeError):

    if (import_37979 != 'pyd_module'):
        __import__(import_37979)
        sys_modules_37980 = sys.modules[import_37979]
        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate._ivp', sys_modules_37980.module_type_store, module_type_store, ['solve_ivp', 'OdeSolution', 'DenseOutput', 'OdeSolver', 'RK23', 'RK45', 'Radau', 'BDF', 'LSODA'])
        nest_module(stypy.reporting.localization.Localization(__file__, 94, 0), __file__, sys_modules_37980, sys_modules_37980.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp import solve_ivp, OdeSolution, DenseOutput, OdeSolver, RK23, RK45, Radau, BDF, LSODA

        import_from_module(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate._ivp', None, module_type_store, ['solve_ivp', 'OdeSolution', 'DenseOutput', 'OdeSolver', 'RK23', 'RK45', 'Radau', 'BDF', 'LSODA'], [solve_ivp, OdeSolution, DenseOutput, OdeSolver, RK23, RK45, Radau, BDF, LSODA])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'scipy.integrate._ivp', import_37979)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


# Assigning a ListComp to a Name (line 97):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 97)
# Processing the call keyword arguments (line 97)
kwargs_37989 = {}
# Getting the type of 'dir' (line 97)
dir_37988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'dir', False)
# Calling dir(args, kwargs) (line 97)
dir_call_result_37990 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), dir_37988, *[], **kwargs_37989)

comprehension_37991 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 11), dir_call_result_37990)
# Assigning a type to the variable 's' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 's', comprehension_37991)


# Call to startswith(...): (line 97)
# Processing the call arguments (line 97)
str_37984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 48), 'str', '_')
# Processing the call keyword arguments (line 97)
kwargs_37985 = {}
# Getting the type of 's' (line 97)
s_37982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 97)
startswith_37983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 35), s_37982, 'startswith')
# Calling startswith(args, kwargs) (line 97)
startswith_call_result_37986 = invoke(stypy.reporting.localization.Localization(__file__, 97, 35), startswith_37983, *[str_37984], **kwargs_37985)

# Applying the 'not' unary operator (line 97)
result_not__37987 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 31), 'not', startswith_call_result_37986)

# Getting the type of 's' (line 97)
s_37981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 's')
list_37992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 11), list_37992, s_37981)
# Assigning a type to the variable '__all__' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), '__all__', list_37992)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 99, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 99)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_37993 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy._lib._testutils')

if (type(import_37993) is not StypyTypeError):

    if (import_37993 != 'pyd_module'):
        __import__(import_37993)
        sys_modules_37994 = sys.modules[import_37993]
        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy._lib._testutils', sys_modules_37994.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 99, 0), __file__, sys_modules_37994, sys_modules_37994.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'scipy._lib._testutils', import_37993)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


# Assigning a Call to a Name (line 100):

# Call to PytestTester(...): (line 100)
# Processing the call arguments (line 100)
# Getting the type of '__name__' (line 100)
name___37996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), '__name__', False)
# Processing the call keyword arguments (line 100)
kwargs_37997 = {}
# Getting the type of 'PytestTester' (line 100)
PytestTester_37995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 100)
PytestTester_call_result_37998 = invoke(stypy.reporting.localization.Localization(__file__, 100, 7), PytestTester_37995, *[name___37996], **kwargs_37997)

# Assigning a type to the variable 'test' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'test', PytestTester_call_result_37998)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 101, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
