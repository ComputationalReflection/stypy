
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: import pytest
5: 
6: from scipy.special._testutils import MissingModule, check_version
7: from scipy.special._mptestutils import (
8:     Arg, IntArg, mp_assert_allclose, assert_mpmath_equal)
9: from scipy.special._precompute.gammainc_asy import (
10:     compute_g, compute_alpha, compute_d)
11: from scipy.special._precompute.gammainc_data import gammainc, gammaincc
12: 
13: try:
14:     import sympy
15: except ImportError:
16:     sympy = MissingModule('sympy')
17: 
18: try:
19:     import mpmath as mp
20: except ImportError:
21:     mp = MissingModule('mpmath')
22: 
23: 
24: _is_32bit_platform = np.intp(0).itemsize < 8
25: 
26: 
27: @check_version(mp, '0.19')
28: def test_g():
29:     # Test data for the g_k. See DLMF 5.11.4.
30:     with mp.workdps(30):
31:         g = [mp.mpf(1), mp.mpf(1)/12, mp.mpf(1)/288,
32:              -mp.mpf(139)/51840, -mp.mpf(571)/2488320,
33:              mp.mpf(163879)/209018880, mp.mpf(5246819)/75246796800]
34:         mp_assert_allclose(compute_g(7), g)
35: 
36: 
37: @pytest.mark.slow
38: @check_version(mp, '0.19')
39: @check_version(sympy, '0.7')
40: @pytest.mark.xfail(condition=_is_32bit_platform, reason="rtol only 2e-11, see gh-6938")
41: def test_alpha():
42:     # Test data for the alpha_k. See DLMF 8.12.14.
43:     with mp.workdps(30):
44:         alpha = [mp.mpf(0), mp.mpf(1), mp.mpf(1)/3, mp.mpf(1)/36,
45:                  -mp.mpf(1)/270, mp.mpf(1)/4320, mp.mpf(1)/17010,
46:                  -mp.mpf(139)/5443200, mp.mpf(1)/204120]
47:         mp_assert_allclose(compute_alpha(9), alpha)
48: 
49: 
50: @pytest.mark.xslow
51: @check_version(mp, '0.19')
52: @check_version(sympy, '0.7')
53: def test_d():
54:     # Compare the d_{k, n} to the results in appendix F of [1].
55:     #
56:     # Sources
57:     # -------
58:     # [1] DiDonato and Morris, Computation of the Incomplete Gamma
59:     #     Function Ratios and their Inverse, ACM Transactions on
60:     #     Mathematical Software, 1986.
61: 
62:     with mp.workdps(50):
63:         dataset = [(0, 0, -mp.mpf('0.333333333333333333333333333333')),
64:                    (0, 12, mp.mpf('0.102618097842403080425739573227e-7')),
65:                    (1, 0, -mp.mpf('0.185185185185185185185185185185e-2')),
66:                    (1, 12, mp.mpf('0.119516285997781473243076536700e-7')),
67:                    (2, 0, mp.mpf('0.413359788359788359788359788360e-2')),
68:                    (2, 12, -mp.mpf('0.140925299108675210532930244154e-7')),
69:                    (3, 0, mp.mpf('0.649434156378600823045267489712e-3')),
70:                    (3, 12, -mp.mpf('0.191111684859736540606728140873e-7')),
71:                    (4, 0, -mp.mpf('0.861888290916711698604702719929e-3')),
72:                    (4, 12, mp.mpf('0.288658297427087836297341274604e-7')),
73:                    (5, 0, -mp.mpf('0.336798553366358150308767592718e-3')),
74:                    (5, 12, mp.mpf('0.482409670378941807563762631739e-7')),
75:                    (6, 0, mp.mpf('0.531307936463992223165748542978e-3')),
76:                    (6, 12, -mp.mpf('0.882860074633048352505085243179e-7')),
77:                    (7, 0, mp.mpf('0.344367606892377671254279625109e-3')),
78:                    (7, 12, -mp.mpf('0.175629733590604619378669693914e-6')),
79:                    (8, 0, -mp.mpf('0.652623918595309418922034919727e-3')),
80:                    (8, 12, mp.mpf('0.377358774161109793380344937299e-6')),
81:                    (9, 0, -mp.mpf('0.596761290192746250124390067179e-3')),
82:                    (9, 12, mp.mpf('0.870823417786464116761231237189e-6'))]
83:         d = compute_d(10, 13)
84:         res = []
85:         for k, n, std in dataset:
86:             res.append(d[k][n])
87:         std = map(lambda x: x[2], dataset)
88:         mp_assert_allclose(res, std)
89: 
90: 
91: @check_version(mp, '0.19')
92: def test_gammainc():
93:     # Quick check that the gammainc in
94:     # special._precompute.gammainc_data agrees with mpmath's
95:     # gammainc.
96:     assert_mpmath_equal(gammainc,
97:                         lambda a, x: mp.gammainc(a, b=x, regularized=True),
98:                         [Arg(0, 100, inclusive_a=False), Arg(0, 100)],
99:                         nan_ok=False, rtol=1e-17, n=50, dps=50)
100: 
101: 
102: @check_version(mp, '0.19')
103: def test_gammaincc():
104:     # Quick check that the gammaincc in
105:     # special._precompute.gammainc_data agrees with mpmath's
106:     # gammainc.
107:     assert_mpmath_equal(lambda a, x: gammaincc(a, x, dps=1000),
108:                         lambda a, x: mp.gammainc(a, a=x, regularized=True),
109:                         [Arg(20, 100), Arg(20, 100)],
110:                         nan_ok=False, rtol=1e-17, n=50, dps=1000)
111: 
112:     # Test the fast integer path
113:     assert_mpmath_equal(gammaincc,
114:                         lambda a, x: mp.gammainc(a, a=x, regularized=True),
115:                         [IntArg(1, 100), Arg(0, 100)],
116:                         nan_ok=False, rtol=1e-17, n=50, dps=50)
117: 
118: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559164 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_559164) is not StypyTypeError):

    if (import_559164 != 'pyd_module'):
        __import__(import_559164)
        sys_modules_559165 = sys.modules[import_559164]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_559165.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_559164)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import pytest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_559166) is not StypyTypeError):

    if (import_559166 != 'pyd_module'):
        __import__(import_559166)
        sys_modules_559167 = sys.modules[import_559166]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_559167.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_559166)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._testutils import MissingModule, check_version' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559168 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils')

if (type(import_559168) is not StypyTypeError):

    if (import_559168 != 'pyd_module'):
        __import__(import_559168)
        sys_modules_559169 = sys.modules[import_559168]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', sys_modules_559169.module_type_store, module_type_store, ['MissingModule', 'check_version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_559169, sys_modules_559169.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import MissingModule, check_version

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', None, module_type_store, ['MissingModule', 'check_version'], [MissingModule, check_version])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', import_559168)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._mptestutils import Arg, IntArg, mp_assert_allclose, assert_mpmath_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559170 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._mptestutils')

if (type(import_559170) is not StypyTypeError):

    if (import_559170 != 'pyd_module'):
        __import__(import_559170)
        sys_modules_559171 = sys.modules[import_559170]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._mptestutils', sys_modules_559171.module_type_store, module_type_store, ['Arg', 'IntArg', 'mp_assert_allclose', 'assert_mpmath_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_559171, sys_modules_559171.module_type_store, module_type_store)
    else:
        from scipy.special._mptestutils import Arg, IntArg, mp_assert_allclose, assert_mpmath_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._mptestutils', None, module_type_store, ['Arg', 'IntArg', 'mp_assert_allclose', 'assert_mpmath_equal'], [Arg, IntArg, mp_assert_allclose, assert_mpmath_equal])

else:
    # Assigning a type to the variable 'scipy.special._mptestutils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._mptestutils', import_559170)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.special._precompute.gammainc_asy import compute_g, compute_alpha, compute_d' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559172 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special._precompute.gammainc_asy')

if (type(import_559172) is not StypyTypeError):

    if (import_559172 != 'pyd_module'):
        __import__(import_559172)
        sys_modules_559173 = sys.modules[import_559172]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special._precompute.gammainc_asy', sys_modules_559173.module_type_store, module_type_store, ['compute_g', 'compute_alpha', 'compute_d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_559173, sys_modules_559173.module_type_store, module_type_store)
    else:
        from scipy.special._precompute.gammainc_asy import compute_g, compute_alpha, compute_d

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special._precompute.gammainc_asy', None, module_type_store, ['compute_g', 'compute_alpha', 'compute_d'], [compute_g, compute_alpha, compute_d])

else:
    # Assigning a type to the variable 'scipy.special._precompute.gammainc_asy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special._precompute.gammainc_asy', import_559172)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.special._precompute.gammainc_data import gammainc, gammaincc' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._precompute.gammainc_data')

if (type(import_559174) is not StypyTypeError):

    if (import_559174 != 'pyd_module'):
        __import__(import_559174)
        sys_modules_559175 = sys.modules[import_559174]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._precompute.gammainc_data', sys_modules_559175.module_type_store, module_type_store, ['gammainc', 'gammaincc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_559175, sys_modules_559175.module_type_store, module_type_store)
    else:
        from scipy.special._precompute.gammainc_data import gammainc, gammaincc

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._precompute.gammainc_data', None, module_type_store, ['gammainc', 'gammaincc'], [gammainc, gammaincc])

else:
    # Assigning a type to the variable 'scipy.special._precompute.gammainc_data' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._precompute.gammainc_data', import_559174)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')



# SSA begins for try-except statement (line 13)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))

# 'import sympy' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559176 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'sympy')

if (type(import_559176) is not StypyTypeError):

    if (import_559176 != 'pyd_module'):
        __import__(import_559176)
        sys_modules_559177 = sys.modules[import_559176]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'sympy', sys_modules_559177.module_type_store, module_type_store)
    else:
        import sympy

        import_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'sympy', sympy, module_type_store)

else:
    # Assigning a type to the variable 'sympy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'sympy', import_559176)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 13)
# SSA branch for the except 'ImportError' branch of a try statement (line 13)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 16):

# Call to MissingModule(...): (line 16)
# Processing the call arguments (line 16)
str_559179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'sympy')
# Processing the call keyword arguments (line 16)
kwargs_559180 = {}
# Getting the type of 'MissingModule' (line 16)
MissingModule_559178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 16)
MissingModule_call_result_559181 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), MissingModule_559178, *[str_559179], **kwargs_559180)

# Assigning a type to the variable 'sympy' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'sympy', MissingModule_call_result_559181)
# SSA join for try-except statement (line 13)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 18)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'import mpmath' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559182 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'mpmath')

if (type(import_559182) is not StypyTypeError):

    if (import_559182 != 'pyd_module'):
        __import__(import_559182)
        sys_modules_559183 = sys.modules[import_559182]
        import_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'mp', sys_modules_559183.module_type_store, module_type_store)
    else:
        import mpmath as mp

        import_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'mp', mpmath, module_type_store)

else:
    # Assigning a type to the variable 'mpmath' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'mpmath', import_559182)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# SSA branch for the except part of a try statement (line 18)
# SSA branch for the except 'ImportError' branch of a try statement (line 18)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 21):

# Call to MissingModule(...): (line 21)
# Processing the call arguments (line 21)
str_559185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'str', 'mpmath')
# Processing the call keyword arguments (line 21)
kwargs_559186 = {}
# Getting the type of 'MissingModule' (line 21)
MissingModule_559184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'MissingModule', False)
# Calling MissingModule(args, kwargs) (line 21)
MissingModule_call_result_559187 = invoke(stypy.reporting.localization.Localization(__file__, 21, 9), MissingModule_559184, *[str_559185], **kwargs_559186)

# Assigning a type to the variable 'mp' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'mp', MissingModule_call_result_559187)
# SSA join for try-except statement (line 18)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Compare to a Name (line 24):


# Call to intp(...): (line 24)
# Processing the call arguments (line 24)
int_559190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
# Processing the call keyword arguments (line 24)
kwargs_559191 = {}
# Getting the type of 'np' (line 24)
np_559188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'np', False)
# Obtaining the member 'intp' of a type (line 24)
intp_559189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 21), np_559188, 'intp')
# Calling intp(args, kwargs) (line 24)
intp_call_result_559192 = invoke(stypy.reporting.localization.Localization(__file__, 24, 21), intp_559189, *[int_559190], **kwargs_559191)

# Obtaining the member 'itemsize' of a type (line 24)
itemsize_559193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 21), intp_call_result_559192, 'itemsize')
int_559194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'int')
# Applying the binary operator '<' (line 24)
result_lt_559195 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), '<', itemsize_559193, int_559194)

# Assigning a type to the variable '_is_32bit_platform' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_is_32bit_platform', result_lt_559195)

@norecursion
def test_g(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_g'
    module_type_store = module_type_store.open_function_context('test_g', 27, 0, False)
    
    # Passed parameters checking function
    test_g.stypy_localization = localization
    test_g.stypy_type_of_self = None
    test_g.stypy_type_store = module_type_store
    test_g.stypy_function_name = 'test_g'
    test_g.stypy_param_names_list = []
    test_g.stypy_varargs_param_name = None
    test_g.stypy_kwargs_param_name = None
    test_g.stypy_call_defaults = defaults
    test_g.stypy_call_varargs = varargs
    test_g.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_g', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_g', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_g(...)' code ##################

    
    # Call to workdps(...): (line 30)
    # Processing the call arguments (line 30)
    int_559198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'int')
    # Processing the call keyword arguments (line 30)
    kwargs_559199 = {}
    # Getting the type of 'mp' (line 30)
    mp_559196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'mp', False)
    # Obtaining the member 'workdps' of a type (line 30)
    workdps_559197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), mp_559196, 'workdps')
    # Calling workdps(args, kwargs) (line 30)
    workdps_call_result_559200 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), workdps_559197, *[int_559198], **kwargs_559199)
    
    with_559201 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 30, 9), workdps_call_result_559200, 'with parameter', '__enter__', '__exit__')

    if with_559201:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 30)
        enter___559202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), workdps_call_result_559200, '__enter__')
        with_enter_559203 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), enter___559202)
        
        # Assigning a List to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_559204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        
        # Call to mpf(...): (line 31)
        # Processing the call arguments (line 31)
        int_559207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_559208 = {}
        # Getting the type of 'mp' (line 31)
        mp_559205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 31)
        mpf_559206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), mp_559205, 'mpf')
        # Calling mpf(args, kwargs) (line 31)
        mpf_call_result_559209 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), mpf_559206, *[int_559207], **kwargs_559208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, mpf_call_result_559209)
        # Adding element type (line 31)
        
        # Call to mpf(...): (line 31)
        # Processing the call arguments (line 31)
        int_559212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_559213 = {}
        # Getting the type of 'mp' (line 31)
        mp_559210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 31)
        mpf_559211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 24), mp_559210, 'mpf')
        # Calling mpf(args, kwargs) (line 31)
        mpf_call_result_559214 = invoke(stypy.reporting.localization.Localization(__file__, 31, 24), mpf_559211, *[int_559212], **kwargs_559213)
        
        int_559215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 34), 'int')
        # Applying the binary operator 'div' (line 31)
        result_div_559216 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 24), 'div', mpf_call_result_559214, int_559215)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559216)
        # Adding element type (line 31)
        
        # Call to mpf(...): (line 31)
        # Processing the call arguments (line 31)
        int_559219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 45), 'int')
        # Processing the call keyword arguments (line 31)
        kwargs_559220 = {}
        # Getting the type of 'mp' (line 31)
        mp_559217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 38), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 31)
        mpf_559218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 38), mp_559217, 'mpf')
        # Calling mpf(args, kwargs) (line 31)
        mpf_call_result_559221 = invoke(stypy.reporting.localization.Localization(__file__, 31, 38), mpf_559218, *[int_559219], **kwargs_559220)
        
        int_559222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'int')
        # Applying the binary operator 'div' (line 31)
        result_div_559223 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 38), 'div', mpf_call_result_559221, int_559222)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559223)
        # Adding element type (line 31)
        
        
        # Call to mpf(...): (line 32)
        # Processing the call arguments (line 32)
        int_559226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'int')
        # Processing the call keyword arguments (line 32)
        kwargs_559227 = {}
        # Getting the type of 'mp' (line 32)
        mp_559224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 32)
        mpf_559225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 14), mp_559224, 'mpf')
        # Calling mpf(args, kwargs) (line 32)
        mpf_call_result_559228 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), mpf_559225, *[int_559226], **kwargs_559227)
        
        # Applying the 'usub' unary operator (line 32)
        result___neg___559229 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 13), 'usub', mpf_call_result_559228)
        
        int_559230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 26), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_559231 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 13), 'div', result___neg___559229, int_559230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559231)
        # Adding element type (line 31)
        
        
        # Call to mpf(...): (line 32)
        # Processing the call arguments (line 32)
        int_559234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 41), 'int')
        # Processing the call keyword arguments (line 32)
        kwargs_559235 = {}
        # Getting the type of 'mp' (line 32)
        mp_559232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 32)
        mpf_559233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 34), mp_559232, 'mpf')
        # Calling mpf(args, kwargs) (line 32)
        mpf_call_result_559236 = invoke(stypy.reporting.localization.Localization(__file__, 32, 34), mpf_559233, *[int_559234], **kwargs_559235)
        
        # Applying the 'usub' unary operator (line 32)
        result___neg___559237 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 33), 'usub', mpf_call_result_559236)
        
        int_559238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 46), 'int')
        # Applying the binary operator 'div' (line 32)
        result_div_559239 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 33), 'div', result___neg___559237, int_559238)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559239)
        # Adding element type (line 31)
        
        # Call to mpf(...): (line 33)
        # Processing the call arguments (line 33)
        int_559242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_559243 = {}
        # Getting the type of 'mp' (line 33)
        mp_559240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 33)
        mpf_559241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), mp_559240, 'mpf')
        # Calling mpf(args, kwargs) (line 33)
        mpf_call_result_559244 = invoke(stypy.reporting.localization.Localization(__file__, 33, 13), mpf_559241, *[int_559242], **kwargs_559243)
        
        int_559245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
        # Applying the binary operator 'div' (line 33)
        result_div_559246 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), 'div', mpf_call_result_559244, int_559245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559246)
        # Adding element type (line 31)
        
        # Call to mpf(...): (line 33)
        # Processing the call arguments (line 33)
        int_559249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 46), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_559250 = {}
        # Getting the type of 'mp' (line 33)
        mp_559247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 33)
        mpf_559248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 39), mp_559247, 'mpf')
        # Calling mpf(args, kwargs) (line 33)
        mpf_call_result_559251 = invoke(stypy.reporting.localization.Localization(__file__, 33, 39), mpf_559248, *[int_559249], **kwargs_559250)
        
        long_559252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'long')
        # Applying the binary operator 'div' (line 33)
        result_div_559253 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 39), 'div', mpf_call_result_559251, long_559252)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 12), list_559204, result_div_559253)
        
        # Assigning a type to the variable 'g' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'g', list_559204)
        
        # Call to mp_assert_allclose(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to compute_g(...): (line 34)
        # Processing the call arguments (line 34)
        int_559256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 37), 'int')
        # Processing the call keyword arguments (line 34)
        kwargs_559257 = {}
        # Getting the type of 'compute_g' (line 34)
        compute_g_559255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'compute_g', False)
        # Calling compute_g(args, kwargs) (line 34)
        compute_g_call_result_559258 = invoke(stypy.reporting.localization.Localization(__file__, 34, 27), compute_g_559255, *[int_559256], **kwargs_559257)
        
        # Getting the type of 'g' (line 34)
        g_559259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'g', False)
        # Processing the call keyword arguments (line 34)
        kwargs_559260 = {}
        # Getting the type of 'mp_assert_allclose' (line 34)
        mp_assert_allclose_559254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'mp_assert_allclose', False)
        # Calling mp_assert_allclose(args, kwargs) (line 34)
        mp_assert_allclose_call_result_559261 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), mp_assert_allclose_559254, *[compute_g_call_result_559258, g_559259], **kwargs_559260)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 30)
        exit___559262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 9), workdps_call_result_559200, '__exit__')
        with_exit_559263 = invoke(stypy.reporting.localization.Localization(__file__, 30, 9), exit___559262, None, None, None)

    
    # ################# End of 'test_g(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_g' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_559264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559264)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_g'
    return stypy_return_type_559264

# Assigning a type to the variable 'test_g' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'test_g', test_g)

@norecursion
def test_alpha(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_alpha'
    module_type_store = module_type_store.open_function_context('test_alpha', 37, 0, False)
    
    # Passed parameters checking function
    test_alpha.stypy_localization = localization
    test_alpha.stypy_type_of_self = None
    test_alpha.stypy_type_store = module_type_store
    test_alpha.stypy_function_name = 'test_alpha'
    test_alpha.stypy_param_names_list = []
    test_alpha.stypy_varargs_param_name = None
    test_alpha.stypy_kwargs_param_name = None
    test_alpha.stypy_call_defaults = defaults
    test_alpha.stypy_call_varargs = varargs
    test_alpha.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_alpha', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_alpha', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_alpha(...)' code ##################

    
    # Call to workdps(...): (line 43)
    # Processing the call arguments (line 43)
    int_559267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_559268 = {}
    # Getting the type of 'mp' (line 43)
    mp_559265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 'mp', False)
    # Obtaining the member 'workdps' of a type (line 43)
    workdps_559266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 9), mp_559265, 'workdps')
    # Calling workdps(args, kwargs) (line 43)
    workdps_call_result_559269 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), workdps_559266, *[int_559267], **kwargs_559268)
    
    with_559270 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 43, 9), workdps_call_result_559269, 'with parameter', '__enter__', '__exit__')

    if with_559270:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 43)
        enter___559271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 9), workdps_call_result_559269, '__enter__')
        with_enter_559272 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), enter___559271)
        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_559273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 44)
        # Processing the call arguments (line 44)
        int_559276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_559277 = {}
        # Getting the type of 'mp' (line 44)
        mp_559274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 44)
        mpf_559275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 17), mp_559274, 'mpf')
        # Calling mpf(args, kwargs) (line 44)
        mpf_call_result_559278 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), mpf_559275, *[int_559276], **kwargs_559277)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, mpf_call_result_559278)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 44)
        # Processing the call arguments (line 44)
        int_559281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_559282 = {}
        # Getting the type of 'mp' (line 44)
        mp_559279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 44)
        mpf_559280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 28), mp_559279, 'mpf')
        # Calling mpf(args, kwargs) (line 44)
        mpf_call_result_559283 = invoke(stypy.reporting.localization.Localization(__file__, 44, 28), mpf_559280, *[int_559281], **kwargs_559282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, mpf_call_result_559283)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 44)
        # Processing the call arguments (line 44)
        int_559286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_559287 = {}
        # Getting the type of 'mp' (line 44)
        mp_559284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 44)
        mpf_559285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 39), mp_559284, 'mpf')
        # Calling mpf(args, kwargs) (line 44)
        mpf_call_result_559288 = invoke(stypy.reporting.localization.Localization(__file__, 44, 39), mpf_559285, *[int_559286], **kwargs_559287)
        
        int_559289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 49), 'int')
        # Applying the binary operator 'div' (line 44)
        result_div_559290 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 39), 'div', mpf_call_result_559288, int_559289)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559290)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 44)
        # Processing the call arguments (line 44)
        int_559293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 59), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_559294 = {}
        # Getting the type of 'mp' (line 44)
        mp_559291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 52), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 44)
        mpf_559292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 52), mp_559291, 'mpf')
        # Calling mpf(args, kwargs) (line 44)
        mpf_call_result_559295 = invoke(stypy.reporting.localization.Localization(__file__, 44, 52), mpf_559292, *[int_559293], **kwargs_559294)
        
        int_559296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 62), 'int')
        # Applying the binary operator 'div' (line 44)
        result_div_559297 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 52), 'div', mpf_call_result_559295, int_559296)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559297)
        # Adding element type (line 44)
        
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        int_559300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_559301 = {}
        # Getting the type of 'mp' (line 45)
        mp_559298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_559299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 18), mp_559298, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_559302 = invoke(stypy.reporting.localization.Localization(__file__, 45, 18), mpf_559299, *[int_559300], **kwargs_559301)
        
        # Applying the 'usub' unary operator (line 45)
        result___neg___559303 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 17), 'usub', mpf_call_result_559302)
        
        int_559304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'int')
        # Applying the binary operator 'div' (line 45)
        result_div_559305 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 17), 'div', result___neg___559303, int_559304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559305)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        int_559308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 40), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_559309 = {}
        # Getting the type of 'mp' (line 45)
        mp_559306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_559307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 33), mp_559306, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_559310 = invoke(stypy.reporting.localization.Localization(__file__, 45, 33), mpf_559307, *[int_559308], **kwargs_559309)
        
        int_559311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'int')
        # Applying the binary operator 'div' (line 45)
        result_div_559312 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 33), 'div', mpf_call_result_559310, int_559311)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559312)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 45)
        # Processing the call arguments (line 45)
        int_559315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 56), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_559316 = {}
        # Getting the type of 'mp' (line 45)
        mp_559313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 49), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 45)
        mpf_559314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 49), mp_559313, 'mpf')
        # Calling mpf(args, kwargs) (line 45)
        mpf_call_result_559317 = invoke(stypy.reporting.localization.Localization(__file__, 45, 49), mpf_559314, *[int_559315], **kwargs_559316)
        
        int_559318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 59), 'int')
        # Applying the binary operator 'div' (line 45)
        result_div_559319 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 49), 'div', mpf_call_result_559317, int_559318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559319)
        # Adding element type (line 44)
        
        
        # Call to mpf(...): (line 46)
        # Processing the call arguments (line 46)
        int_559322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_559323 = {}
        # Getting the type of 'mp' (line 46)
        mp_559320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 46)
        mpf_559321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), mp_559320, 'mpf')
        # Calling mpf(args, kwargs) (line 46)
        mpf_call_result_559324 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), mpf_559321, *[int_559322], **kwargs_559323)
        
        # Applying the 'usub' unary operator (line 46)
        result___neg___559325 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 17), 'usub', mpf_call_result_559324)
        
        int_559326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 30), 'int')
        # Applying the binary operator 'div' (line 46)
        result_div_559327 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 17), 'div', result___neg___559325, int_559326)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559327)
        # Adding element type (line 44)
        
        # Call to mpf(...): (line 46)
        # Processing the call arguments (line 46)
        int_559330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 46), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_559331 = {}
        # Getting the type of 'mp' (line 46)
        mp_559328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 46)
        mpf_559329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 39), mp_559328, 'mpf')
        # Calling mpf(args, kwargs) (line 46)
        mpf_call_result_559332 = invoke(stypy.reporting.localization.Localization(__file__, 46, 39), mpf_559329, *[int_559330], **kwargs_559331)
        
        int_559333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 49), 'int')
        # Applying the binary operator 'div' (line 46)
        result_div_559334 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 39), 'div', mpf_call_result_559332, int_559333)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 16), list_559273, result_div_559334)
        
        # Assigning a type to the variable 'alpha' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'alpha', list_559273)
        
        # Call to mp_assert_allclose(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to compute_alpha(...): (line 47)
        # Processing the call arguments (line 47)
        int_559337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 41), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_559338 = {}
        # Getting the type of 'compute_alpha' (line 47)
        compute_alpha_559336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'compute_alpha', False)
        # Calling compute_alpha(args, kwargs) (line 47)
        compute_alpha_call_result_559339 = invoke(stypy.reporting.localization.Localization(__file__, 47, 27), compute_alpha_559336, *[int_559337], **kwargs_559338)
        
        # Getting the type of 'alpha' (line 47)
        alpha_559340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 45), 'alpha', False)
        # Processing the call keyword arguments (line 47)
        kwargs_559341 = {}
        # Getting the type of 'mp_assert_allclose' (line 47)
        mp_assert_allclose_559335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'mp_assert_allclose', False)
        # Calling mp_assert_allclose(args, kwargs) (line 47)
        mp_assert_allclose_call_result_559342 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), mp_assert_allclose_559335, *[compute_alpha_call_result_559339, alpha_559340], **kwargs_559341)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 43)
        exit___559343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 9), workdps_call_result_559269, '__exit__')
        with_exit_559344 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), exit___559343, None, None, None)

    
    # ################# End of 'test_alpha(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_alpha' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_559345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_alpha'
    return stypy_return_type_559345

# Assigning a type to the variable 'test_alpha' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'test_alpha', test_alpha)

@norecursion
def test_d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_d'
    module_type_store = module_type_store.open_function_context('test_d', 50, 0, False)
    
    # Passed parameters checking function
    test_d.stypy_localization = localization
    test_d.stypy_type_of_self = None
    test_d.stypy_type_store = module_type_store
    test_d.stypy_function_name = 'test_d'
    test_d.stypy_param_names_list = []
    test_d.stypy_varargs_param_name = None
    test_d.stypy_kwargs_param_name = None
    test_d.stypy_call_defaults = defaults
    test_d.stypy_call_varargs = varargs
    test_d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_d', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_d(...)' code ##################

    
    # Call to workdps(...): (line 62)
    # Processing the call arguments (line 62)
    int_559348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'int')
    # Processing the call keyword arguments (line 62)
    kwargs_559349 = {}
    # Getting the type of 'mp' (line 62)
    mp_559346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 9), 'mp', False)
    # Obtaining the member 'workdps' of a type (line 62)
    workdps_559347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), mp_559346, 'workdps')
    # Calling workdps(args, kwargs) (line 62)
    workdps_call_result_559350 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), workdps_559347, *[int_559348], **kwargs_559349)
    
    with_559351 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 62, 9), workdps_call_result_559350, 'with parameter', '__enter__', '__exit__')

    if with_559351:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 62)
        enter___559352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), workdps_call_result_559350, '__enter__')
        with_enter_559353 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), enter___559352)
        
        # Assigning a List to a Name (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_559354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 63)
        tuple_559355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 63)
        # Adding element type (line 63)
        int_559356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_559355, int_559356)
        # Adding element type (line 63)
        int_559357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_559355, int_559357)
        # Adding element type (line 63)
        
        
        # Call to mpf(...): (line 63)
        # Processing the call arguments (line 63)
        str_559360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'str', '0.333333333333333333333333333333')
        # Processing the call keyword arguments (line 63)
        kwargs_559361 = {}
        # Getting the type of 'mp' (line 63)
        mp_559358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 63)
        mpf_559359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 27), mp_559358, 'mpf')
        # Calling mpf(args, kwargs) (line 63)
        mpf_call_result_559362 = invoke(stypy.reporting.localization.Localization(__file__, 63, 27), mpf_559359, *[str_559360], **kwargs_559361)
        
        # Applying the 'usub' unary operator (line 63)
        result___neg___559363 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 26), 'usub', mpf_call_result_559362)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), tuple_559355, result___neg___559363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559355)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_559364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        int_559365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_559364, int_559365)
        # Adding element type (line 64)
        int_559366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_559364, int_559366)
        # Adding element type (line 64)
        
        # Call to mpf(...): (line 64)
        # Processing the call arguments (line 64)
        str_559369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'str', '0.102618097842403080425739573227e-7')
        # Processing the call keyword arguments (line 64)
        kwargs_559370 = {}
        # Getting the type of 'mp' (line 64)
        mp_559367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 64)
        mpf_559368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 27), mp_559367, 'mpf')
        # Calling mpf(args, kwargs) (line 64)
        mpf_call_result_559371 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), mpf_559368, *[str_559369], **kwargs_559370)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 20), tuple_559364, mpf_call_result_559371)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559364)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_559372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        int_559373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), tuple_559372, int_559373)
        # Adding element type (line 65)
        int_559374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), tuple_559372, int_559374)
        # Adding element type (line 65)
        
        
        # Call to mpf(...): (line 65)
        # Processing the call arguments (line 65)
        str_559377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'str', '0.185185185185185185185185185185e-2')
        # Processing the call keyword arguments (line 65)
        kwargs_559378 = {}
        # Getting the type of 'mp' (line 65)
        mp_559375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 65)
        mpf_559376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 27), mp_559375, 'mpf')
        # Calling mpf(args, kwargs) (line 65)
        mpf_call_result_559379 = invoke(stypy.reporting.localization.Localization(__file__, 65, 27), mpf_559376, *[str_559377], **kwargs_559378)
        
        # Applying the 'usub' unary operator (line 65)
        result___neg___559380 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 26), 'usub', mpf_call_result_559379)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 20), tuple_559372, result___neg___559380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559372)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_559381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        int_559382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_559381, int_559382)
        # Adding element type (line 66)
        int_559383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_559381, int_559383)
        # Adding element type (line 66)
        
        # Call to mpf(...): (line 66)
        # Processing the call arguments (line 66)
        str_559386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'str', '0.119516285997781473243076536700e-7')
        # Processing the call keyword arguments (line 66)
        kwargs_559387 = {}
        # Getting the type of 'mp' (line 66)
        mp_559384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 66)
        mpf_559385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 27), mp_559384, 'mpf')
        # Calling mpf(args, kwargs) (line 66)
        mpf_call_result_559388 = invoke(stypy.reporting.localization.Localization(__file__, 66, 27), mpf_559385, *[str_559386], **kwargs_559387)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 20), tuple_559381, mpf_call_result_559388)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559381)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 67)
        tuple_559389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 67)
        # Adding element type (line 67)
        int_559390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), tuple_559389, int_559390)
        # Adding element type (line 67)
        int_559391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), tuple_559389, int_559391)
        # Adding element type (line 67)
        
        # Call to mpf(...): (line 67)
        # Processing the call arguments (line 67)
        str_559394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'str', '0.413359788359788359788359788360e-2')
        # Processing the call keyword arguments (line 67)
        kwargs_559395 = {}
        # Getting the type of 'mp' (line 67)
        mp_559392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 67)
        mpf_559393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 26), mp_559392, 'mpf')
        # Calling mpf(args, kwargs) (line 67)
        mpf_call_result_559396 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), mpf_559393, *[str_559394], **kwargs_559395)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 20), tuple_559389, mpf_call_result_559396)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559389)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 68)
        tuple_559397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 68)
        # Adding element type (line 68)
        int_559398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), tuple_559397, int_559398)
        # Adding element type (line 68)
        int_559399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), tuple_559397, int_559399)
        # Adding element type (line 68)
        
        
        # Call to mpf(...): (line 68)
        # Processing the call arguments (line 68)
        str_559402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'str', '0.140925299108675210532930244154e-7')
        # Processing the call keyword arguments (line 68)
        kwargs_559403 = {}
        # Getting the type of 'mp' (line 68)
        mp_559400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 68)
        mpf_559401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 28), mp_559400, 'mpf')
        # Calling mpf(args, kwargs) (line 68)
        mpf_call_result_559404 = invoke(stypy.reporting.localization.Localization(__file__, 68, 28), mpf_559401, *[str_559402], **kwargs_559403)
        
        # Applying the 'usub' unary operator (line 68)
        result___neg___559405 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 27), 'usub', mpf_call_result_559404)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), tuple_559397, result___neg___559405)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559397)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_559406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        int_559407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), tuple_559406, int_559407)
        # Adding element type (line 69)
        int_559408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), tuple_559406, int_559408)
        # Adding element type (line 69)
        
        # Call to mpf(...): (line 69)
        # Processing the call arguments (line 69)
        str_559411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'str', '0.649434156378600823045267489712e-3')
        # Processing the call keyword arguments (line 69)
        kwargs_559412 = {}
        # Getting the type of 'mp' (line 69)
        mp_559409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 69)
        mpf_559410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 26), mp_559409, 'mpf')
        # Calling mpf(args, kwargs) (line 69)
        mpf_call_result_559413 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), mpf_559410, *[str_559411], **kwargs_559412)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 20), tuple_559406, mpf_call_result_559413)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559406)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_559414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        int_559415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_559414, int_559415)
        # Adding element type (line 70)
        int_559416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_559414, int_559416)
        # Adding element type (line 70)
        
        
        # Call to mpf(...): (line 70)
        # Processing the call arguments (line 70)
        str_559419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 35), 'str', '0.191111684859736540606728140873e-7')
        # Processing the call keyword arguments (line 70)
        kwargs_559420 = {}
        # Getting the type of 'mp' (line 70)
        mp_559417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 70)
        mpf_559418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 28), mp_559417, 'mpf')
        # Calling mpf(args, kwargs) (line 70)
        mpf_call_result_559421 = invoke(stypy.reporting.localization.Localization(__file__, 70, 28), mpf_559418, *[str_559419], **kwargs_559420)
        
        # Applying the 'usub' unary operator (line 70)
        result___neg___559422 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 27), 'usub', mpf_call_result_559421)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 20), tuple_559414, result___neg___559422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559414)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_559423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        int_559424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 20), tuple_559423, int_559424)
        # Adding element type (line 71)
        int_559425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 20), tuple_559423, int_559425)
        # Adding element type (line 71)
        
        
        # Call to mpf(...): (line 71)
        # Processing the call arguments (line 71)
        str_559428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'str', '0.861888290916711698604702719929e-3')
        # Processing the call keyword arguments (line 71)
        kwargs_559429 = {}
        # Getting the type of 'mp' (line 71)
        mp_559426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 71)
        mpf_559427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 27), mp_559426, 'mpf')
        # Calling mpf(args, kwargs) (line 71)
        mpf_call_result_559430 = invoke(stypy.reporting.localization.Localization(__file__, 71, 27), mpf_559427, *[str_559428], **kwargs_559429)
        
        # Applying the 'usub' unary operator (line 71)
        result___neg___559431 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 26), 'usub', mpf_call_result_559430)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 20), tuple_559423, result___neg___559431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559423)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_559432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        int_559433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), tuple_559432, int_559433)
        # Adding element type (line 72)
        int_559434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), tuple_559432, int_559434)
        # Adding element type (line 72)
        
        # Call to mpf(...): (line 72)
        # Processing the call arguments (line 72)
        str_559437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'str', '0.288658297427087836297341274604e-7')
        # Processing the call keyword arguments (line 72)
        kwargs_559438 = {}
        # Getting the type of 'mp' (line 72)
        mp_559435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 72)
        mpf_559436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 27), mp_559435, 'mpf')
        # Calling mpf(args, kwargs) (line 72)
        mpf_call_result_559439 = invoke(stypy.reporting.localization.Localization(__file__, 72, 27), mpf_559436, *[str_559437], **kwargs_559438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), tuple_559432, mpf_call_result_559439)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559432)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_559440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        int_559441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), tuple_559440, int_559441)
        # Adding element type (line 73)
        int_559442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), tuple_559440, int_559442)
        # Adding element type (line 73)
        
        
        # Call to mpf(...): (line 73)
        # Processing the call arguments (line 73)
        str_559445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'str', '0.336798553366358150308767592718e-3')
        # Processing the call keyword arguments (line 73)
        kwargs_559446 = {}
        # Getting the type of 'mp' (line 73)
        mp_559443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 73)
        mpf_559444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), mp_559443, 'mpf')
        # Calling mpf(args, kwargs) (line 73)
        mpf_call_result_559447 = invoke(stypy.reporting.localization.Localization(__file__, 73, 27), mpf_559444, *[str_559445], **kwargs_559446)
        
        # Applying the 'usub' unary operator (line 73)
        result___neg___559448 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 26), 'usub', mpf_call_result_559447)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), tuple_559440, result___neg___559448)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559440)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_559449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        int_559450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_559449, int_559450)
        # Adding element type (line 74)
        int_559451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_559449, int_559451)
        # Adding element type (line 74)
        
        # Call to mpf(...): (line 74)
        # Processing the call arguments (line 74)
        str_559454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 34), 'str', '0.482409670378941807563762631739e-7')
        # Processing the call keyword arguments (line 74)
        kwargs_559455 = {}
        # Getting the type of 'mp' (line 74)
        mp_559452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 74)
        mpf_559453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 27), mp_559452, 'mpf')
        # Calling mpf(args, kwargs) (line 74)
        mpf_call_result_559456 = invoke(stypy.reporting.localization.Localization(__file__, 74, 27), mpf_559453, *[str_559454], **kwargs_559455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 20), tuple_559449, mpf_call_result_559456)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559449)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_559457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        int_559458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), tuple_559457, int_559458)
        # Adding element type (line 75)
        int_559459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), tuple_559457, int_559459)
        # Adding element type (line 75)
        
        # Call to mpf(...): (line 75)
        # Processing the call arguments (line 75)
        str_559462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'str', '0.531307936463992223165748542978e-3')
        # Processing the call keyword arguments (line 75)
        kwargs_559463 = {}
        # Getting the type of 'mp' (line 75)
        mp_559460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 75)
        mpf_559461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), mp_559460, 'mpf')
        # Calling mpf(args, kwargs) (line 75)
        mpf_call_result_559464 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), mpf_559461, *[str_559462], **kwargs_559463)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 20), tuple_559457, mpf_call_result_559464)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559457)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_559465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        int_559466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 20), tuple_559465, int_559466)
        # Adding element type (line 76)
        int_559467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 20), tuple_559465, int_559467)
        # Adding element type (line 76)
        
        
        # Call to mpf(...): (line 76)
        # Processing the call arguments (line 76)
        str_559470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 35), 'str', '0.882860074633048352505085243179e-7')
        # Processing the call keyword arguments (line 76)
        kwargs_559471 = {}
        # Getting the type of 'mp' (line 76)
        mp_559468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 76)
        mpf_559469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 28), mp_559468, 'mpf')
        # Calling mpf(args, kwargs) (line 76)
        mpf_call_result_559472 = invoke(stypy.reporting.localization.Localization(__file__, 76, 28), mpf_559469, *[str_559470], **kwargs_559471)
        
        # Applying the 'usub' unary operator (line 76)
        result___neg___559473 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), 'usub', mpf_call_result_559472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 20), tuple_559465, result___neg___559473)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559465)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_559474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        int_559475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_559474, int_559475)
        # Adding element type (line 77)
        int_559476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_559474, int_559476)
        # Adding element type (line 77)
        
        # Call to mpf(...): (line 77)
        # Processing the call arguments (line 77)
        str_559479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 33), 'str', '0.344367606892377671254279625109e-3')
        # Processing the call keyword arguments (line 77)
        kwargs_559480 = {}
        # Getting the type of 'mp' (line 77)
        mp_559477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 77)
        mpf_559478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 26), mp_559477, 'mpf')
        # Calling mpf(args, kwargs) (line 77)
        mpf_call_result_559481 = invoke(stypy.reporting.localization.Localization(__file__, 77, 26), mpf_559478, *[str_559479], **kwargs_559480)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_559474, mpf_call_result_559481)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559474)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_559482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        int_559483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 20), tuple_559482, int_559483)
        # Adding element type (line 78)
        int_559484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 20), tuple_559482, int_559484)
        # Adding element type (line 78)
        
        
        # Call to mpf(...): (line 78)
        # Processing the call arguments (line 78)
        str_559487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 35), 'str', '0.175629733590604619378669693914e-6')
        # Processing the call keyword arguments (line 78)
        kwargs_559488 = {}
        # Getting the type of 'mp' (line 78)
        mp_559485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 78)
        mpf_559486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), mp_559485, 'mpf')
        # Calling mpf(args, kwargs) (line 78)
        mpf_call_result_559489 = invoke(stypy.reporting.localization.Localization(__file__, 78, 28), mpf_559486, *[str_559487], **kwargs_559488)
        
        # Applying the 'usub' unary operator (line 78)
        result___neg___559490 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 27), 'usub', mpf_call_result_559489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 20), tuple_559482, result___neg___559490)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559482)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_559491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        int_559492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_559491, int_559492)
        # Adding element type (line 79)
        int_559493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_559491, int_559493)
        # Adding element type (line 79)
        
        
        # Call to mpf(...): (line 79)
        # Processing the call arguments (line 79)
        str_559496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'str', '0.652623918595309418922034919727e-3')
        # Processing the call keyword arguments (line 79)
        kwargs_559497 = {}
        # Getting the type of 'mp' (line 79)
        mp_559494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 79)
        mpf_559495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 27), mp_559494, 'mpf')
        # Calling mpf(args, kwargs) (line 79)
        mpf_call_result_559498 = invoke(stypy.reporting.localization.Localization(__file__, 79, 27), mpf_559495, *[str_559496], **kwargs_559497)
        
        # Applying the 'usub' unary operator (line 79)
        result___neg___559499 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 26), 'usub', mpf_call_result_559498)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_559491, result___neg___559499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559491)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_559500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        int_559501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), tuple_559500, int_559501)
        # Adding element type (line 80)
        int_559502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), tuple_559500, int_559502)
        # Adding element type (line 80)
        
        # Call to mpf(...): (line 80)
        # Processing the call arguments (line 80)
        str_559505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 34), 'str', '0.377358774161109793380344937299e-6')
        # Processing the call keyword arguments (line 80)
        kwargs_559506 = {}
        # Getting the type of 'mp' (line 80)
        mp_559503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 80)
        mpf_559504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 27), mp_559503, 'mpf')
        # Calling mpf(args, kwargs) (line 80)
        mpf_call_result_559507 = invoke(stypy.reporting.localization.Localization(__file__, 80, 27), mpf_559504, *[str_559505], **kwargs_559506)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 20), tuple_559500, mpf_call_result_559507)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559500)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_559508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        int_559509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_559508, int_559509)
        # Adding element type (line 81)
        int_559510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_559508, int_559510)
        # Adding element type (line 81)
        
        
        # Call to mpf(...): (line 81)
        # Processing the call arguments (line 81)
        str_559513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'str', '0.596761290192746250124390067179e-3')
        # Processing the call keyword arguments (line 81)
        kwargs_559514 = {}
        # Getting the type of 'mp' (line 81)
        mp_559511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 81)
        mpf_559512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 27), mp_559511, 'mpf')
        # Calling mpf(args, kwargs) (line 81)
        mpf_call_result_559515 = invoke(stypy.reporting.localization.Localization(__file__, 81, 27), mpf_559512, *[str_559513], **kwargs_559514)
        
        # Applying the 'usub' unary operator (line 81)
        result___neg___559516 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 26), 'usub', mpf_call_result_559515)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_559508, result___neg___559516)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559508)
        # Adding element type (line 63)
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_559517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        int_559518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_559517, int_559518)
        # Adding element type (line 82)
        int_559519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_559517, int_559519)
        # Adding element type (line 82)
        
        # Call to mpf(...): (line 82)
        # Processing the call arguments (line 82)
        str_559522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 34), 'str', '0.870823417786464116761231237189e-6')
        # Processing the call keyword arguments (line 82)
        kwargs_559523 = {}
        # Getting the type of 'mp' (line 82)
        mp_559520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'mp', False)
        # Obtaining the member 'mpf' of a type (line 82)
        mpf_559521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), mp_559520, 'mpf')
        # Calling mpf(args, kwargs) (line 82)
        mpf_call_result_559524 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), mpf_559521, *[str_559522], **kwargs_559523)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 20), tuple_559517, mpf_call_result_559524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 18), list_559354, tuple_559517)
        
        # Assigning a type to the variable 'dataset' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dataset', list_559354)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to compute_d(...): (line 83)
        # Processing the call arguments (line 83)
        int_559526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'int')
        int_559527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'int')
        # Processing the call keyword arguments (line 83)
        kwargs_559528 = {}
        # Getting the type of 'compute_d' (line 83)
        compute_d_559525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'compute_d', False)
        # Calling compute_d(args, kwargs) (line 83)
        compute_d_call_result_559529 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), compute_d_559525, *[int_559526, int_559527], **kwargs_559528)
        
        # Assigning a type to the variable 'd' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'd', compute_d_call_result_559529)
        
        # Assigning a List to a Name (line 84):
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_559530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        
        # Assigning a type to the variable 'res' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'res', list_559530)
        
        # Getting the type of 'dataset' (line 85)
        dataset_559531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'dataset')
        # Testing the type of a for loop iterable (line 85)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 8), dataset_559531)
        # Getting the type of the for loop variable (line 85)
        for_loop_var_559532 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 8), dataset_559531)
        # Assigning a type to the variable 'k' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), for_loop_var_559532))
        # Assigning a type to the variable 'n' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), for_loop_var_559532))
        # Assigning a type to the variable 'std' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'std', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), for_loop_var_559532))
        # SSA begins for a for statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 86)
        n_559535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'n', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 86)
        k_559536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'k', False)
        # Getting the type of 'd' (line 86)
        d_559537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___559538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), d_559537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_559539 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), getitem___559538, k_559536)
        
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___559540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 23), subscript_call_result_559539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_559541 = invoke(stypy.reporting.localization.Localization(__file__, 86, 23), getitem___559540, n_559535)
        
        # Processing the call keyword arguments (line 86)
        kwargs_559542 = {}
        # Getting the type of 'res' (line 86)
        res_559533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'res', False)
        # Obtaining the member 'append' of a type (line 86)
        append_559534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), res_559533, 'append')
        # Calling append(args, kwargs) (line 86)
        append_call_result_559543 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), append_559534, *[subscript_call_result_559541], **kwargs_559542)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 87):
        
        # Call to map(...): (line 87)
        # Processing the call arguments (line 87)

        @norecursion
        def _stypy_temp_lambda_476(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_476'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_476', 87, 18, True)
            # Passed parameters checking function
            _stypy_temp_lambda_476.stypy_localization = localization
            _stypy_temp_lambda_476.stypy_type_of_self = None
            _stypy_temp_lambda_476.stypy_type_store = module_type_store
            _stypy_temp_lambda_476.stypy_function_name = '_stypy_temp_lambda_476'
            _stypy_temp_lambda_476.stypy_param_names_list = ['x']
            _stypy_temp_lambda_476.stypy_varargs_param_name = None
            _stypy_temp_lambda_476.stypy_kwargs_param_name = None
            _stypy_temp_lambda_476.stypy_call_defaults = defaults
            _stypy_temp_lambda_476.stypy_call_varargs = varargs
            _stypy_temp_lambda_476.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_476', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_476', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_559545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
            # Getting the type of 'x' (line 87)
            x_559546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 87)
            getitem___559547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 28), x_559546, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 87)
            subscript_call_result_559548 = invoke(stypy.reporting.localization.Localization(__file__, 87, 28), getitem___559547, int_559545)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'stypy_return_type', subscript_call_result_559548)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_476' in the type store
            # Getting the type of 'stypy_return_type' (line 87)
            stypy_return_type_559549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_559549)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_476'
            return stypy_return_type_559549

        # Assigning a type to the variable '_stypy_temp_lambda_476' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), '_stypy_temp_lambda_476', _stypy_temp_lambda_476)
        # Getting the type of '_stypy_temp_lambda_476' (line 87)
        _stypy_temp_lambda_476_559550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), '_stypy_temp_lambda_476')
        # Getting the type of 'dataset' (line 87)
        dataset_559551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'dataset', False)
        # Processing the call keyword arguments (line 87)
        kwargs_559552 = {}
        # Getting the type of 'map' (line 87)
        map_559544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'map', False)
        # Calling map(args, kwargs) (line 87)
        map_call_result_559553 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), map_559544, *[_stypy_temp_lambda_476_559550, dataset_559551], **kwargs_559552)
        
        # Assigning a type to the variable 'std' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'std', map_call_result_559553)
        
        # Call to mp_assert_allclose(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'res' (line 88)
        res_559555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'res', False)
        # Getting the type of 'std' (line 88)
        std_559556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'std', False)
        # Processing the call keyword arguments (line 88)
        kwargs_559557 = {}
        # Getting the type of 'mp_assert_allclose' (line 88)
        mp_assert_allclose_559554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'mp_assert_allclose', False)
        # Calling mp_assert_allclose(args, kwargs) (line 88)
        mp_assert_allclose_call_result_559558 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), mp_assert_allclose_559554, *[res_559555, std_559556], **kwargs_559557)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 62)
        exit___559559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 9), workdps_call_result_559350, '__exit__')
        with_exit_559560 = invoke(stypy.reporting.localization.Localization(__file__, 62, 9), exit___559559, None, None, None)

    
    # ################# End of 'test_d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_d' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_559561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559561)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_d'
    return stypy_return_type_559561

# Assigning a type to the variable 'test_d' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_d', test_d)

@norecursion
def test_gammainc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gammainc'
    module_type_store = module_type_store.open_function_context('test_gammainc', 91, 0, False)
    
    # Passed parameters checking function
    test_gammainc.stypy_localization = localization
    test_gammainc.stypy_type_of_self = None
    test_gammainc.stypy_type_store = module_type_store
    test_gammainc.stypy_function_name = 'test_gammainc'
    test_gammainc.stypy_param_names_list = []
    test_gammainc.stypy_varargs_param_name = None
    test_gammainc.stypy_kwargs_param_name = None
    test_gammainc.stypy_call_defaults = defaults
    test_gammainc.stypy_call_varargs = varargs
    test_gammainc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gammainc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gammainc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gammainc(...)' code ##################

    
    # Call to assert_mpmath_equal(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'gammainc' (line 96)
    gammainc_559563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'gammainc', False)

    @norecursion
    def _stypy_temp_lambda_477(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_477'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_477', 97, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_477.stypy_localization = localization
        _stypy_temp_lambda_477.stypy_type_of_self = None
        _stypy_temp_lambda_477.stypy_type_store = module_type_store
        _stypy_temp_lambda_477.stypy_function_name = '_stypy_temp_lambda_477'
        _stypy_temp_lambda_477.stypy_param_names_list = ['a', 'x']
        _stypy_temp_lambda_477.stypy_varargs_param_name = None
        _stypy_temp_lambda_477.stypy_kwargs_param_name = None
        _stypy_temp_lambda_477.stypy_call_defaults = defaults
        _stypy_temp_lambda_477.stypy_call_varargs = varargs
        _stypy_temp_lambda_477.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_477', ['a', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_477', ['a', 'x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to gammainc(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'a' (line 97)
        a_559566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 49), 'a', False)
        # Processing the call keyword arguments (line 97)
        # Getting the type of 'x' (line 97)
        x_559567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'x', False)
        keyword_559568 = x_559567
        # Getting the type of 'True' (line 97)
        True_559569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 69), 'True', False)
        keyword_559570 = True_559569
        kwargs_559571 = {'regularized': keyword_559570, 'b': keyword_559568}
        # Getting the type of 'mp' (line 97)
        mp_559564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'mp', False)
        # Obtaining the member 'gammainc' of a type (line 97)
        gammainc_559565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 37), mp_559564, 'gammainc')
        # Calling gammainc(args, kwargs) (line 97)
        gammainc_call_result_559572 = invoke(stypy.reporting.localization.Localization(__file__, 97, 37), gammainc_559565, *[a_559566], **kwargs_559571)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'stypy_return_type', gammainc_call_result_559572)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_477' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_559573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_477'
        return stypy_return_type_559573

    # Assigning a type to the variable '_stypy_temp_lambda_477' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), '_stypy_temp_lambda_477', _stypy_temp_lambda_477)
    # Getting the type of '_stypy_temp_lambda_477' (line 97)
    _stypy_temp_lambda_477_559574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), '_stypy_temp_lambda_477')
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_559575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    
    # Call to Arg(...): (line 98)
    # Processing the call arguments (line 98)
    int_559577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'int')
    int_559578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 32), 'int')
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'False' (line 98)
    False_559579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 49), 'False', False)
    keyword_559580 = False_559579
    kwargs_559581 = {'inclusive_a': keyword_559580}
    # Getting the type of 'Arg' (line 98)
    Arg_559576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'Arg', False)
    # Calling Arg(args, kwargs) (line 98)
    Arg_call_result_559582 = invoke(stypy.reporting.localization.Localization(__file__, 98, 25), Arg_559576, *[int_559577, int_559578], **kwargs_559581)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_559575, Arg_call_result_559582)
    # Adding element type (line 98)
    
    # Call to Arg(...): (line 98)
    # Processing the call arguments (line 98)
    int_559584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 61), 'int')
    int_559585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 64), 'int')
    # Processing the call keyword arguments (line 98)
    kwargs_559586 = {}
    # Getting the type of 'Arg' (line 98)
    Arg_559583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 57), 'Arg', False)
    # Calling Arg(args, kwargs) (line 98)
    Arg_call_result_559587 = invoke(stypy.reporting.localization.Localization(__file__, 98, 57), Arg_559583, *[int_559584, int_559585], **kwargs_559586)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_559575, Arg_call_result_559587)
    
    # Processing the call keyword arguments (line 96)
    # Getting the type of 'False' (line 99)
    False_559588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'False', False)
    keyword_559589 = False_559588
    float_559590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 43), 'float')
    keyword_559591 = float_559590
    int_559592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 52), 'int')
    keyword_559593 = int_559592
    int_559594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 60), 'int')
    keyword_559595 = int_559594
    kwargs_559596 = {'rtol': keyword_559591, 'dps': keyword_559595, 'nan_ok': keyword_559589, 'n': keyword_559593}
    # Getting the type of 'assert_mpmath_equal' (line 96)
    assert_mpmath_equal_559562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'assert_mpmath_equal', False)
    # Calling assert_mpmath_equal(args, kwargs) (line 96)
    assert_mpmath_equal_call_result_559597 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), assert_mpmath_equal_559562, *[gammainc_559563, _stypy_temp_lambda_477_559574, list_559575], **kwargs_559596)
    
    
    # ################# End of 'test_gammainc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gammainc' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_559598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559598)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gammainc'
    return stypy_return_type_559598

# Assigning a type to the variable 'test_gammainc' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'test_gammainc', test_gammainc)

@norecursion
def test_gammaincc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gammaincc'
    module_type_store = module_type_store.open_function_context('test_gammaincc', 102, 0, False)
    
    # Passed parameters checking function
    test_gammaincc.stypy_localization = localization
    test_gammaincc.stypy_type_of_self = None
    test_gammaincc.stypy_type_store = module_type_store
    test_gammaincc.stypy_function_name = 'test_gammaincc'
    test_gammaincc.stypy_param_names_list = []
    test_gammaincc.stypy_varargs_param_name = None
    test_gammaincc.stypy_kwargs_param_name = None
    test_gammaincc.stypy_call_defaults = defaults
    test_gammaincc.stypy_call_varargs = varargs
    test_gammaincc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gammaincc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gammaincc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gammaincc(...)' code ##################

    
    # Call to assert_mpmath_equal(...): (line 107)
    # Processing the call arguments (line 107)

    @norecursion
    def _stypy_temp_lambda_478(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_478'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_478', 107, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_478.stypy_localization = localization
        _stypy_temp_lambda_478.stypy_type_of_self = None
        _stypy_temp_lambda_478.stypy_type_store = module_type_store
        _stypy_temp_lambda_478.stypy_function_name = '_stypy_temp_lambda_478'
        _stypy_temp_lambda_478.stypy_param_names_list = ['a', 'x']
        _stypy_temp_lambda_478.stypy_varargs_param_name = None
        _stypy_temp_lambda_478.stypy_kwargs_param_name = None
        _stypy_temp_lambda_478.stypy_call_defaults = defaults
        _stypy_temp_lambda_478.stypy_call_varargs = varargs
        _stypy_temp_lambda_478.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_478', ['a', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_478', ['a', 'x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to gammaincc(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'a' (line 107)
        a_559601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'a', False)
        # Getting the type of 'x' (line 107)
        x_559602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 50), 'x', False)
        # Processing the call keyword arguments (line 107)
        int_559603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 57), 'int')
        keyword_559604 = int_559603
        kwargs_559605 = {'dps': keyword_559604}
        # Getting the type of 'gammaincc' (line 107)
        gammaincc_559600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 37), 'gammaincc', False)
        # Calling gammaincc(args, kwargs) (line 107)
        gammaincc_call_result_559606 = invoke(stypy.reporting.localization.Localization(__file__, 107, 37), gammaincc_559600, *[a_559601, x_559602], **kwargs_559605)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'stypy_return_type', gammaincc_call_result_559606)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_478' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_559607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_478'
        return stypy_return_type_559607

    # Assigning a type to the variable '_stypy_temp_lambda_478' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), '_stypy_temp_lambda_478', _stypy_temp_lambda_478)
    # Getting the type of '_stypy_temp_lambda_478' (line 107)
    _stypy_temp_lambda_478_559608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), '_stypy_temp_lambda_478')

    @norecursion
    def _stypy_temp_lambda_479(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_479'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_479', 108, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_479.stypy_localization = localization
        _stypy_temp_lambda_479.stypy_type_of_self = None
        _stypy_temp_lambda_479.stypy_type_store = module_type_store
        _stypy_temp_lambda_479.stypy_function_name = '_stypy_temp_lambda_479'
        _stypy_temp_lambda_479.stypy_param_names_list = ['a', 'x']
        _stypy_temp_lambda_479.stypy_varargs_param_name = None
        _stypy_temp_lambda_479.stypy_kwargs_param_name = None
        _stypy_temp_lambda_479.stypy_call_defaults = defaults
        _stypy_temp_lambda_479.stypy_call_varargs = varargs
        _stypy_temp_lambda_479.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_479', ['a', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_479', ['a', 'x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to gammainc(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'a' (line 108)
        a_559611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 49), 'a', False)
        # Processing the call keyword arguments (line 108)
        # Getting the type of 'x' (line 108)
        x_559612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 54), 'x', False)
        keyword_559613 = x_559612
        # Getting the type of 'True' (line 108)
        True_559614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 69), 'True', False)
        keyword_559615 = True_559614
        kwargs_559616 = {'a': keyword_559613, 'regularized': keyword_559615}
        # Getting the type of 'mp' (line 108)
        mp_559609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'mp', False)
        # Obtaining the member 'gammainc' of a type (line 108)
        gammainc_559610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 37), mp_559609, 'gammainc')
        # Calling gammainc(args, kwargs) (line 108)
        gammainc_call_result_559617 = invoke(stypy.reporting.localization.Localization(__file__, 108, 37), gammainc_559610, *[a_559611], **kwargs_559616)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'stypy_return_type', gammainc_call_result_559617)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_479' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_559618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559618)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_479'
        return stypy_return_type_559618

    # Assigning a type to the variable '_stypy_temp_lambda_479' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), '_stypy_temp_lambda_479', _stypy_temp_lambda_479)
    # Getting the type of '_stypy_temp_lambda_479' (line 108)
    _stypy_temp_lambda_479_559619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), '_stypy_temp_lambda_479')
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_559620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    
    # Call to Arg(...): (line 109)
    # Processing the call arguments (line 109)
    int_559622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 29), 'int')
    int_559623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'int')
    # Processing the call keyword arguments (line 109)
    kwargs_559624 = {}
    # Getting the type of 'Arg' (line 109)
    Arg_559621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'Arg', False)
    # Calling Arg(args, kwargs) (line 109)
    Arg_call_result_559625 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), Arg_559621, *[int_559622, int_559623], **kwargs_559624)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_559620, Arg_call_result_559625)
    # Adding element type (line 109)
    
    # Call to Arg(...): (line 109)
    # Processing the call arguments (line 109)
    int_559627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 43), 'int')
    int_559628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'int')
    # Processing the call keyword arguments (line 109)
    kwargs_559629 = {}
    # Getting the type of 'Arg' (line 109)
    Arg_559626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'Arg', False)
    # Calling Arg(args, kwargs) (line 109)
    Arg_call_result_559630 = invoke(stypy.reporting.localization.Localization(__file__, 109, 39), Arg_559626, *[int_559627, int_559628], **kwargs_559629)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_559620, Arg_call_result_559630)
    
    # Processing the call keyword arguments (line 107)
    # Getting the type of 'False' (line 110)
    False_559631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'False', False)
    keyword_559632 = False_559631
    float_559633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'float')
    keyword_559634 = float_559633
    int_559635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 52), 'int')
    keyword_559636 = int_559635
    int_559637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 60), 'int')
    keyword_559638 = int_559637
    kwargs_559639 = {'rtol': keyword_559634, 'dps': keyword_559638, 'nan_ok': keyword_559632, 'n': keyword_559636}
    # Getting the type of 'assert_mpmath_equal' (line 107)
    assert_mpmath_equal_559599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'assert_mpmath_equal', False)
    # Calling assert_mpmath_equal(args, kwargs) (line 107)
    assert_mpmath_equal_call_result_559640 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), assert_mpmath_equal_559599, *[_stypy_temp_lambda_478_559608, _stypy_temp_lambda_479_559619, list_559620], **kwargs_559639)
    
    
    # Call to assert_mpmath_equal(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'gammaincc' (line 113)
    gammaincc_559642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'gammaincc', False)

    @norecursion
    def _stypy_temp_lambda_480(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_480'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_480', 114, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_480.stypy_localization = localization
        _stypy_temp_lambda_480.stypy_type_of_self = None
        _stypy_temp_lambda_480.stypy_type_store = module_type_store
        _stypy_temp_lambda_480.stypy_function_name = '_stypy_temp_lambda_480'
        _stypy_temp_lambda_480.stypy_param_names_list = ['a', 'x']
        _stypy_temp_lambda_480.stypy_varargs_param_name = None
        _stypy_temp_lambda_480.stypy_kwargs_param_name = None
        _stypy_temp_lambda_480.stypy_call_defaults = defaults
        _stypy_temp_lambda_480.stypy_call_varargs = varargs
        _stypy_temp_lambda_480.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_480', ['a', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_480', ['a', 'x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to gammainc(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'a' (line 114)
        a_559645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 49), 'a', False)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'x' (line 114)
        x_559646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 54), 'x', False)
        keyword_559647 = x_559646
        # Getting the type of 'True' (line 114)
        True_559648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 69), 'True', False)
        keyword_559649 = True_559648
        kwargs_559650 = {'a': keyword_559647, 'regularized': keyword_559649}
        # Getting the type of 'mp' (line 114)
        mp_559643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'mp', False)
        # Obtaining the member 'gammainc' of a type (line 114)
        gammainc_559644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 37), mp_559643, 'gammainc')
        # Calling gammainc(args, kwargs) (line 114)
        gammainc_call_result_559651 = invoke(stypy.reporting.localization.Localization(__file__, 114, 37), gammainc_559644, *[a_559645], **kwargs_559650)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'stypy_return_type', gammainc_call_result_559651)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_480' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_559652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_480'
        return stypy_return_type_559652

    # Assigning a type to the variable '_stypy_temp_lambda_480' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), '_stypy_temp_lambda_480', _stypy_temp_lambda_480)
    # Getting the type of '_stypy_temp_lambda_480' (line 114)
    _stypy_temp_lambda_480_559653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), '_stypy_temp_lambda_480')
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_559654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    
    # Call to IntArg(...): (line 115)
    # Processing the call arguments (line 115)
    int_559656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
    int_559657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'int')
    # Processing the call keyword arguments (line 115)
    kwargs_559658 = {}
    # Getting the type of 'IntArg' (line 115)
    IntArg_559655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'IntArg', False)
    # Calling IntArg(args, kwargs) (line 115)
    IntArg_call_result_559659 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), IntArg_559655, *[int_559656, int_559657], **kwargs_559658)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 24), list_559654, IntArg_call_result_559659)
    # Adding element type (line 115)
    
    # Call to Arg(...): (line 115)
    # Processing the call arguments (line 115)
    int_559661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'int')
    int_559662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 48), 'int')
    # Processing the call keyword arguments (line 115)
    kwargs_559663 = {}
    # Getting the type of 'Arg' (line 115)
    Arg_559660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'Arg', False)
    # Calling Arg(args, kwargs) (line 115)
    Arg_call_result_559664 = invoke(stypy.reporting.localization.Localization(__file__, 115, 41), Arg_559660, *[int_559661, int_559662], **kwargs_559663)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 24), list_559654, Arg_call_result_559664)
    
    # Processing the call keyword arguments (line 113)
    # Getting the type of 'False' (line 116)
    False_559665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'False', False)
    keyword_559666 = False_559665
    float_559667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 43), 'float')
    keyword_559668 = float_559667
    int_559669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 52), 'int')
    keyword_559670 = int_559669
    int_559671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'int')
    keyword_559672 = int_559671
    kwargs_559673 = {'rtol': keyword_559668, 'dps': keyword_559672, 'nan_ok': keyword_559666, 'n': keyword_559670}
    # Getting the type of 'assert_mpmath_equal' (line 113)
    assert_mpmath_equal_559641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'assert_mpmath_equal', False)
    # Calling assert_mpmath_equal(args, kwargs) (line 113)
    assert_mpmath_equal_call_result_559674 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), assert_mpmath_equal_559641, *[gammaincc_559642, _stypy_temp_lambda_480_559653, list_559654], **kwargs_559673)
    
    
    # ################# End of 'test_gammaincc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gammaincc' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_559675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gammaincc'
    return stypy_return_type_559675

# Assigning a type to the variable 'test_gammaincc' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'test_gammaincc', test_gammaincc)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
