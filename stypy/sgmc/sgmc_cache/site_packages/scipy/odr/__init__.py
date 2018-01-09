
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =================================================
3: Orthogonal distance regression (:mod:`scipy.odr`)
4: =================================================
5: 
6: .. currentmodule:: scipy.odr
7: 
8: Package Content
9: ===============
10: 
11: .. autosummary::
12:    :toctree: generated/
13: 
14:    Data          -- The data to fit.
15:    RealData      -- Data with weights as actual std. dev.s and/or covariances.
16:    Model         -- Stores information about the function to be fit.
17:    ODR           -- Gathers all info & manages the main fitting routine.
18:    Output        -- Result from the fit.
19:    odr           -- Low-level function for ODR.
20: 
21:    OdrWarning    -- Warning about potential problems when running ODR
22:    OdrError      -- Error exception.
23:    OdrStop       -- Stop exception.
24: 
25:    odr_error     -- Same as OdrError (for backwards compatibility)
26:    odr_stop      -- Same as OdrStop (for backwards compatibility)
27: 
28: Prebuilt models:
29: 
30: .. autosummary::
31:    :toctree: generated/
32: 
33:    polynomial
34: 
35: .. data:: exponential
36: 
37: .. data:: multilinear
38: 
39: .. data:: unilinear
40: 
41: .. data:: quadratic
42: 
43: .. data:: polynomial
44: 
45: Usage information
46: =================
47: 
48: Introduction
49: ------------
50: 
51: Why Orthogonal Distance Regression (ODR)?  Sometimes one has
52: measurement errors in the explanatory (a.k.a., "independent")
53: variable(s), not just the response (a.k.a., "dependent") variable(s).
54: Ordinary Least Squares (OLS) fitting procedures treat the data for
55: explanatory variables as fixed, i.e., not subject to error of any kind.
56: Furthermore, OLS procedures require that the response variables be an
57: explicit function of the explanatory variables; sometimes making the
58: equation explicit is impractical and/or introduces errors.  ODR can
59: handle both of these cases with ease, and can even reduce to the OLS
60: case if that is sufficient for the problem.
61: 
62: ODRPACK is a FORTRAN-77 library for performing ODR with possibly
63: non-linear fitting functions.  It uses a modified trust-region
64: Levenberg-Marquardt-type algorithm [1]_ to estimate the function
65: parameters.  The fitting functions are provided by Python functions
66: operating on NumPy arrays.  The required derivatives may be provided
67: by Python functions as well, or may be estimated numerically.  ODRPACK
68: can do explicit or implicit ODR fits, or it can do OLS.  Input and
69: output variables may be multi-dimensional.  Weights can be provided to
70: account for different variances of the observations, and even
71: covariances between dimensions of the variables.
72: 
73: The `scipy.odr` package offers an object-oriented interface to
74: ODRPACK, in addition to the low-level `odr` function.
75: 
76: Additional background information about ODRPACK can be found in the
77: `ODRPACK User's Guide
78: <https://docs.scipy.org/doc/external/odrpack_guide.pdf>`_, reading
79: which is recommended.
80: 
81: Basic usage
82: -----------
83: 
84: 1. Define the function you want to fit against.::
85: 
86:        def f(B, x):
87:            '''Linear function y = m*x + b'''
88:            # B is a vector of the parameters.
89:            # x is an array of the current x values.
90:            # x is in the same format as the x passed to Data or RealData.
91:            #
92:            # Return an array in the same format as y passed to Data or RealData.
93:            return B[0]*x + B[1]
94: 
95: 2. Create a Model.::
96: 
97:        linear = Model(f)
98: 
99: 3. Create a Data or RealData instance.::
100: 
101:        mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))
102: 
103:    or, when the actual covariances are known::
104: 
105:        mydata = RealData(x, y, sx=sx, sy=sy)
106: 
107: 4. Instantiate ODR with your data, model and initial parameter estimate.::
108: 
109:        myodr = ODR(mydata, linear, beta0=[1., 2.])
110: 
111: 5. Run the fit.::
112: 
113:        myoutput = myodr.run()
114: 
115: 6. Examine output.::
116: 
117:        myoutput.pprint()
118: 
119: 
120: References
121: ----------
122: .. [1] P. T. Boggs and J. E. Rogers, "Orthogonal Distance Regression,"
123:    in "Statistical analysis of measurement error models and
124:    applications: proceedings of the AMS-IMS-SIAM joint summer research
125:    conference held June 10-16, 1989," Contemporary Mathematics,
126:    vol. 112, pg. 186, 1990.
127: 
128: '''
129: # version: 0.7
130: # author: Robert Kern <robert.kern@gmail.com>
131: # date: 2006-09-21
132: 
133: from __future__ import division, print_function, absolute_import
134: 
135: from .odrpack import *
136: from .models import *
137: from . import add_newdocs
138: 
139: __all__ = [s for s in dir() if not s.startswith('_')]
140: 
141: from scipy._lib._testutils import PytestTester
142: test = PytestTester(__name__)
143: del PytestTester
144: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_165752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n=================================================\nOrthogonal distance regression (:mod:`scipy.odr`)\n=================================================\n\n.. currentmodule:: scipy.odr\n\nPackage Content\n===============\n\n.. autosummary::\n   :toctree: generated/\n\n   Data          -- The data to fit.\n   RealData      -- Data with weights as actual std. dev.s and/or covariances.\n   Model         -- Stores information about the function to be fit.\n   ODR           -- Gathers all info & manages the main fitting routine.\n   Output        -- Result from the fit.\n   odr           -- Low-level function for ODR.\n\n   OdrWarning    -- Warning about potential problems when running ODR\n   OdrError      -- Error exception.\n   OdrStop       -- Stop exception.\n\n   odr_error     -- Same as OdrError (for backwards compatibility)\n   odr_stop      -- Same as OdrStop (for backwards compatibility)\n\nPrebuilt models:\n\n.. autosummary::\n   :toctree: generated/\n\n   polynomial\n\n.. data:: exponential\n\n.. data:: multilinear\n\n.. data:: unilinear\n\n.. data:: quadratic\n\n.. data:: polynomial\n\nUsage information\n=================\n\nIntroduction\n------------\n\nWhy Orthogonal Distance Regression (ODR)?  Sometimes one has\nmeasurement errors in the explanatory (a.k.a., "independent")\nvariable(s), not just the response (a.k.a., "dependent") variable(s).\nOrdinary Least Squares (OLS) fitting procedures treat the data for\nexplanatory variables as fixed, i.e., not subject to error of any kind.\nFurthermore, OLS procedures require that the response variables be an\nexplicit function of the explanatory variables; sometimes making the\nequation explicit is impractical and/or introduces errors.  ODR can\nhandle both of these cases with ease, and can even reduce to the OLS\ncase if that is sufficient for the problem.\n\nODRPACK is a FORTRAN-77 library for performing ODR with possibly\nnon-linear fitting functions.  It uses a modified trust-region\nLevenberg-Marquardt-type algorithm [1]_ to estimate the function\nparameters.  The fitting functions are provided by Python functions\noperating on NumPy arrays.  The required derivatives may be provided\nby Python functions as well, or may be estimated numerically.  ODRPACK\ncan do explicit or implicit ODR fits, or it can do OLS.  Input and\noutput variables may be multi-dimensional.  Weights can be provided to\naccount for different variances of the observations, and even\ncovariances between dimensions of the variables.\n\nThe `scipy.odr` package offers an object-oriented interface to\nODRPACK, in addition to the low-level `odr` function.\n\nAdditional background information about ODRPACK can be found in the\n`ODRPACK User\'s Guide\n<https://docs.scipy.org/doc/external/odrpack_guide.pdf>`_, reading\nwhich is recommended.\n\nBasic usage\n-----------\n\n1. Define the function you want to fit against.::\n\n       def f(B, x):\n           \'\'\'Linear function y = m*x + b\'\'\'\n           # B is a vector of the parameters.\n           # x is an array of the current x values.\n           # x is in the same format as the x passed to Data or RealData.\n           #\n           # Return an array in the same format as y passed to Data or RealData.\n           return B[0]*x + B[1]\n\n2. Create a Model.::\n\n       linear = Model(f)\n\n3. Create a Data or RealData instance.::\n\n       mydata = Data(x, y, wd=1./power(sx,2), we=1./power(sy,2))\n\n   or, when the actual covariances are known::\n\n       mydata = RealData(x, y, sx=sx, sy=sy)\n\n4. Instantiate ODR with your data, model and initial parameter estimate.::\n\n       myodr = ODR(mydata, linear, beta0=[1., 2.])\n\n5. Run the fit.::\n\n       myoutput = myodr.run()\n\n6. Examine output.::\n\n       myoutput.pprint()\n\n\nReferences\n----------\n.. [1] P. T. Boggs and J. E. Rogers, "Orthogonal Distance Regression,"\n   in "Statistical analysis of measurement error models and\n   applications: proceedings of the AMS-IMS-SIAM joint summer research\n   conference held June 10-16, 1989," Contemporary Mathematics,\n   vol. 112, pg. 186, 1990.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 135, 0))

# 'from scipy.odr.odrpack import ' statement (line 135)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_165753 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 135, 0), 'scipy.odr.odrpack')

if (type(import_165753) is not StypyTypeError):

    if (import_165753 != 'pyd_module'):
        __import__(import_165753)
        sys_modules_165754 = sys.modules[import_165753]
        import_from_module(stypy.reporting.localization.Localization(__file__, 135, 0), 'scipy.odr.odrpack', sys_modules_165754.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 135, 0), __file__, sys_modules_165754, sys_modules_165754.module_type_store, module_type_store)
    else:
        from scipy.odr.odrpack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 135, 0), 'scipy.odr.odrpack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.odr.odrpack' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'scipy.odr.odrpack', import_165753)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 136, 0))

# 'from scipy.odr.models import ' statement (line 136)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_165755 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'scipy.odr.models')

if (type(import_165755) is not StypyTypeError):

    if (import_165755 != 'pyd_module'):
        __import__(import_165755)
        sys_modules_165756 = sys.modules[import_165755]
        import_from_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'scipy.odr.models', sys_modules_165756.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 136, 0), __file__, sys_modules_165756, sys_modules_165756.module_type_store, module_type_store)
    else:
        from scipy.odr.models import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 136, 0), 'scipy.odr.models', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.odr.models' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'scipy.odr.models', import_165755)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 137, 0))

# 'from scipy.odr import add_newdocs' statement (line 137)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_165757 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'scipy.odr')

if (type(import_165757) is not StypyTypeError):

    if (import_165757 != 'pyd_module'):
        __import__(import_165757)
        sys_modules_165758 = sys.modules[import_165757]
        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'scipy.odr', sys_modules_165758.module_type_store, module_type_store, ['add_newdocs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 137, 0), __file__, sys_modules_165758, sys_modules_165758.module_type_store, module_type_store)
    else:
        from scipy.odr import add_newdocs

        import_from_module(stypy.reporting.localization.Localization(__file__, 137, 0), 'scipy.odr', None, module_type_store, ['add_newdocs'], [add_newdocs])

else:
    # Assigning a type to the variable 'scipy.odr' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'scipy.odr', import_165757)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


# Assigning a ListComp to a Name (line 139):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 139)
# Processing the call keyword arguments (line 139)
kwargs_165767 = {}
# Getting the type of 'dir' (line 139)
dir_165766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'dir', False)
# Calling dir(args, kwargs) (line 139)
dir_call_result_165768 = invoke(stypy.reporting.localization.Localization(__file__, 139, 22), dir_165766, *[], **kwargs_165767)

comprehension_165769 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 11), dir_call_result_165768)
# Assigning a type to the variable 's' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 's', comprehension_165769)


# Call to startswith(...): (line 139)
# Processing the call arguments (line 139)
str_165762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 48), 'str', '_')
# Processing the call keyword arguments (line 139)
kwargs_165763 = {}
# Getting the type of 's' (line 139)
s_165760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 139)
startswith_165761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 35), s_165760, 'startswith')
# Calling startswith(args, kwargs) (line 139)
startswith_call_result_165764 = invoke(stypy.reporting.localization.Localization(__file__, 139, 35), startswith_165761, *[str_165762], **kwargs_165763)

# Applying the 'not' unary operator (line 139)
result_not__165765 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), 'not', startswith_call_result_165764)

# Getting the type of 's' (line 139)
s_165759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 's')
list_165770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 11), list_165770, s_165759)
# Assigning a type to the variable '__all__' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), '__all__', list_165770)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 141, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 141)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_165771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 141, 0), 'scipy._lib._testutils')

if (type(import_165771) is not StypyTypeError):

    if (import_165771 != 'pyd_module'):
        __import__(import_165771)
        sys_modules_165772 = sys.modules[import_165771]
        import_from_module(stypy.reporting.localization.Localization(__file__, 141, 0), 'scipy._lib._testutils', sys_modules_165772.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 141, 0), __file__, sys_modules_165772, sys_modules_165772.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 141, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'scipy._lib._testutils', import_165771)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


# Assigning a Call to a Name (line 142):

# Call to PytestTester(...): (line 142)
# Processing the call arguments (line 142)
# Getting the type of '__name__' (line 142)
name___165774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), '__name__', False)
# Processing the call keyword arguments (line 142)
kwargs_165775 = {}
# Getting the type of 'PytestTester' (line 142)
PytestTester_165773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 142)
PytestTester_call_result_165776 = invoke(stypy.reporting.localization.Localization(__file__, 142, 7), PytestTester_165773, *[name___165774], **kwargs_165775)

# Assigning a type to the variable 'test' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'test', PytestTester_call_result_165776)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 143, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
