
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Linear Solvers
3: ==============
4: 
5: The default solver is SuperLU (included in the scipy distribution),
6: which can solve real or complex linear systems in both single and
7: double precisions.  It is automatically replaced by UMFPACK, if
8: available.  Note that UMFPACK works in double precision only, so
9: switch it off by::
10: 
11:     >>> use_solver(useUmfpack=False)
12: 
13: to solve in the single precision. See also use_solver documentation.
14: 
15: Example session::
16: 
17:     >>> from scipy.sparse import csc_matrix, spdiags
18:     >>> from numpy import array
19:     >>> from scipy.sparse.linalg import spsolve, use_solver
20:     >>>
21:     >>> print("Inverting a sparse linear system:")
22:     >>> print("The sparse matrix (constructed from diagonals):")
23:     >>> a = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5)
24:     >>> b = array([1, 2, 3, 4, 5])
25:     >>> print("Solve: single precision complex:")
26:     >>> use_solver( useUmfpack = False )
27:     >>> a = a.astype('F')
28:     >>> x = spsolve(a, b)
29:     >>> print(x)
30:     >>> print("Error: ", a*x-b)
31:     >>>
32:     >>> print("Solve: double precision complex:")
33:     >>> use_solver( useUmfpack = True )
34:     >>> a = a.astype('D')
35:     >>> x = spsolve(a, b)
36:     >>> print(x)
37:     >>> print("Error: ", a*x-b)
38:     >>>
39:     >>> print("Solve: double precision:")
40:     >>> a = a.astype('d')
41:     >>> x = spsolve(a, b)
42:     >>> print(x)
43:     >>> print("Error: ", a*x-b)
44:     >>>
45:     >>> print("Solve: single precision:")
46:     >>> use_solver( useUmfpack = False )
47:     >>> a = a.astype('f')
48:     >>> x = spsolve(a, b.astype('f'))
49:     >>> print(x)
50:     >>> print("Error: ", a*x-b)
51: 
52: '''
53: 
54: from __future__ import division, print_function, absolute_import
55: 
56: #import umfpack
57: #__doc__ = '\n\n'.join( (__doc__,  umfpack.__doc__) )
58: #del umfpack
59: 
60: from .linsolve import *
61: from ._superlu import SuperLU
62: from . import _add_newdocs
63: 
64: __all__ = [s for s in dir() if not s.startswith('_')]
65: 
66: from scipy._lib._testutils import PytestTester
67: test = PytestTester(__name__)
68: del PytestTester
69: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_392907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', '\nLinear Solvers\n==============\n\nThe default solver is SuperLU (included in the scipy distribution),\nwhich can solve real or complex linear systems in both single and\ndouble precisions.  It is automatically replaced by UMFPACK, if\navailable.  Note that UMFPACK works in double precision only, so\nswitch it off by::\n\n    >>> use_solver(useUmfpack=False)\n\nto solve in the single precision. See also use_solver documentation.\n\nExample session::\n\n    >>> from scipy.sparse import csc_matrix, spdiags\n    >>> from numpy import array\n    >>> from scipy.sparse.linalg import spsolve, use_solver\n    >>>\n    >>> print("Inverting a sparse linear system:")\n    >>> print("The sparse matrix (constructed from diagonals):")\n    >>> a = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5)\n    >>> b = array([1, 2, 3, 4, 5])\n    >>> print("Solve: single precision complex:")\n    >>> use_solver( useUmfpack = False )\n    >>> a = a.astype(\'F\')\n    >>> x = spsolve(a, b)\n    >>> print(x)\n    >>> print("Error: ", a*x-b)\n    >>>\n    >>> print("Solve: double precision complex:")\n    >>> use_solver( useUmfpack = True )\n    >>> a = a.astype(\'D\')\n    >>> x = spsolve(a, b)\n    >>> print(x)\n    >>> print("Error: ", a*x-b)\n    >>>\n    >>> print("Solve: double precision:")\n    >>> a = a.astype(\'d\')\n    >>> x = spsolve(a, b)\n    >>> print(x)\n    >>> print("Error: ", a*x-b)\n    >>>\n    >>> print("Solve: single precision:")\n    >>> use_solver( useUmfpack = False )\n    >>> a = a.astype(\'f\')\n    >>> x = spsolve(a, b.astype(\'f\'))\n    >>> print(x)\n    >>> print("Error: ", a*x-b)\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 60, 0))

# 'from scipy.sparse.linalg.dsolve.linsolve import ' statement (line 60)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392908 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'scipy.sparse.linalg.dsolve.linsolve')

if (type(import_392908) is not StypyTypeError):

    if (import_392908 != 'pyd_module'):
        __import__(import_392908)
        sys_modules_392909 = sys.modules[import_392908]
        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'scipy.sparse.linalg.dsolve.linsolve', sys_modules_392909.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 60, 0), __file__, sys_modules_392909, sys_modules_392909.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve.linsolve import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 60, 0), 'scipy.sparse.linalg.dsolve.linsolve', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve.linsolve' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'scipy.sparse.linalg.dsolve.linsolve', import_392908)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 61, 0))

# 'from scipy.sparse.linalg.dsolve._superlu import SuperLU' statement (line 61)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'scipy.sparse.linalg.dsolve._superlu')

if (type(import_392910) is not StypyTypeError):

    if (import_392910 != 'pyd_module'):
        __import__(import_392910)
        sys_modules_392911 = sys.modules[import_392910]
        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'scipy.sparse.linalg.dsolve._superlu', sys_modules_392911.module_type_store, module_type_store, ['SuperLU'])
        nest_module(stypy.reporting.localization.Localization(__file__, 61, 0), __file__, sys_modules_392911, sys_modules_392911.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve._superlu import SuperLU

        import_from_module(stypy.reporting.localization.Localization(__file__, 61, 0), 'scipy.sparse.linalg.dsolve._superlu', None, module_type_store, ['SuperLU'], [SuperLU])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve._superlu' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'scipy.sparse.linalg.dsolve._superlu', import_392910)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'from scipy.sparse.linalg.dsolve import _add_newdocs' statement (line 62)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392912 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'scipy.sparse.linalg.dsolve')

if (type(import_392912) is not StypyTypeError):

    if (import_392912 != 'pyd_module'):
        __import__(import_392912)
        sys_modules_392913 = sys.modules[import_392912]
        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'scipy.sparse.linalg.dsolve', sys_modules_392913.module_type_store, module_type_store, ['_add_newdocs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 62, 0), __file__, sys_modules_392913, sys_modules_392913.module_type_store, module_type_store)
    else:
        from scipy.sparse.linalg.dsolve import _add_newdocs

        import_from_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'scipy.sparse.linalg.dsolve', None, module_type_store, ['_add_newdocs'], [_add_newdocs])

else:
    # Assigning a type to the variable 'scipy.sparse.linalg.dsolve' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'scipy.sparse.linalg.dsolve', import_392912)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')


# Assigning a ListComp to a Name (line 64):
# Calculating list comprehension
# Calculating comprehension expression

# Call to dir(...): (line 64)
# Processing the call keyword arguments (line 64)
kwargs_392922 = {}
# Getting the type of 'dir' (line 64)
dir_392921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'dir', False)
# Calling dir(args, kwargs) (line 64)
dir_call_result_392923 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), dir_392921, *[], **kwargs_392922)

comprehension_392924 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 11), dir_call_result_392923)
# Assigning a type to the variable 's' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 's', comprehension_392924)


# Call to startswith(...): (line 64)
# Processing the call arguments (line 64)
str_392917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 48), 'str', '_')
# Processing the call keyword arguments (line 64)
kwargs_392918 = {}
# Getting the type of 's' (line 64)
s_392915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 35), 's', False)
# Obtaining the member 'startswith' of a type (line 64)
startswith_392916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 35), s_392915, 'startswith')
# Calling startswith(args, kwargs) (line 64)
startswith_call_result_392919 = invoke(stypy.reporting.localization.Localization(__file__, 64, 35), startswith_392916, *[str_392917], **kwargs_392918)

# Applying the 'not' unary operator (line 64)
result_not__392920 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 31), 'not', startswith_call_result_392919)

# Getting the type of 's' (line 64)
s_392914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 's')
list_392925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 11), list_392925, s_392914)
# Assigning a type to the variable '__all__' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '__all__', list_392925)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'scipy._lib._testutils')

if (type(import_392926) is not StypyTypeError):

    if (import_392926 != 'pyd_module'):
        __import__(import_392926)
        sys_modules_392927 = sys.modules[import_392926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'scipy._lib._testutils', sys_modules_392927.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 0), __file__, sys_modules_392927, sys_modules_392927.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'scipy._lib._testutils', import_392926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')


# Assigning a Call to a Name (line 67):

# Call to PytestTester(...): (line 67)
# Processing the call arguments (line 67)
# Getting the type of '__name__' (line 67)
name___392929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), '__name__', False)
# Processing the call keyword arguments (line 67)
kwargs_392930 = {}
# Getting the type of 'PytestTester' (line 67)
PytestTester_392928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 67)
PytestTester_call_result_392931 = invoke(stypy.reporting.localization.Localization(__file__, 67, 7), PytestTester_392928, *[name___392929], **kwargs_392930)

# Assigning a type to the variable 'test' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'test', PytestTester_call_result_392931)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 68, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
