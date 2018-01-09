
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A sub-package for efficiently dealing with polynomials.
3: 
4: Within the documentation for this sub-package, a "finite power series,"
5: i.e., a polynomial (also referred to simply as a "series") is represented
6: by a 1-D numpy array of the polynomial's coefficients, ordered from lowest
7: order term to highest.  For example, array([1,2,3]) represents
8: ``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
9: applicable to the specific module in question, e.g., `polynomial` (which
10: "wraps" the "standard" basis) or `chebyshev`.  For optimal performance,
11: all operations on polynomials, including evaluation at an argument, are
12: implemented as operations on the coefficients.  Additional (module-specific)
13: information can be found in the docstring for the module of interest.
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: from .polynomial import Polynomial
19: from .chebyshev import Chebyshev
20: from .legendre import Legendre
21: from .hermite import Hermite
22: from .hermite_e import HermiteE
23: from .laguerre import Laguerre
24: 
25: from numpy.testing.nosetester import _numpy_tester
26: test = _numpy_tester().test
27: bench = _numpy_tester().bench
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_180551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\nA sub-package for efficiently dealing with polynomials.\n\nWithin the documentation for this sub-package, a "finite power series,"\ni.e., a polynomial (also referred to simply as a "series") is represented\nby a 1-D numpy array of the polynomial\'s coefficients, ordered from lowest\norder term to highest.  For example, array([1,2,3]) represents\n``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial\napplicable to the specific module in question, e.g., `polynomial` (which\n"wraps" the "standard" basis) or `chebyshev`.  For optimal performance,\nall operations on polynomials, including evaluation at an argument, are\nimplemented as operations on the coefficients.  Additional (module-specific)\ninformation can be found in the docstring for the module of interest.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.polynomial.polynomial import Polynomial' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180552 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.polynomial.polynomial')

if (type(import_180552) is not StypyTypeError):

    if (import_180552 != 'pyd_module'):
        __import__(import_180552)
        sys_modules_180553 = sys.modules[import_180552]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.polynomial.polynomial', sys_modules_180553.module_type_store, module_type_store, ['Polynomial'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_180553, sys_modules_180553.module_type_store, module_type_store)
    else:
        from numpy.polynomial.polynomial import Polynomial

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.polynomial.polynomial', None, module_type_store, ['Polynomial'], [Polynomial])

else:
    # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.polynomial.polynomial', import_180552)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy.polynomial.chebyshev import Chebyshev' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.polynomial.chebyshev')

if (type(import_180554) is not StypyTypeError):

    if (import_180554 != 'pyd_module'):
        __import__(import_180554)
        sys_modules_180555 = sys.modules[import_180554]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.polynomial.chebyshev', sys_modules_180555.module_type_store, module_type_store, ['Chebyshev'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_180555, sys_modules_180555.module_type_store, module_type_store)
    else:
        from numpy.polynomial.chebyshev import Chebyshev

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.polynomial.chebyshev', None, module_type_store, ['Chebyshev'], [Chebyshev])

else:
    # Assigning a type to the variable 'numpy.polynomial.chebyshev' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy.polynomial.chebyshev', import_180554)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.polynomial.legendre import Legendre' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.polynomial.legendre')

if (type(import_180556) is not StypyTypeError):

    if (import_180556 != 'pyd_module'):
        __import__(import_180556)
        sys_modules_180557 = sys.modules[import_180556]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.polynomial.legendre', sys_modules_180557.module_type_store, module_type_store, ['Legendre'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_180557, sys_modules_180557.module_type_store, module_type_store)
    else:
        from numpy.polynomial.legendre import Legendre

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.polynomial.legendre', None, module_type_store, ['Legendre'], [Legendre])

else:
    # Assigning a type to the variable 'numpy.polynomial.legendre' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.polynomial.legendre', import_180556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.polynomial.hermite import Hermite' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.polynomial.hermite')

if (type(import_180558) is not StypyTypeError):

    if (import_180558 != 'pyd_module'):
        __import__(import_180558)
        sys_modules_180559 = sys.modules[import_180558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.polynomial.hermite', sys_modules_180559.module_type_store, module_type_store, ['Hermite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_180559, sys_modules_180559.module_type_store, module_type_store)
    else:
        from numpy.polynomial.hermite import Hermite

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.polynomial.hermite', None, module_type_store, ['Hermite'], [Hermite])

else:
    # Assigning a type to the variable 'numpy.polynomial.hermite' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.polynomial.hermite', import_180558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.polynomial.hermite_e import HermiteE' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.polynomial.hermite_e')

if (type(import_180560) is not StypyTypeError):

    if (import_180560 != 'pyd_module'):
        __import__(import_180560)
        sys_modules_180561 = sys.modules[import_180560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.polynomial.hermite_e', sys_modules_180561.module_type_store, module_type_store, ['HermiteE'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_180561, sys_modules_180561.module_type_store, module_type_store)
    else:
        from numpy.polynomial.hermite_e import HermiteE

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.polynomial.hermite_e', None, module_type_store, ['HermiteE'], [HermiteE])

else:
    # Assigning a type to the variable 'numpy.polynomial.hermite_e' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.polynomial.hermite_e', import_180560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.polynomial.laguerre import Laguerre' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180562 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.polynomial.laguerre')

if (type(import_180562) is not StypyTypeError):

    if (import_180562 != 'pyd_module'):
        __import__(import_180562)
        sys_modules_180563 = sys.modules[import_180562]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.polynomial.laguerre', sys_modules_180563.module_type_store, module_type_store, ['Laguerre'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_180563, sys_modules_180563.module_type_store, module_type_store)
    else:
        from numpy.polynomial.laguerre import Laguerre

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.polynomial.laguerre', None, module_type_store, ['Laguerre'], [Laguerre])

else:
    # Assigning a type to the variable 'numpy.polynomial.laguerre' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.polynomial.laguerre', import_180562)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_180564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.testing.nosetester')

if (type(import_180564) is not StypyTypeError):

    if (import_180564 != 'pyd_module'):
        __import__(import_180564)
        sys_modules_180565 = sys.modules[import_180564]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.testing.nosetester', sys_modules_180565.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_180565, sys_modules_180565.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.testing.nosetester', import_180564)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a Attribute to a Name (line 26):

# Call to _numpy_tester(...): (line 26)
# Processing the call keyword arguments (line 26)
kwargs_180567 = {}
# Getting the type of '_numpy_tester' (line 26)
_numpy_tester_180566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 26)
_numpy_tester_call_result_180568 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), _numpy_tester_180566, *[], **kwargs_180567)

# Obtaining the member 'test' of a type (line 26)
test_180569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 7), _numpy_tester_call_result_180568, 'test')
# Assigning a type to the variable 'test' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'test', test_180569)

# Assigning a Attribute to a Name (line 27):

# Call to _numpy_tester(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_180571 = {}
# Getting the type of '_numpy_tester' (line 27)
_numpy_tester_180570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 27)
_numpy_tester_call_result_180572 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), _numpy_tester_180570, *[], **kwargs_180571)

# Obtaining the member 'bench' of a type (line 27)
bench_180573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), _numpy_tester_call_result_180572, 'bench')
# Assigning a type to the variable 'bench' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'bench', bench_180573)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
