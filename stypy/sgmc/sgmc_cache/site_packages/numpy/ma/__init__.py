
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: =============
3: Masked Arrays
4: =============
5: 
6: Arrays sometimes contain invalid or missing data.  When doing operations
7: on such arrays, we wish to suppress invalid values, which is the purpose masked
8: arrays fulfill (an example of typical use is given below).
9: 
10: For example, examine the following array:
11: 
12: >>> x = np.array([2, 1, 3, np.nan, 5, 2, 3, np.nan])
13: 
14: When we try to calculate the mean of the data, the result is undetermined:
15: 
16: >>> np.mean(x)
17: nan
18: 
19: The mean is calculated using roughly ``np.sum(x)/len(x)``, but since
20: any number added to ``NaN`` [1]_ produces ``NaN``, this doesn't work.  Enter
21: masked arrays:
22: 
23: >>> m = np.ma.masked_array(x, np.isnan(x))
24: >>> m
25: masked_array(data = [2.0 1.0 3.0 -- 5.0 2.0 3.0 --],
26:       mask = [False False False  True False False False  True],
27:       fill_value=1e+20)
28: 
29: Here, we construct a masked array that suppress all ``NaN`` values.  We
30: may now proceed to calculate the mean of the other values:
31: 
32: >>> np.mean(m)
33: 2.6666666666666665
34: 
35: .. [1] Not-a-Number, a floating point value that is the result of an
36:        invalid operation.
37: 
38: .. moduleauthor:: Pierre Gerard-Marchant
39: .. moduleauthor:: Jarrod Millman
40: 
41: '''
42: from __future__ import division, absolute_import, print_function
43: 
44: from . import core
45: from .core import *
46: 
47: from . import extras
48: from .extras import *
49: 
50: __all__ = ['core', 'extras']
51: __all__ += core.__all__
52: __all__ += extras.__all__
53: 
54: from numpy.testing.nosetester import _numpy_tester
55: test = _numpy_tester().test
56: bench = _numpy_tester().bench
57: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_160463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', "\n=============\nMasked Arrays\n=============\n\nArrays sometimes contain invalid or missing data.  When doing operations\non such arrays, we wish to suppress invalid values, which is the purpose masked\narrays fulfill (an example of typical use is given below).\n\nFor example, examine the following array:\n\n>>> x = np.array([2, 1, 3, np.nan, 5, 2, 3, np.nan])\n\nWhen we try to calculate the mean of the data, the result is undetermined:\n\n>>> np.mean(x)\nnan\n\nThe mean is calculated using roughly ``np.sum(x)/len(x)``, but since\nany number added to ``NaN`` [1]_ produces ``NaN``, this doesn't work.  Enter\nmasked arrays:\n\n>>> m = np.ma.masked_array(x, np.isnan(x))\n>>> m\nmasked_array(data = [2.0 1.0 3.0 -- 5.0 2.0 3.0 --],\n      mask = [False False False  True False False False  True],\n      fill_value=1e+20)\n\nHere, we construct a masked array that suppress all ``NaN`` values.  We\nmay now proceed to calculate the mean of the other values:\n\n>>> np.mean(m)\n2.6666666666666665\n\n.. [1] Not-a-Number, a floating point value that is the result of an\n       invalid operation.\n\n.. moduleauthor:: Pierre Gerard-Marchant\n.. moduleauthor:: Jarrod Millman\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from numpy.ma import core' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.ma')

if (type(import_160464) is not StypyTypeError):

    if (import_160464 != 'pyd_module'):
        __import__(import_160464)
        sys_modules_160465 = sys.modules[import_160464]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.ma', sys_modules_160465.module_type_store, module_type_store, ['core'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_160465, sys_modules_160465.module_type_store, module_type_store)
    else:
        from numpy.ma import core

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.ma', None, module_type_store, ['core'], [core])

else:
    # Assigning a type to the variable 'numpy.ma' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.ma', import_160464)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'from numpy.ma.core import ' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160466 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.ma.core')

if (type(import_160466) is not StypyTypeError):

    if (import_160466 != 'pyd_module'):
        __import__(import_160466)
        sys_modules_160467 = sys.modules[import_160466]
        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.ma.core', sys_modules_160467.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 45, 0), __file__, sys_modules_160467, sys_modules_160467.module_type_store, module_type_store)
    else:
        from numpy.ma.core import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.ma.core', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.ma.core' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.ma.core', import_160466)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# 'from numpy.ma import extras' statement (line 47)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.ma')

if (type(import_160468) is not StypyTypeError):

    if (import_160468 != 'pyd_module'):
        __import__(import_160468)
        sys_modules_160469 = sys.modules[import_160468]
        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.ma', sys_modules_160469.module_type_store, module_type_store, ['extras'])
        nest_module(stypy.reporting.localization.Localization(__file__, 47, 0), __file__, sys_modules_160469, sys_modules_160469.module_type_store, module_type_store)
    else:
        from numpy.ma import extras

        import_from_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.ma', None, module_type_store, ['extras'], [extras])

else:
    # Assigning a type to the variable 'numpy.ma' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'numpy.ma', import_160468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 48, 0))

# 'from numpy.ma.extras import ' statement (line 48)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160470 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy.ma.extras')

if (type(import_160470) is not StypyTypeError):

    if (import_160470 != 'pyd_module'):
        __import__(import_160470)
        sys_modules_160471 = sys.modules[import_160470]
        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy.ma.extras', sys_modules_160471.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 48, 0), __file__, sys_modules_160471, sys_modules_160471.module_type_store, module_type_store)
    else:
        from numpy.ma.extras import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy.ma.extras', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.ma.extras' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'numpy.ma.extras', import_160470)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a List to a Name (line 50):
__all__ = ['core', 'extras']
module_type_store.set_exportable_members(['core', 'extras'])

# Obtaining an instance of the builtin type 'list' (line 50)
list_160472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
str_160473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', 'core')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_160472, str_160473)
# Adding element type (line 50)
str_160474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'str', 'extras')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_160472, str_160474)

# Assigning a type to the variable '__all__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__', list_160472)

# Getting the type of '__all__' (line 51)
all___160475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '__all__')
# Getting the type of 'core' (line 51)
core_160476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'core')
# Obtaining the member '__all__' of a type (line 51)
all___160477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), core_160476, '__all__')
# Applying the binary operator '+=' (line 51)
result_iadd_160478 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 0), '+=', all___160475, all___160477)
# Assigning a type to the variable '__all__' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '__all__', result_iadd_160478)


# Getting the type of '__all__' (line 52)
all___160479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '__all__')
# Getting the type of 'extras' (line 52)
extras_160480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'extras')
# Obtaining the member '__all__' of a type (line 52)
all___160481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), extras_160480, '__all__')
# Applying the binary operator '+=' (line 52)
result_iadd_160482 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 0), '+=', all___160479, all___160481)
# Assigning a type to the variable '__all__' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '__all__', result_iadd_160482)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 54)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'numpy.testing.nosetester')

if (type(import_160483) is not StypyTypeError):

    if (import_160483 != 'pyd_module'):
        __import__(import_160483)
        sys_modules_160484 = sys.modules[import_160483]
        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'numpy.testing.nosetester', sys_modules_160484.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 54, 0), __file__, sys_modules_160484, sys_modules_160484.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'numpy.testing.nosetester', import_160483)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a Attribute to a Name (line 55):

# Call to _numpy_tester(...): (line 55)
# Processing the call keyword arguments (line 55)
kwargs_160486 = {}
# Getting the type of '_numpy_tester' (line 55)
_numpy_tester_160485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 55)
_numpy_tester_call_result_160487 = invoke(stypy.reporting.localization.Localization(__file__, 55, 7), _numpy_tester_160485, *[], **kwargs_160486)

# Obtaining the member 'test' of a type (line 55)
test_160488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 7), _numpy_tester_call_result_160487, 'test')
# Assigning a type to the variable 'test' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'test', test_160488)

# Assigning a Attribute to a Name (line 56):

# Call to _numpy_tester(...): (line 56)
# Processing the call keyword arguments (line 56)
kwargs_160490 = {}
# Getting the type of '_numpy_tester' (line 56)
_numpy_tester_160489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 56)
_numpy_tester_call_result_160491 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), _numpy_tester_160489, *[], **kwargs_160490)

# Obtaining the member 'bench' of a type (line 56)
bench_160492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), _numpy_tester_call_result_160491, 'bench')
# Assigning a type to the variable 'bench' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'bench', bench_160492)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
