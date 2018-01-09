
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Common test support for all numpy test scripts.
2: 
3: This single module should provide all the common functionality for numpy tests
4: in a single location, so that test scripts can just import it and work right
5: away.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: from unittest import TestCase
11: 
12: from . import decorators as dec
13: from .nosetester import run_module_suite, NoseTester as Tester
14: from .utils import *
15: test = nosetester._numpy_tester().test
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_186512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'Common test support for all numpy test scripts.\n\nThis single module should provide all the common functionality for numpy tests\nin a single location, so that test scripts can just import it and work right\naway.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from unittest import TestCase' statement (line 10)
from unittest import TestCase

import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'unittest', None, module_type_store, ['TestCase'], [TestCase])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.testing import dec' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_186513 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing')

if (type(import_186513) is not StypyTypeError):

    if (import_186513 != 'pyd_module'):
        __import__(import_186513)
        sys_modules_186514 = sys.modules[import_186513]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', sys_modules_186514.module_type_store, module_type_store, ['decorators'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_186514, sys_modules_186514.module_type_store, module_type_store)
    else:
        from numpy.testing import decorators as dec

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', None, module_type_store, ['decorators'], [dec])

else:
    # Assigning a type to the variable 'numpy.testing' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.testing', import_186513)

# Adding an alias
module_type_store.add_alias('dec', 'decorators')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.testing.nosetester import run_module_suite, Tester' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_186515 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing.nosetester')

if (type(import_186515) is not StypyTypeError):

    if (import_186515 != 'pyd_module'):
        __import__(import_186515)
        sys_modules_186516 = sys.modules[import_186515]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing.nosetester', sys_modules_186516.module_type_store, module_type_store, ['run_module_suite', 'NoseTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_186516, sys_modules_186516.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import run_module_suite, NoseTester as Tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing.nosetester', None, module_type_store, ['run_module_suite', 'NoseTester'], [run_module_suite, Tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing.nosetester', import_186515)

# Adding an alias
module_type_store.add_alias('Tester', 'NoseTester')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing.utils import ' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
import_186517 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing.utils')

if (type(import_186517) is not StypyTypeError):

    if (import_186517 != 'pyd_module'):
        __import__(import_186517)
        sys_modules_186518 = sys.modules[import_186517]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing.utils', sys_modules_186518.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_186518, sys_modules_186518.module_type_store, module_type_store)
    else:
        from numpy.testing.utils import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing.utils', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing.utils', import_186517)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')


# Assigning a Attribute to a Name (line 15):

# Call to _numpy_tester(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_186521 = {}
# Getting the type of 'nosetester' (line 15)
nosetester_186519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'nosetester', False)
# Obtaining the member '_numpy_tester' of a type (line 15)
_numpy_tester_186520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), nosetester_186519, '_numpy_tester')
# Calling _numpy_tester(args, kwargs) (line 15)
_numpy_tester_call_result_186522 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), _numpy_tester_186520, *[], **kwargs_186521)

# Obtaining the member 'test' of a type (line 15)
test_186523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), _numpy_tester_call_result_186522, 'test')
# Assigning a type to the variable 'test' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test', test_186523)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
