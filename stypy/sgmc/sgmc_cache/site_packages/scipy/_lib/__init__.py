
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module containing private utility functions
3: ===========================================
4: 
5: The ``scipy._lib`` namespace is empty (for now).  Tests for all
6: utilities in submodules of ``_lib`` can be run with::
7: 
8:     from scipy import _lib
9:     _lib.test()
10: 
11: '''
12: from __future__ import division, print_function, absolute_import
13: 
14: from scipy._lib._testutils import PytestTester
15: test = PytestTester(__name__)
16: del PytestTester
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_711091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nModule containing private utility functions\n===========================================\n\nThe ``scipy._lib`` namespace is empty (for now).  Tests for all\nutilities in submodules of ``_lib`` can be run with::\n\n    from scipy import _lib\n    _lib.test()\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_711092 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils')

if (type(import_711092) is not StypyTypeError):

    if (import_711092 != 'pyd_module'):
        __import__(import_711092)
        sys_modules_711093 = sys.modules[import_711092]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', sys_modules_711093.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_711093, sys_modules_711093.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib._testutils', import_711092)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')


# Assigning a Call to a Name (line 15):

# Call to PytestTester(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of '__name__' (line 15)
name___711095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), '__name__', False)
# Processing the call keyword arguments (line 15)
kwargs_711096 = {}
# Getting the type of 'PytestTester' (line 15)
PytestTester_711094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 15)
PytestTester_call_result_711097 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), PytestTester_711094, *[name___711095], **kwargs_711096)

# Assigning a type to the variable 'test' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test', PytestTester_call_result_711097)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 16, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
