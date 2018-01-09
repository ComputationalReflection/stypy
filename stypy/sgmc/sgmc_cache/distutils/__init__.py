
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils
2: 
3: The main package for the Python Module Distribution Utilities.  Normally
4: used from a setup script as
5: 
6:    from distutils.core import setup
7: 
8:    setup (...)
9: '''
10: 
11: import sys
12: 
13: __version__ = sys.version[:sys.version.index(' ')]
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', 'distutils\n\nThe main package for the Python Module Distribution Utilities.  Normally\nused from a setup script as\n\n   from distutils.core import setup\n\n   setup (...)\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)


# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript

# Call to index(...): (line 13)
# Processing the call arguments (line 13)
str_11669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 45), 'str', ' ')
# Processing the call keyword arguments (line 13)
kwargs_11670 = {}
# Getting the type of 'sys' (line 13)
sys_11666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'sys', False)
# Obtaining the member 'version' of a type (line 13)
version_11667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 27), sys_11666, 'version')
# Obtaining the member 'index' of a type (line 13)
index_11668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 27), version_11667, 'index')
# Calling index(args, kwargs) (line 13)
index_call_result_11671 = invoke(stypy.reporting.localization.Localization(__file__, 13, 27), index_11668, *[str_11669], **kwargs_11670)

slice_11672 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 14), None, index_call_result_11671, None)
# Getting the type of 'sys' (line 13)
sys_11673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'sys')
# Obtaining the member 'version' of a type (line 13)
version_11674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), sys_11673, 'version')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___11675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), version_11674, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_11676 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), getitem___11675, slice_11672)

# Assigning a type to the variable '__version__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__version__', subscript_call_result_11676)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
