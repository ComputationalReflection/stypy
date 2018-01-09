
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: major = 2
4: 
5: try:
6:     from __svn_version__ import version
7:     version_info = (major, version)
8:     version = '%s_%s' % version_info
9: except (ImportError, ValueError):
10:     version = str(major)
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_99985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Assigning a type to the variable 'major' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'major', int_99985)


# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from __svn_version__ import version' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), '__svn_version__')

if (type(import_99986) is not StypyTypeError):

    if (import_99986 != 'pyd_module'):
        __import__(import_99986)
        sys_modules_99987 = sys.modules[import_99986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), '__svn_version__', sys_modules_99987.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_99987, sys_modules_99987.module_type_store, module_type_store)
    else:
        from __svn_version__ import version

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), '__svn_version__', None, module_type_store, ['version'], [version])

else:
    # Assigning a type to the variable '__svn_version__' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), '__svn_version__', import_99986)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a Tuple to a Name (line 7):

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_99988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'major' (line 7)
major_99989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'major')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 20), tuple_99988, major_99989)
# Adding element type (line 7)
# Getting the type of 'version' (line 7)
version_99990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 27), 'version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 20), tuple_99988, version_99990)

# Assigning a type to the variable 'version_info' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'version_info', tuple_99988)

# Assigning a BinOp to a Name (line 8):
str_99991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'str', '%s_%s')
# Getting the type of 'version_info' (line 8)
version_info_99992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'version_info')
# Applying the binary operator '%' (line 8)
result_mod_99993 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 14), '%', str_99991, version_info_99992)

# Assigning a type to the variable 'version' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'version', result_mod_99993)
# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'Tuple' branch of a try statement (line 5)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 10):

# Call to str(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'major' (line 10)
major_99995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'major', False)
# Processing the call keyword arguments (line 10)
kwargs_99996 = {}
# Getting the type of 'str' (line 10)
str_99994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', False)
# Calling str(args, kwargs) (line 10)
str_call_result_99997 = invoke(stypy.reporting.localization.Localization(__file__, 10, 14), str_99994, *[major_99995], **kwargs_99996)

# Assigning a type to the variable 'version' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'version', str_call_result_99997)
# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
