
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from .tz import *
2: 
3: __all__ = ["tzutc", "tzoffset", "tzlocal", "tzfile", "tzrange",
4:            "tzstr", "tzical", "tzwin", "tzwinlocal", "gettz",
5:            "enfold", "datetime_ambiguous", "datetime_exists"]
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from dateutil.tz.tz import ' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/tz/')
import_324810 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'dateutil.tz.tz')

if (type(import_324810) is not StypyTypeError):

    if (import_324810 != 'pyd_module'):
        __import__(import_324810)
        sys_modules_324811 = sys.modules[import_324810]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'dateutil.tz.tz', sys_modules_324811.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_324811, sys_modules_324811.module_type_store, module_type_store)
    else:
        from dateutil.tz.tz import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'dateutil.tz.tz', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'dateutil.tz.tz' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'dateutil.tz.tz', import_324810)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/tz/')


# Assigning a List to a Name (line 3):
__all__ = ['tzutc', 'tzoffset', 'tzlocal', 'tzfile', 'tzrange', 'tzstr', 'tzical', 'tzwin', 'tzwinlocal', 'gettz', 'enfold', 'datetime_ambiguous', 'datetime_exists']
module_type_store.set_exportable_members(['tzutc', 'tzoffset', 'tzlocal', 'tzfile', 'tzrange', 'tzstr', 'tzical', 'tzwin', 'tzwinlocal', 'gettz', 'enfold', 'datetime_ambiguous', 'datetime_exists'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_324812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_324813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'tzutc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324813)
# Adding element type (line 3)
str_324814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 20), 'str', 'tzoffset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324814)
# Adding element type (line 3)
str_324815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 32), 'str', 'tzlocal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324815)
# Adding element type (line 3)
str_324816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 43), 'str', 'tzfile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324816)
# Adding element type (line 3)
str_324817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 53), 'str', 'tzrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324817)
# Adding element type (line 3)
str_324818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'tzstr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324818)
# Adding element type (line 3)
str_324819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'str', 'tzical')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324819)
# Adding element type (line 3)
str_324820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 30), 'str', 'tzwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324820)
# Adding element type (line 3)
str_324821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 39), 'str', 'tzwinlocal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324821)
# Adding element type (line 3)
str_324822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 53), 'str', 'gettz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324822)
# Adding element type (line 3)
str_324823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'str', 'enfold')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324823)
# Adding element type (line 3)
str_324824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'str', 'datetime_ambiguous')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324824)
# Adding element type (line 3)
str_324825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 43), 'str', 'datetime_exists')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_324812, str_324825)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_324812)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
