
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command
2: 
3: Package containing implementation of all the standard Distutils
4: commands.'''
5: 
6: __revision__ = "$Id$"
7: 
8: __all__ = ['build',
9:            'build_py',
10:            'build_ext',
11:            'build_clib',
12:            'build_scripts',
13:            'clean',
14:            'install',
15:            'install_lib',
16:            'install_headers',
17:            'install_scripts',
18:            'install_data',
19:            'sdist',
20:            'register',
21:            'bdist',
22:            'bdist_dumb',
23:            'bdist_rpm',
24:            'bdist_wininst',
25:            'upload',
26:            'check',
27:            # These two are reserved for future use:
28:            #'bdist_sdux',
29:            #'bdist_pkgtool',
30:            # Note:
31:            # bdist_packager is not included because it only provides
32:            # an abstract base class
33:           ]
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_27317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'distutils.command\n\nPackage containing implementation of all the standard Distutils\ncommands.')

# Assigning a Str to a Name (line 6):
str_27318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_27318)

# Assigning a List to a Name (line 8):
__all__ = ['build', 'build_py', 'build_ext', 'build_clib', 'build_scripts', 'clean', 'install', 'install_lib', 'install_headers', 'install_scripts', 'install_data', 'sdist', 'register', 'bdist', 'bdist_dumb', 'bdist_rpm', 'bdist_wininst', 'upload', 'check']
module_type_store.set_exportable_members(['build', 'build_py', 'build_ext', 'build_clib', 'build_scripts', 'clean', 'install', 'install_lib', 'install_headers', 'install_scripts', 'install_data', 'sdist', 'register', 'bdist', 'bdist_dumb', 'bdist_rpm', 'bdist_wininst', 'upload', 'check'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_27319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_27320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27320)
# Adding element type (line 8)
str_27321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'build_py')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27321)
# Adding element type (line 8)
str_27322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'build_ext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27322)
# Adding element type (line 8)
str_27323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'build_clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27323)
# Adding element type (line 8)
str_27324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'build_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27324)
# Adding element type (line 8)
str_27325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'clean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27325)
# Adding element type (line 8)
str_27326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'install')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27326)
# Adding element type (line 8)
str_27327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'install_lib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27327)
# Adding element type (line 8)
str_27328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'install_headers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27328)
# Adding element type (line 8)
str_27329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'install_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27329)
# Adding element type (line 8)
str_27330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'install_data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27330)
# Adding element type (line 8)
str_27331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'sdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27331)
# Adding element type (line 8)
str_27332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'register')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27332)
# Adding element type (line 8)
str_27333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'bdist')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27333)
# Adding element type (line 8)
str_27334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27334)
# Adding element type (line 8)
str_27335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'str', 'bdist_rpm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27335)
# Adding element type (line 8)
str_27336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 'bdist_wininst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27336)
# Adding element type (line 8)
str_27337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'upload')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27337)
# Adding element type (line 8)
str_27338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'check')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_27319, str_27338)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_27319)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
