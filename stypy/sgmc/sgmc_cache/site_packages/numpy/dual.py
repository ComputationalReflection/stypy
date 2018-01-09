
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Aliases for functions which may be accelerated by Scipy.
3: 
4: Scipy_ can be built to use accelerated or otherwise improved libraries
5: for FFTs, linear algebra, and special functions. This module allows
6: developers to transparently support these accelerated functions when
7: scipy is available but still support users who have only installed
8: Numpy.
9: 
10: .. _Scipy : http://www.scipy.org
11: 
12: '''
13: from __future__ import division, absolute_import, print_function
14: 
15: # This module should be used for functions both in numpy and scipy if
16: #  you want to use the numpy version if available but the scipy version
17: #  otherwise.
18: #  Usage  --- from numpy.dual import fft, inv
19: 
20: __all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2',
21:            'norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigvals',
22:            'eigh', 'eigvalsh', 'lstsq', 'pinv', 'cholesky', 'i0']
23: 
24: import numpy.linalg as linpkg
25: import numpy.fft as fftpkg
26: from numpy.lib import i0
27: import sys
28: 
29: 
30: fft = fftpkg.fft
31: ifft = fftpkg.ifft
32: fftn = fftpkg.fftn
33: ifftn = fftpkg.ifftn
34: fft2 = fftpkg.fft2
35: ifft2 = fftpkg.ifft2
36: 
37: norm = linpkg.norm
38: inv = linpkg.inv
39: svd = linpkg.svd
40: solve = linpkg.solve
41: det = linpkg.det
42: eig = linpkg.eig
43: eigvals = linpkg.eigvals
44: eigh = linpkg.eigh
45: eigvalsh = linpkg.eigvalsh
46: lstsq = linpkg.lstsq
47: pinv = linpkg.pinv
48: cholesky = linpkg.cholesky
49: 
50: _restore_dict = {}
51: 
52: def register_func(name, func):
53:     if name not in __all__:
54:         raise ValueError("%s not a dual function." % name)
55:     f = sys._getframe(0).f_globals
56:     _restore_dict[name] = f[name]
57:     f[name] = func
58: 
59: def restore_func(name):
60:     if name not in __all__:
61:         raise ValueError("%s not a dual function." % name)
62:     try:
63:         val = _restore_dict[name]
64:     except KeyError:
65:         return
66:     else:
67:         sys._getframe(0).f_globals[name] = val
68: 
69: def restore_all():
70:     for name in _restore_dict.keys():
71:         restore_func(name)
72: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_23934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\nAliases for functions which may be accelerated by Scipy.\n\nScipy_ can be built to use accelerated or otherwise improved libraries\nfor FFTs, linear algebra, and special functions. This module allows\ndevelopers to transparently support these accelerated functions when\nscipy is available but still support users who have only installed\nNumpy.\n\n.. _Scipy : http://www.scipy.org\n\n')

# Assigning a List to a Name (line 20):
__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2', 'norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigvals', 'eigh', 'eigvalsh', 'lstsq', 'pinv', 'cholesky', 'i0']
module_type_store.set_exportable_members(['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2', 'norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigvals', 'eigh', 'eigvalsh', 'lstsq', 'pinv', 'cholesky', 'i0'])

# Obtaining an instance of the builtin type 'list' (line 20)
list_23935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_23936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23936)
# Adding element type (line 20)
str_23937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23937)
# Adding element type (line 20)
str_23938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'fftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23938)
# Adding element type (line 20)
str_23939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'str', 'ifftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23939)
# Adding element type (line 20)
str_23940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 43), 'str', 'fft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23940)
# Adding element type (line 20)
str_23941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 51), 'str', 'ifft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23941)
# Adding element type (line 20)
str_23942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'norm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23942)
# Adding element type (line 20)
str_23943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'str', 'inv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23943)
# Adding element type (line 20)
str_23944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'svd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23944)
# Adding element type (line 20)
str_23945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'str', 'solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23945)
# Adding element type (line 20)
str_23946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'str', 'det')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23946)
# Adding element type (line 20)
str_23947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 49), 'str', 'eig')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23947)
# Adding element type (line 20)
str_23948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 56), 'str', 'eigvals')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23948)
# Adding element type (line 20)
str_23949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'eigh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23949)
# Adding element type (line 20)
str_23950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'str', 'eigvalsh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23950)
# Adding element type (line 20)
str_23951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'str', 'lstsq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23951)
# Adding element type (line 20)
str_23952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 40), 'str', 'pinv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23952)
# Adding element type (line 20)
str_23953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'str', 'cholesky')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23953)
# Adding element type (line 20)
str_23954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 60), 'str', 'i0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_23935, str_23954)

# Assigning a type to the variable '__all__' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__all__', list_23935)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import numpy.linalg' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_23955 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg')

if (type(import_23955) is not StypyTypeError):

    if (import_23955 != 'pyd_module'):
        __import__(import_23955)
        sys_modules_23956 = sys.modules[import_23955]
        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'linpkg', sys_modules_23956.module_type_store, module_type_store)
    else:
        import numpy.linalg as linpkg

        import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'linpkg', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.linalg', import_23955)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'import numpy.fft' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_23957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.fft')

if (type(import_23957) is not StypyTypeError):

    if (import_23957 != 'pyd_module'):
        __import__(import_23957)
        sys_modules_23958 = sys.modules[import_23957]
        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'fftpkg', sys_modules_23958.module_type_store, module_type_store)
    else:
        import numpy.fft as fftpkg

        import_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'fftpkg', numpy.fft, module_type_store)

else:
    # Assigning a type to the variable 'numpy.fft' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.fft', import_23957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.lib import i0' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_23959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib')

if (type(import_23959) is not StypyTypeError):

    if (import_23959 != 'pyd_module'):
        __import__(import_23959)
        sys_modules_23960 = sys.modules[import_23959]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib', sys_modules_23960.module_type_store, module_type_store, ['i0'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_23960, sys_modules_23960.module_type_store, module_type_store)
    else:
        from numpy.lib import i0

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib', None, module_type_store, ['i0'], [i0])

else:
    # Assigning a type to the variable 'numpy.lib' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib', import_23959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'import sys' statement (line 27)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'sys', sys, module_type_store)


# Assigning a Attribute to a Name (line 30):
# Getting the type of 'fftpkg' (line 30)
fftpkg_23961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'fftpkg')
# Obtaining the member 'fft' of a type (line 30)
fft_23962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 6), fftpkg_23961, 'fft')
# Assigning a type to the variable 'fft' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'fft', fft_23962)

# Assigning a Attribute to a Name (line 31):
# Getting the type of 'fftpkg' (line 31)
fftpkg_23963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'fftpkg')
# Obtaining the member 'ifft' of a type (line 31)
ifft_23964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), fftpkg_23963, 'ifft')
# Assigning a type to the variable 'ifft' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'ifft', ifft_23964)

# Assigning a Attribute to a Name (line 32):
# Getting the type of 'fftpkg' (line 32)
fftpkg_23965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 7), 'fftpkg')
# Obtaining the member 'fftn' of a type (line 32)
fftn_23966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 7), fftpkg_23965, 'fftn')
# Assigning a type to the variable 'fftn' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'fftn', fftn_23966)

# Assigning a Attribute to a Name (line 33):
# Getting the type of 'fftpkg' (line 33)
fftpkg_23967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'fftpkg')
# Obtaining the member 'ifftn' of a type (line 33)
ifftn_23968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), fftpkg_23967, 'ifftn')
# Assigning a type to the variable 'ifftn' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'ifftn', ifftn_23968)

# Assigning a Attribute to a Name (line 34):
# Getting the type of 'fftpkg' (line 34)
fftpkg_23969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'fftpkg')
# Obtaining the member 'fft2' of a type (line 34)
fft2_23970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 7), fftpkg_23969, 'fft2')
# Assigning a type to the variable 'fft2' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'fft2', fft2_23970)

# Assigning a Attribute to a Name (line 35):
# Getting the type of 'fftpkg' (line 35)
fftpkg_23971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'fftpkg')
# Obtaining the member 'ifft2' of a type (line 35)
ifft2_23972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), fftpkg_23971, 'ifft2')
# Assigning a type to the variable 'ifft2' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'ifft2', ifft2_23972)

# Assigning a Attribute to a Name (line 37):
# Getting the type of 'linpkg' (line 37)
linpkg_23973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'linpkg')
# Obtaining the member 'norm' of a type (line 37)
norm_23974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 7), linpkg_23973, 'norm')
# Assigning a type to the variable 'norm' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'norm', norm_23974)

# Assigning a Attribute to a Name (line 38):
# Getting the type of 'linpkg' (line 38)
linpkg_23975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 6), 'linpkg')
# Obtaining the member 'inv' of a type (line 38)
inv_23976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 6), linpkg_23975, 'inv')
# Assigning a type to the variable 'inv' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'inv', inv_23976)

# Assigning a Attribute to a Name (line 39):
# Getting the type of 'linpkg' (line 39)
linpkg_23977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 6), 'linpkg')
# Obtaining the member 'svd' of a type (line 39)
svd_23978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 6), linpkg_23977, 'svd')
# Assigning a type to the variable 'svd' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'svd', svd_23978)

# Assigning a Attribute to a Name (line 40):
# Getting the type of 'linpkg' (line 40)
linpkg_23979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'linpkg')
# Obtaining the member 'solve' of a type (line 40)
solve_23980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), linpkg_23979, 'solve')
# Assigning a type to the variable 'solve' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'solve', solve_23980)

# Assigning a Attribute to a Name (line 41):
# Getting the type of 'linpkg' (line 41)
linpkg_23981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 6), 'linpkg')
# Obtaining the member 'det' of a type (line 41)
det_23982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 6), linpkg_23981, 'det')
# Assigning a type to the variable 'det' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'det', det_23982)

# Assigning a Attribute to a Name (line 42):
# Getting the type of 'linpkg' (line 42)
linpkg_23983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 6), 'linpkg')
# Obtaining the member 'eig' of a type (line 42)
eig_23984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 6), linpkg_23983, 'eig')
# Assigning a type to the variable 'eig' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'eig', eig_23984)

# Assigning a Attribute to a Name (line 43):
# Getting the type of 'linpkg' (line 43)
linpkg_23985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 10), 'linpkg')
# Obtaining the member 'eigvals' of a type (line 43)
eigvals_23986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 10), linpkg_23985, 'eigvals')
# Assigning a type to the variable 'eigvals' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'eigvals', eigvals_23986)

# Assigning a Attribute to a Name (line 44):
# Getting the type of 'linpkg' (line 44)
linpkg_23987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'linpkg')
# Obtaining the member 'eigh' of a type (line 44)
eigh_23988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), linpkg_23987, 'eigh')
# Assigning a type to the variable 'eigh' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'eigh', eigh_23988)

# Assigning a Attribute to a Name (line 45):
# Getting the type of 'linpkg' (line 45)
linpkg_23989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'linpkg')
# Obtaining the member 'eigvalsh' of a type (line 45)
eigvalsh_23990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), linpkg_23989, 'eigvalsh')
# Assigning a type to the variable 'eigvalsh' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'eigvalsh', eigvalsh_23990)

# Assigning a Attribute to a Name (line 46):
# Getting the type of 'linpkg' (line 46)
linpkg_23991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'linpkg')
# Obtaining the member 'lstsq' of a type (line 46)
lstsq_23992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), linpkg_23991, 'lstsq')
# Assigning a type to the variable 'lstsq' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'lstsq', lstsq_23992)

# Assigning a Attribute to a Name (line 47):
# Getting the type of 'linpkg' (line 47)
linpkg_23993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'linpkg')
# Obtaining the member 'pinv' of a type (line 47)
pinv_23994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 7), linpkg_23993, 'pinv')
# Assigning a type to the variable 'pinv' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'pinv', pinv_23994)

# Assigning a Attribute to a Name (line 48):
# Getting the type of 'linpkg' (line 48)
linpkg_23995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'linpkg')
# Obtaining the member 'cholesky' of a type (line 48)
cholesky_23996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), linpkg_23995, 'cholesky')
# Assigning a type to the variable 'cholesky' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'cholesky', cholesky_23996)

# Assigning a Dict to a Name (line 50):

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_23997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)

# Assigning a type to the variable '_restore_dict' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_restore_dict', dict_23997)

@norecursion
def register_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'register_func'
    module_type_store = module_type_store.open_function_context('register_func', 52, 0, False)
    
    # Passed parameters checking function
    register_func.stypy_localization = localization
    register_func.stypy_type_of_self = None
    register_func.stypy_type_store = module_type_store
    register_func.stypy_function_name = 'register_func'
    register_func.stypy_param_names_list = ['name', 'func']
    register_func.stypy_varargs_param_name = None
    register_func.stypy_kwargs_param_name = None
    register_func.stypy_call_defaults = defaults
    register_func.stypy_call_varargs = varargs
    register_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'register_func', ['name', 'func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'register_func', localization, ['name', 'func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'register_func(...)' code ##################

    
    
    # Getting the type of 'name' (line 53)
    name_23998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'name')
    # Getting the type of '__all__' (line 53)
    all___23999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), '__all__')
    # Applying the binary operator 'notin' (line 53)
    result_contains_24000 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), 'notin', name_23998, all___23999)
    
    # Testing the type of an if condition (line 53)
    if_condition_24001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_contains_24000)
    # Assigning a type to the variable 'if_condition_24001' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_24001', if_condition_24001)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 54)
    # Processing the call arguments (line 54)
    str_24003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'str', '%s not a dual function.')
    # Getting the type of 'name' (line 54)
    name_24004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'name', False)
    # Applying the binary operator '%' (line 54)
    result_mod_24005 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 25), '%', str_24003, name_24004)
    
    # Processing the call keyword arguments (line 54)
    kwargs_24006 = {}
    # Getting the type of 'ValueError' (line 54)
    ValueError_24002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 54)
    ValueError_call_result_24007 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), ValueError_24002, *[result_mod_24005], **kwargs_24006)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 54, 8), ValueError_call_result_24007, 'raise parameter', BaseException)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 55):
    
    # Call to _getframe(...): (line 55)
    # Processing the call arguments (line 55)
    int_24010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_24011 = {}
    # Getting the type of 'sys' (line 55)
    sys_24008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'sys', False)
    # Obtaining the member '_getframe' of a type (line 55)
    _getframe_24009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), sys_24008, '_getframe')
    # Calling _getframe(args, kwargs) (line 55)
    _getframe_call_result_24012 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), _getframe_24009, *[int_24010], **kwargs_24011)
    
    # Obtaining the member 'f_globals' of a type (line 55)
    f_globals_24013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), _getframe_call_result_24012, 'f_globals')
    # Assigning a type to the variable 'f' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'f', f_globals_24013)
    
    # Assigning a Subscript to a Subscript (line 56):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 56)
    name_24014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'name')
    # Getting the type of 'f' (line 56)
    f_24015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'f')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___24016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 26), f_24015, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_24017 = invoke(stypy.reporting.localization.Localization(__file__, 56, 26), getitem___24016, name_24014)
    
    # Getting the type of '_restore_dict' (line 56)
    _restore_dict_24018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), '_restore_dict')
    # Getting the type of 'name' (line 56)
    name_24019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'name')
    # Storing an element on a container (line 56)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), _restore_dict_24018, (name_24019, subscript_call_result_24017))
    
    # Assigning a Name to a Subscript (line 57):
    # Getting the type of 'func' (line 57)
    func_24020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'func')
    # Getting the type of 'f' (line 57)
    f_24021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'f')
    # Getting the type of 'name' (line 57)
    name_24022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 6), 'name')
    # Storing an element on a container (line 57)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 4), f_24021, (name_24022, func_24020))
    
    # ################# End of 'register_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'register_func' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_24023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24023)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'register_func'
    return stypy_return_type_24023

# Assigning a type to the variable 'register_func' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'register_func', register_func)

@norecursion
def restore_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'restore_func'
    module_type_store = module_type_store.open_function_context('restore_func', 59, 0, False)
    
    # Passed parameters checking function
    restore_func.stypy_localization = localization
    restore_func.stypy_type_of_self = None
    restore_func.stypy_type_store = module_type_store
    restore_func.stypy_function_name = 'restore_func'
    restore_func.stypy_param_names_list = ['name']
    restore_func.stypy_varargs_param_name = None
    restore_func.stypy_kwargs_param_name = None
    restore_func.stypy_call_defaults = defaults
    restore_func.stypy_call_varargs = varargs
    restore_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'restore_func', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'restore_func', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'restore_func(...)' code ##################

    
    
    # Getting the type of 'name' (line 60)
    name_24024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'name')
    # Getting the type of '__all__' (line 60)
    all___24025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), '__all__')
    # Applying the binary operator 'notin' (line 60)
    result_contains_24026 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'notin', name_24024, all___24025)
    
    # Testing the type of an if condition (line 60)
    if_condition_24027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_contains_24026)
    # Assigning a type to the variable 'if_condition_24027' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_24027', if_condition_24027)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 61)
    # Processing the call arguments (line 61)
    str_24029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'str', '%s not a dual function.')
    # Getting the type of 'name' (line 61)
    name_24030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 53), 'name', False)
    # Applying the binary operator '%' (line 61)
    result_mod_24031 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 25), '%', str_24029, name_24030)
    
    # Processing the call keyword arguments (line 61)
    kwargs_24032 = {}
    # Getting the type of 'ValueError' (line 61)
    ValueError_24028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 61)
    ValueError_call_result_24033 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), ValueError_24028, *[result_mod_24031], **kwargs_24032)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 8), ValueError_call_result_24033, 'raise parameter', BaseException)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 63):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 63)
    name_24034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'name')
    # Getting the type of '_restore_dict' (line 63)
    _restore_dict_24035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), '_restore_dict')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___24036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), _restore_dict_24035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_24037 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), getitem___24036, name_24034)
    
    # Assigning a type to the variable 'val' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'val', subscript_call_result_24037)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except 'KeyError' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', types.NoneType)
    # SSA branch for the else branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Name to a Subscript (line 67):
    # Getting the type of 'val' (line 67)
    val_24038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 43), 'val')
    
    # Call to _getframe(...): (line 67)
    # Processing the call arguments (line 67)
    int_24041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'int')
    # Processing the call keyword arguments (line 67)
    kwargs_24042 = {}
    # Getting the type of 'sys' (line 67)
    sys_24039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'sys', False)
    # Obtaining the member '_getframe' of a type (line 67)
    _getframe_24040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), sys_24039, '_getframe')
    # Calling _getframe(args, kwargs) (line 67)
    _getframe_call_result_24043 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), _getframe_24040, *[int_24041], **kwargs_24042)
    
    # Obtaining the member 'f_globals' of a type (line 67)
    f_globals_24044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), _getframe_call_result_24043, 'f_globals')
    # Getting the type of 'name' (line 67)
    name_24045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 35), 'name')
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), f_globals_24044, (name_24045, val_24038))
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'restore_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'restore_func' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_24046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24046)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'restore_func'
    return stypy_return_type_24046

# Assigning a type to the variable 'restore_func' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'restore_func', restore_func)

@norecursion
def restore_all(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'restore_all'
    module_type_store = module_type_store.open_function_context('restore_all', 69, 0, False)
    
    # Passed parameters checking function
    restore_all.stypy_localization = localization
    restore_all.stypy_type_of_self = None
    restore_all.stypy_type_store = module_type_store
    restore_all.stypy_function_name = 'restore_all'
    restore_all.stypy_param_names_list = []
    restore_all.stypy_varargs_param_name = None
    restore_all.stypy_kwargs_param_name = None
    restore_all.stypy_call_defaults = defaults
    restore_all.stypy_call_varargs = varargs
    restore_all.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'restore_all', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'restore_all', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'restore_all(...)' code ##################

    
    
    # Call to keys(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_24049 = {}
    # Getting the type of '_restore_dict' (line 70)
    _restore_dict_24047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), '_restore_dict', False)
    # Obtaining the member 'keys' of a type (line 70)
    keys_24048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), _restore_dict_24047, 'keys')
    # Calling keys(args, kwargs) (line 70)
    keys_call_result_24050 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), keys_24048, *[], **kwargs_24049)
    
    # Testing the type of a for loop iterable (line 70)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 4), keys_call_result_24050)
    # Getting the type of the for loop variable (line 70)
    for_loop_var_24051 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 4), keys_call_result_24050)
    # Assigning a type to the variable 'name' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'name', for_loop_var_24051)
    # SSA begins for a for statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to restore_func(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'name' (line 71)
    name_24053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'name', False)
    # Processing the call keyword arguments (line 71)
    kwargs_24054 = {}
    # Getting the type of 'restore_func' (line 71)
    restore_func_24052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'restore_func', False)
    # Calling restore_func(args, kwargs) (line 71)
    restore_func_call_result_24055 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), restore_func_24052, *[name_24053], **kwargs_24054)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'restore_all(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'restore_all' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_24056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'restore_all'
    return stypy_return_type_24056

# Assigning a type to the variable 'restore_all' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'restore_all', restore_all)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
