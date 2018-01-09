
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from .info import __doc__
4: from numpy.version import version as __version__
5: 
6: # disables OpenBLAS affinity setting of the main thread that limits
7: # python threads or processes to one core
8: import os
9: env_added = []
10: for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
11:     if envkey not in os.environ:
12:         os.environ[envkey] = '1'
13:         env_added.append(envkey)
14: from . import multiarray
15: for envkey in env_added:
16:     del os.environ[envkey]
17: del envkey
18: del env_added
19: del os
20: 
21: from . import umath
22: from . import _internal  # for freeze programs
23: from . import numerictypes as nt
24: multiarray.set_typeDict(nt.sctypeDict)
25: from . import numeric
26: from .numeric import *
27: from . import fromnumeric
28: from .fromnumeric import *
29: from . import defchararray as char
30: from . import records as rec
31: from .records import *
32: from .memmap import *
33: from .defchararray import chararray
34: from . import function_base
35: from .function_base import *
36: from . import machar
37: from .machar import *
38: from . import getlimits
39: from .getlimits import *
40: from . import shape_base
41: from .shape_base import *
42: del nt
43: 
44: from .fromnumeric import amax as max, amin as min, round_ as round
45: from .numeric import absolute as abs
46: 
47: __all__ = ['char', 'rec', 'memmap']
48: __all__ += numeric.__all__
49: __all__ += fromnumeric.__all__
50: __all__ += rec.__all__
51: __all__ += ['chararray']
52: __all__ += function_base.__all__
53: __all__ += machar.__all__
54: __all__ += getlimits.__all__
55: __all__ += shape_base.__all__
56: 
57: 
58: from numpy.testing.nosetester import _numpy_tester
59: test = _numpy_tester().test
60: bench = _numpy_tester().bench
61: 
62: # Make it possible so that ufuncs can be pickled
63: #  Here are the loading and unloading functions
64: # The name numpy.core._ufunc_reconstruct must be
65: #   available for unpickling to work.
66: def _ufunc_reconstruct(module, name):
67:     # The `fromlist` kwarg is required to ensure that `mod` points to the
68:     # inner-most module rather than the parent package when module name is
69:     # nested. This makes it possible to pickle non-toplevel ufuncs such as
70:     # scipy.special.expit for instance.
71:     mod = __import__(module, fromlist=[name])
72:     return getattr(mod, name)
73: 
74: def _ufunc_reduce(func):
75:     from pickle import whichmodule
76:     name = func.__name__
77:     return _ufunc_reconstruct, (whichmodule(func, name), name)
78: 
79: 
80: import sys
81: if sys.version_info[0] >= 3:
82:     import copyreg
83: else:
84:     import copy_reg as copyreg
85: 
86: copyreg.pickle(ufunc, _ufunc_reduce, _ufunc_reconstruct)
87: # Unclutter namespace (must keep _ufunc_reconstruct for unpickling)
88: del copyreg
89: del sys
90: del _ufunc_reduce
91: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.core.info import __doc__' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20914 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.info')

if (type(import_20914) is not StypyTypeError):

    if (import_20914 != 'pyd_module'):
        __import__(import_20914)
        sys_modules_20915 = sys.modules[import_20914]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.info', sys_modules_20915.module_type_store, module_type_store, ['__doc__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_20915, sys_modules_20915.module_type_store, module_type_store)
    else:
        from numpy.core.info import __doc__

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.info', None, module_type_store, ['__doc__'], [__doc__])

else:
    # Assigning a type to the variable 'numpy.core.info' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.core.info', import_20914)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.version import __version__' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20916 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.version')

if (type(import_20916) is not StypyTypeError):

    if (import_20916 != 'pyd_module'):
        __import__(import_20916)
        sys_modules_20917 = sys.modules[import_20916]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.version', sys_modules_20917.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_20917, sys_modules_20917.module_type_store, module_type_store)
    else:
        from numpy.version import version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.version', None, module_type_store, ['version'], [__version__])

else:
    # Assigning a type to the variable 'numpy.version' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.version', import_20916)

# Adding an alias
module_type_store.add_alias('__version__', 'version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)


# Assigning a List to a Name (line 9):

# Obtaining an instance of the builtin type 'list' (line 9)
list_20918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)

# Assigning a type to the variable 'env_added' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'env_added', list_20918)


# Obtaining an instance of the builtin type 'list' (line 10)
list_20919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_20920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'str', 'OPENBLAS_MAIN_FREE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_20919, str_20920)
# Adding element type (line 10)
str_20921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 37), 'str', 'GOTOBLAS_MAIN_FREE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 14), list_20919, str_20921)

# Testing the type of a for loop iterable (line 10)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 10, 0), list_20919)
# Getting the type of the for loop variable (line 10)
for_loop_var_20922 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 10, 0), list_20919)
# Assigning a type to the variable 'envkey' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'envkey', for_loop_var_20922)
# SSA begins for a for statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Getting the type of 'envkey' (line 11)
envkey_20923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'envkey')
# Getting the type of 'os' (line 11)
os_20924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'os')
# Obtaining the member 'environ' of a type (line 11)
environ_20925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 21), os_20924, 'environ')
# Applying the binary operator 'notin' (line 11)
result_contains_20926 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 7), 'notin', envkey_20923, environ_20925)

# Testing the type of an if condition (line 11)
if_condition_20927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 4), result_contains_20926)
# Assigning a type to the variable 'if_condition_20927' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'if_condition_20927', if_condition_20927)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Subscript (line 12):
str_20928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'str', '1')
# Getting the type of 'os' (line 12)
os_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'os')
# Obtaining the member 'environ' of a type (line 12)
environ_20930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), os_20929, 'environ')
# Getting the type of 'envkey' (line 12)
envkey_20931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'envkey')
# Storing an element on a container (line 12)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 8), environ_20930, (envkey_20931, str_20928))

# Call to append(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'envkey' (line 13)
envkey_20934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'envkey', False)
# Processing the call keyword arguments (line 13)
kwargs_20935 = {}
# Getting the type of 'env_added' (line 13)
env_added_20932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'env_added', False)
# Obtaining the member 'append' of a type (line 13)
append_20933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), env_added_20932, 'append')
# Calling append(args, kwargs) (line 13)
append_call_result_20936 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), append_20933, *[envkey_20934], **kwargs_20935)

# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.core import multiarray' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20937 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core')

if (type(import_20937) is not StypyTypeError):

    if (import_20937 != 'pyd_module'):
        __import__(import_20937)
        sys_modules_20938 = sys.modules[import_20937]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core', sys_modules_20938.module_type_store, module_type_store, ['multiarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_20938, sys_modules_20938.module_type_store, module_type_store)
    else:
        from numpy.core import multiarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core', None, module_type_store, ['multiarray'], [multiarray])

else:
    # Assigning a type to the variable 'numpy.core' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core', import_20937)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Getting the type of 'env_added' (line 15)
env_added_20939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'env_added')
# Testing the type of a for loop iterable (line 15)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 0), env_added_20939)
# Getting the type of the for loop variable (line 15)
for_loop_var_20940 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 0), env_added_20939)
# Assigning a type to the variable 'envkey' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'envkey', for_loop_var_20940)
# SSA begins for a for statement (line 15)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
# Deleting a member
# Getting the type of 'os' (line 16)
os_20941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'os')
# Obtaining the member 'environ' of a type (line 16)
environ_20942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), os_20941, 'environ')

# Obtaining the type of the subscript
# Getting the type of 'envkey' (line 16)
envkey_20943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'envkey')
# Getting the type of 'os' (line 16)
os_20944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'os')
# Obtaining the member 'environ' of a type (line 16)
environ_20945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), os_20944, 'environ')
# Obtaining the member '__getitem__' of a type (line 16)
getitem___20946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), environ_20945, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 16)
subscript_call_result_20947 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), getitem___20946, envkey_20943)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), environ_20942, subscript_call_result_20947)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 17, 0), module_type_store, 'envkey')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 18, 0), module_type_store, 'env_added')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 19, 0), module_type_store, 'os')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.core import umath' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core')

if (type(import_20948) is not StypyTypeError):

    if (import_20948 != 'pyd_module'):
        __import__(import_20948)
        sys_modules_20949 = sys.modules[import_20948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core', sys_modules_20949.module_type_store, module_type_store, ['umath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_20949, sys_modules_20949.module_type_store, module_type_store)
    else:
        from numpy.core import umath

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core', None, module_type_store, ['umath'], [umath])

else:
    # Assigning a type to the variable 'numpy.core' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.core', import_20948)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.core import _internal' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core')

if (type(import_20950) is not StypyTypeError):

    if (import_20950 != 'pyd_module'):
        __import__(import_20950)
        sys_modules_20951 = sys.modules[import_20950]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core', sys_modules_20951.module_type_store, module_type_store, ['_internal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_20951, sys_modules_20951.module_type_store, module_type_store)
    else:
        from numpy.core import _internal

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core', None, module_type_store, ['_internal'], [_internal])

else:
    # Assigning a type to the variable 'numpy.core' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.core', import_20950)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.core import nt' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core')

if (type(import_20952) is not StypyTypeError):

    if (import_20952 != 'pyd_module'):
        __import__(import_20952)
        sys_modules_20953 = sys.modules[import_20952]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core', sys_modules_20953.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_20953, sys_modules_20953.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as nt

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [nt])

else:
    # Assigning a type to the variable 'numpy.core' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.core', import_20952)

# Adding an alias
module_type_store.add_alias('nt', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Call to set_typeDict(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'nt' (line 24)
nt_20956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'nt', False)
# Obtaining the member 'sctypeDict' of a type (line 24)
sctypeDict_20957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), nt_20956, 'sctypeDict')
# Processing the call keyword arguments (line 24)
kwargs_20958 = {}
# Getting the type of 'multiarray' (line 24)
multiarray_20954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'multiarray', False)
# Obtaining the member 'set_typeDict' of a type (line 24)
set_typeDict_20955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 0), multiarray_20954, 'set_typeDict')
# Calling set_typeDict(args, kwargs) (line 24)
set_typeDict_call_result_20959 = invoke(stypy.reporting.localization.Localization(__file__, 24, 0), set_typeDict_20955, *[sctypeDict_20957], **kwargs_20958)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.core import numeric' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core')

if (type(import_20960) is not StypyTypeError):

    if (import_20960 != 'pyd_module'):
        __import__(import_20960)
        sys_modules_20961 = sys.modules[import_20960]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core', sys_modules_20961.module_type_store, module_type_store, ['numeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_20961, sys_modules_20961.module_type_store, module_type_store)
    else:
        from numpy.core import numeric

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core', None, module_type_store, ['numeric'], [numeric])

else:
    # Assigning a type to the variable 'numpy.core' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.core', import_20960)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.core.numeric import ' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20962 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.core.numeric')

if (type(import_20962) is not StypyTypeError):

    if (import_20962 != 'pyd_module'):
        __import__(import_20962)
        sys_modules_20963 = sys.modules[import_20962]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.core.numeric', sys_modules_20963.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_20963, sys_modules_20963.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.core.numeric', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.core.numeric', import_20962)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.core import fromnumeric' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core')

if (type(import_20964) is not StypyTypeError):

    if (import_20964 != 'pyd_module'):
        __import__(import_20964)
        sys_modules_20965 = sys.modules[import_20964]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core', sys_modules_20965.module_type_store, module_type_store, ['fromnumeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_20965, sys_modules_20965.module_type_store, module_type_store)
    else:
        from numpy.core import fromnumeric

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core', None, module_type_store, ['fromnumeric'], [fromnumeric])

else:
    # Assigning a type to the variable 'numpy.core' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.core', import_20964)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from numpy.core.fromnumeric import ' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20966 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.core.fromnumeric')

if (type(import_20966) is not StypyTypeError):

    if (import_20966 != 'pyd_module'):
        __import__(import_20966)
        sys_modules_20967 = sys.modules[import_20966]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.core.fromnumeric', sys_modules_20967.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_20967, sys_modules_20967.module_type_store, module_type_store)
    else:
        from numpy.core.fromnumeric import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.core.fromnumeric', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.core.fromnumeric', import_20966)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from numpy.core import char' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.core')

if (type(import_20968) is not StypyTypeError):

    if (import_20968 != 'pyd_module'):
        __import__(import_20968)
        sys_modules_20969 = sys.modules[import_20968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.core', sys_modules_20969.module_type_store, module_type_store, ['defchararray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_20969, sys_modules_20969.module_type_store, module_type_store)
    else:
        from numpy.core import defchararray as char

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.core', None, module_type_store, ['defchararray'], [char])

else:
    # Assigning a type to the variable 'numpy.core' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy.core', import_20968)

# Adding an alias
module_type_store.add_alias('char', 'defchararray')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from numpy.core import rec' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20970 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.core')

if (type(import_20970) is not StypyTypeError):

    if (import_20970 != 'pyd_module'):
        __import__(import_20970)
        sys_modules_20971 = sys.modules[import_20970]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.core', sys_modules_20971.module_type_store, module_type_store, ['records'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_20971, sys_modules_20971.module_type_store, module_type_store)
    else:
        from numpy.core import records as rec

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.core', None, module_type_store, ['records'], [rec])

else:
    # Assigning a type to the variable 'numpy.core' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.core', import_20970)

# Adding an alias
module_type_store.add_alias('rec', 'records')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 0))

# 'from numpy.core.records import ' statement (line 31)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.core.records')

if (type(import_20972) is not StypyTypeError):

    if (import_20972 != 'pyd_module'):
        __import__(import_20972)
        sys_modules_20973 = sys.modules[import_20972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.core.records', sys_modules_20973.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 31, 0), __file__, sys_modules_20973, sys_modules_20973.module_type_store, module_type_store)
    else:
        from numpy.core.records import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.core.records', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.records' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'numpy.core.records', import_20972)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from numpy.core.memmap import ' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.core.memmap')

if (type(import_20974) is not StypyTypeError):

    if (import_20974 != 'pyd_module'):
        __import__(import_20974)
        sys_modules_20975 = sys.modules[import_20974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.core.memmap', sys_modules_20975.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_20975, sys_modules_20975.module_type_store, module_type_store)
    else:
        from numpy.core.memmap import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.core.memmap', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.memmap' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy.core.memmap', import_20974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from numpy.core.defchararray import chararray' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.core.defchararray')

if (type(import_20976) is not StypyTypeError):

    if (import_20976 != 'pyd_module'):
        __import__(import_20976)
        sys_modules_20977 = sys.modules[import_20976]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.core.defchararray', sys_modules_20977.module_type_store, module_type_store, ['chararray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_20977, sys_modules_20977.module_type_store, module_type_store)
    else:
        from numpy.core.defchararray import chararray

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.core.defchararray', None, module_type_store, ['chararray'], [chararray])

else:
    # Assigning a type to the variable 'numpy.core.defchararray' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy.core.defchararray', import_20976)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from numpy.core import function_base' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.core')

if (type(import_20978) is not StypyTypeError):

    if (import_20978 != 'pyd_module'):
        __import__(import_20978)
        sys_modules_20979 = sys.modules[import_20978]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.core', sys_modules_20979.module_type_store, module_type_store, ['function_base'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_20979, sys_modules_20979.module_type_store, module_type_store)
    else:
        from numpy.core import function_base

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.core', None, module_type_store, ['function_base'], [function_base])

else:
    # Assigning a type to the variable 'numpy.core' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy.core', import_20978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from numpy.core.function_base import ' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.core.function_base')

if (type(import_20980) is not StypyTypeError):

    if (import_20980 != 'pyd_module'):
        __import__(import_20980)
        sys_modules_20981 = sys.modules[import_20980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.core.function_base', sys_modules_20981.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_20981, sys_modules_20981.module_type_store, module_type_store)
    else:
        from numpy.core.function_base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.core.function_base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.function_base' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy.core.function_base', import_20980)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from numpy.core import machar' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'numpy.core')

if (type(import_20982) is not StypyTypeError):

    if (import_20982 != 'pyd_module'):
        __import__(import_20982)
        sys_modules_20983 = sys.modules[import_20982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'numpy.core', sys_modules_20983.module_type_store, module_type_store, ['machar'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_20983, sys_modules_20983.module_type_store, module_type_store)
    else:
        from numpy.core import machar

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'numpy.core', None, module_type_store, ['machar'], [machar])

else:
    # Assigning a type to the variable 'numpy.core' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'numpy.core', import_20982)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from numpy.core.machar import ' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20984 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.core.machar')

if (type(import_20984) is not StypyTypeError):

    if (import_20984 != 'pyd_module'):
        __import__(import_20984)
        sys_modules_20985 = sys.modules[import_20984]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.core.machar', sys_modules_20985.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_20985, sys_modules_20985.module_type_store, module_type_store)
    else:
        from numpy.core.machar import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.core.machar', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.machar' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'numpy.core.machar', import_20984)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from numpy.core import getlimits' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core')

if (type(import_20986) is not StypyTypeError):

    if (import_20986 != 'pyd_module'):
        __import__(import_20986)
        sys_modules_20987 = sys.modules[import_20986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', sys_modules_20987.module_type_store, module_type_store, ['getlimits'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_20987, sys_modules_20987.module_type_store, module_type_store)
    else:
        from numpy.core import getlimits

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', None, module_type_store, ['getlimits'], [getlimits])

else:
    # Assigning a type to the variable 'numpy.core' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', import_20986)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from numpy.core.getlimits import ' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20988 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.core.getlimits')

if (type(import_20988) is not StypyTypeError):

    if (import_20988 != 'pyd_module'):
        __import__(import_20988)
        sys_modules_20989 = sys.modules[import_20988]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.core.getlimits', sys_modules_20989.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_20989, sys_modules_20989.module_type_store, module_type_store)
    else:
        from numpy.core.getlimits import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.core.getlimits', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.getlimits' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.core.getlimits', import_20988)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from numpy.core import shape_base' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20990 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.core')

if (type(import_20990) is not StypyTypeError):

    if (import_20990 != 'pyd_module'):
        __import__(import_20990)
        sys_modules_20991 = sys.modules[import_20990]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.core', sys_modules_20991.module_type_store, module_type_store, ['shape_base'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_20991, sys_modules_20991.module_type_store, module_type_store)
    else:
        from numpy.core import shape_base

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.core', None, module_type_store, ['shape_base'], [shape_base])

else:
    # Assigning a type to the variable 'numpy.core' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.core', import_20990)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 0))

# 'from numpy.core.shape_base import ' statement (line 41)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20992 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy.core.shape_base')

if (type(import_20992) is not StypyTypeError):

    if (import_20992 != 'pyd_module'):
        __import__(import_20992)
        sys_modules_20993 = sys.modules[import_20992]
        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy.core.shape_base', sys_modules_20993.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 41, 0), __file__, sys_modules_20993, sys_modules_20993.module_type_store, module_type_store)
    else:
        from numpy.core.shape_base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy.core.shape_base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core.shape_base' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'numpy.core.shape_base', import_20992)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 42, 0), module_type_store, 'nt')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from numpy.core.fromnumeric import max, min, round' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20994 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.core.fromnumeric')

if (type(import_20994) is not StypyTypeError):

    if (import_20994 != 'pyd_module'):
        __import__(import_20994)
        sys_modules_20995 = sys.modules[import_20994]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.core.fromnumeric', sys_modules_20995.module_type_store, module_type_store, ['amax', 'amin', 'round_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_20995, sys_modules_20995.module_type_store, module_type_store)
    else:
        from numpy.core.fromnumeric import amax as max, amin as min, round_ as round

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.core.fromnumeric', None, module_type_store, ['amax', 'amin', 'round_'], [max, min, round])

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.core.fromnumeric', import_20994)

# Adding an alias
module_type_store.add_alias('round', 'round_')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 0))

# 'from numpy.core.numeric import abs' statement (line 45)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_20996 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.core.numeric')

if (type(import_20996) is not StypyTypeError):

    if (import_20996 != 'pyd_module'):
        __import__(import_20996)
        sys_modules_20997 = sys.modules[import_20996]
        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.core.numeric', sys_modules_20997.module_type_store, module_type_store, ['absolute'])
        nest_module(stypy.reporting.localization.Localization(__file__, 45, 0), __file__, sys_modules_20997, sys_modules_20997.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import absolute as abs

        import_from_module(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.core.numeric', None, module_type_store, ['absolute'], [abs])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'numpy.core.numeric', import_20996)

# Adding an alias
module_type_store.add_alias('abs', 'absolute')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a List to a Name (line 47):
__all__ = ['char', 'rec', 'memmap']
module_type_store.set_exportable_members(['char', 'rec', 'memmap'])

# Obtaining an instance of the builtin type 'list' (line 47)
list_20998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_20999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_20998, str_20999)
# Adding element type (line 47)
str_21000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'str', 'rec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_20998, str_21000)
# Adding element type (line 47)
str_21001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 26), 'str', 'memmap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 10), list_20998, str_21001)

# Assigning a type to the variable '__all__' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '__all__', list_20998)

# Getting the type of '__all__' (line 48)
all___21002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '__all__')
# Getting the type of 'numeric' (line 48)
numeric_21003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'numeric')
# Obtaining the member '__all__' of a type (line 48)
all___21004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), numeric_21003, '__all__')
# Applying the binary operator '+=' (line 48)
result_iadd_21005 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 0), '+=', all___21002, all___21004)
# Assigning a type to the variable '__all__' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '__all__', result_iadd_21005)


# Getting the type of '__all__' (line 49)
all___21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), '__all__')
# Getting the type of 'fromnumeric' (line 49)
fromnumeric_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'fromnumeric')
# Obtaining the member '__all__' of a type (line 49)
all___21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), fromnumeric_21007, '__all__')
# Applying the binary operator '+=' (line 49)
result_iadd_21009 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 0), '+=', all___21006, all___21008)
# Assigning a type to the variable '__all__' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), '__all__', result_iadd_21009)


# Getting the type of '__all__' (line 50)
all___21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__')
# Getting the type of 'rec' (line 50)
rec_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'rec')
# Obtaining the member '__all__' of a type (line 50)
all___21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 11), rec_21011, '__all__')
# Applying the binary operator '+=' (line 50)
result_iadd_21013 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 0), '+=', all___21010, all___21012)
# Assigning a type to the variable '__all__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__', result_iadd_21013)


# Getting the type of '__all__' (line 51)
all___21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '__all__')

# Obtaining an instance of the builtin type 'list' (line 51)
list_21015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)
str_21016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 12), 'str', 'chararray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 11), list_21015, str_21016)

# Applying the binary operator '+=' (line 51)
result_iadd_21017 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 0), '+=', all___21014, list_21015)
# Assigning a type to the variable '__all__' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '__all__', result_iadd_21017)


# Getting the type of '__all__' (line 52)
all___21018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '__all__')
# Getting the type of 'function_base' (line 52)
function_base_21019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'function_base')
# Obtaining the member '__all__' of a type (line 52)
all___21020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 11), function_base_21019, '__all__')
# Applying the binary operator '+=' (line 52)
result_iadd_21021 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 0), '+=', all___21018, all___21020)
# Assigning a type to the variable '__all__' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), '__all__', result_iadd_21021)


# Getting the type of '__all__' (line 53)
all___21022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__all__')
# Getting the type of 'machar' (line 53)
machar_21023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'machar')
# Obtaining the member '__all__' of a type (line 53)
all___21024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 11), machar_21023, '__all__')
# Applying the binary operator '+=' (line 53)
result_iadd_21025 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 0), '+=', all___21022, all___21024)
# Assigning a type to the variable '__all__' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__all__', result_iadd_21025)


# Getting the type of '__all__' (line 54)
all___21026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__')
# Getting the type of 'getlimits' (line 54)
getlimits_21027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'getlimits')
# Obtaining the member '__all__' of a type (line 54)
all___21028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), getlimits_21027, '__all__')
# Applying the binary operator '+=' (line 54)
result_iadd_21029 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 0), '+=', all___21026, all___21028)
# Assigning a type to the variable '__all__' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__', result_iadd_21029)


# Getting the type of '__all__' (line 55)
all___21030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '__all__')
# Getting the type of 'shape_base' (line 55)
shape_base_21031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'shape_base')
# Obtaining the member '__all__' of a type (line 55)
all___21032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), shape_base_21031, '__all__')
# Applying the binary operator '+=' (line 55)
result_iadd_21033 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 0), '+=', all___21030, all___21032)
# Assigning a type to the variable '__all__' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '__all__', result_iadd_21033)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_21034 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy.testing.nosetester')

if (type(import_21034) is not StypyTypeError):

    if (import_21034 != 'pyd_module'):
        __import__(import_21034)
        sys_modules_21035 = sys.modules[import_21034]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy.testing.nosetester', sys_modules_21035.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_21035, sys_modules_21035.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy.testing.nosetester', import_21034)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Attribute to a Name (line 59):

# Call to _numpy_tester(...): (line 59)
# Processing the call keyword arguments (line 59)
kwargs_21037 = {}
# Getting the type of '_numpy_tester' (line 59)
_numpy_tester_21036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 59)
_numpy_tester_call_result_21038 = invoke(stypy.reporting.localization.Localization(__file__, 59, 7), _numpy_tester_21036, *[], **kwargs_21037)

# Obtaining the member 'test' of a type (line 59)
test_21039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 7), _numpy_tester_call_result_21038, 'test')
# Assigning a type to the variable 'test' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'test', test_21039)

# Assigning a Attribute to a Name (line 60):

# Call to _numpy_tester(...): (line 60)
# Processing the call keyword arguments (line 60)
kwargs_21041 = {}
# Getting the type of '_numpy_tester' (line 60)
_numpy_tester_21040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 60)
_numpy_tester_call_result_21042 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), _numpy_tester_21040, *[], **kwargs_21041)

# Obtaining the member 'bench' of a type (line 60)
bench_21043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), _numpy_tester_call_result_21042, 'bench')
# Assigning a type to the variable 'bench' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'bench', bench_21043)

@norecursion
def _ufunc_reconstruct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ufunc_reconstruct'
    module_type_store = module_type_store.open_function_context('_ufunc_reconstruct', 66, 0, False)
    
    # Passed parameters checking function
    _ufunc_reconstruct.stypy_localization = localization
    _ufunc_reconstruct.stypy_type_of_self = None
    _ufunc_reconstruct.stypy_type_store = module_type_store
    _ufunc_reconstruct.stypy_function_name = '_ufunc_reconstruct'
    _ufunc_reconstruct.stypy_param_names_list = ['module', 'name']
    _ufunc_reconstruct.stypy_varargs_param_name = None
    _ufunc_reconstruct.stypy_kwargs_param_name = None
    _ufunc_reconstruct.stypy_call_defaults = defaults
    _ufunc_reconstruct.stypy_call_varargs = varargs
    _ufunc_reconstruct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ufunc_reconstruct', ['module', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ufunc_reconstruct', localization, ['module', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ufunc_reconstruct(...)' code ##################

    
    # Assigning a Call to a Name (line 71):
    
    # Call to __import__(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'module' (line 71)
    module_21045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'module', False)
    # Processing the call keyword arguments (line 71)
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_21046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'name' (line 71)
    name_21047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 38), list_21046, name_21047)
    
    keyword_21048 = list_21046
    kwargs_21049 = {'fromlist': keyword_21048}
    # Getting the type of '__import__' (line 71)
    import___21044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), '__import__', False)
    # Calling __import__(args, kwargs) (line 71)
    import___call_result_21050 = invoke(stypy.reporting.localization.Localization(__file__, 71, 10), import___21044, *[module_21045], **kwargs_21049)
    
    # Assigning a type to the variable 'mod' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'mod', import___call_result_21050)
    
    # Call to getattr(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'mod' (line 72)
    mod_21052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'mod', False)
    # Getting the type of 'name' (line 72)
    name_21053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'name', False)
    # Processing the call keyword arguments (line 72)
    kwargs_21054 = {}
    # Getting the type of 'getattr' (line 72)
    getattr_21051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 72)
    getattr_call_result_21055 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), getattr_21051, *[mod_21052, name_21053], **kwargs_21054)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', getattr_call_result_21055)
    
    # ################# End of '_ufunc_reconstruct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ufunc_reconstruct' in the type store
    # Getting the type of 'stypy_return_type' (line 66)
    stypy_return_type_21056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ufunc_reconstruct'
    return stypy_return_type_21056

# Assigning a type to the variable '_ufunc_reconstruct' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), '_ufunc_reconstruct', _ufunc_reconstruct)

@norecursion
def _ufunc_reduce(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ufunc_reduce'
    module_type_store = module_type_store.open_function_context('_ufunc_reduce', 74, 0, False)
    
    # Passed parameters checking function
    _ufunc_reduce.stypy_localization = localization
    _ufunc_reduce.stypy_type_of_self = None
    _ufunc_reduce.stypy_type_store = module_type_store
    _ufunc_reduce.stypy_function_name = '_ufunc_reduce'
    _ufunc_reduce.stypy_param_names_list = ['func']
    _ufunc_reduce.stypy_varargs_param_name = None
    _ufunc_reduce.stypy_kwargs_param_name = None
    _ufunc_reduce.stypy_call_defaults = defaults
    _ufunc_reduce.stypy_call_varargs = varargs
    _ufunc_reduce.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ufunc_reduce', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ufunc_reduce', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ufunc_reduce(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 75, 4))
    
    # 'from pickle import whichmodule' statement (line 75)
    from pickle import whichmodule

    import_from_module(stypy.reporting.localization.Localization(__file__, 75, 4), 'pickle', None, module_type_store, ['whichmodule'], [whichmodule])
    
    
    # Assigning a Attribute to a Name (line 76):
    # Getting the type of 'func' (line 76)
    func_21057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'func')
    # Obtaining the member '__name__' of a type (line 76)
    name___21058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), func_21057, '__name__')
    # Assigning a type to the variable 'name' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'name', name___21058)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_21059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of '_ufunc_reconstruct' (line 77)
    _ufunc_reconstruct_21060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), '_ufunc_reconstruct')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 11), tuple_21059, _ufunc_reconstruct_21060)
    # Adding element type (line 77)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_21061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    
    # Call to whichmodule(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'func' (line 77)
    func_21063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 44), 'func', False)
    # Getting the type of 'name' (line 77)
    name_21064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 50), 'name', False)
    # Processing the call keyword arguments (line 77)
    kwargs_21065 = {}
    # Getting the type of 'whichmodule' (line 77)
    whichmodule_21062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'whichmodule', False)
    # Calling whichmodule(args, kwargs) (line 77)
    whichmodule_call_result_21066 = invoke(stypy.reporting.localization.Localization(__file__, 77, 32), whichmodule_21062, *[func_21063, name_21064], **kwargs_21065)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 32), tuple_21061, whichmodule_call_result_21066)
    # Adding element type (line 77)
    # Getting the type of 'name' (line 77)
    name_21067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 57), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 32), tuple_21061, name_21067)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 11), tuple_21059, tuple_21061)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', tuple_21059)
    
    # ################# End of '_ufunc_reduce(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ufunc_reduce' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_21068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21068)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ufunc_reduce'
    return stypy_return_type_21068

# Assigning a type to the variable '_ufunc_reduce' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_ufunc_reduce', _ufunc_reduce)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 80, 0))

# 'import sys' statement (line 80)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 80, 0), 'sys', sys, module_type_store)




# Obtaining the type of the subscript
int_21069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'int')
# Getting the type of 'sys' (line 81)
sys_21070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 81)
version_info_21071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 3), sys_21070, 'version_info')
# Obtaining the member '__getitem__' of a type (line 81)
getitem___21072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 3), version_info_21071, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 81)
subscript_call_result_21073 = invoke(stypy.reporting.localization.Localization(__file__, 81, 3), getitem___21072, int_21069)

int_21074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'int')
# Applying the binary operator '>=' (line 81)
result_ge_21075 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 3), '>=', subscript_call_result_21073, int_21074)

# Testing the type of an if condition (line 81)
if_condition_21076 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 0), result_ge_21075)
# Assigning a type to the variable 'if_condition_21076' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'if_condition_21076', if_condition_21076)
# SSA begins for if statement (line 81)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 4))

# 'import copyreg' statement (line 82)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_21077 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'copyreg')

if (type(import_21077) is not StypyTypeError):

    if (import_21077 != 'pyd_module'):
        __import__(import_21077)
        sys_modules_21078 = sys.modules[import_21077]
        import_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'copyreg', sys_modules_21078.module_type_store, module_type_store)
    else:
        import copyreg

        import_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'copyreg', copyreg, module_type_store)

else:
    # Assigning a type to the variable 'copyreg' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'copyreg', import_21077)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

# SSA branch for the else part of an if statement (line 81)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 84, 4))

# 'import copy_reg' statement (line 84)
import copy_reg as copyreg

import_module(stypy.reporting.localization.Localization(__file__, 84, 4), 'copyreg', copyreg, module_type_store)

# SSA join for if statement (line 81)
module_type_store = module_type_store.join_ssa_context()


# Call to pickle(...): (line 86)
# Processing the call arguments (line 86)
# Getting the type of 'ufunc' (line 86)
ufunc_21081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'ufunc', False)
# Getting the type of '_ufunc_reduce' (line 86)
_ufunc_reduce_21082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), '_ufunc_reduce', False)
# Getting the type of '_ufunc_reconstruct' (line 86)
_ufunc_reconstruct_21083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), '_ufunc_reconstruct', False)
# Processing the call keyword arguments (line 86)
kwargs_21084 = {}
# Getting the type of 'copyreg' (line 86)
copyreg_21079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'copyreg', False)
# Obtaining the member 'pickle' of a type (line 86)
pickle_21080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 0), copyreg_21079, 'pickle')
# Calling pickle(args, kwargs) (line 86)
pickle_call_result_21085 = invoke(stypy.reporting.localization.Localization(__file__, 86, 0), pickle_21080, *[ufunc_21081, _ufunc_reduce_21082, _ufunc_reconstruct_21083], **kwargs_21084)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 88, 0), module_type_store, 'copyreg')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 89, 0), module_type_store, 'sys')
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 90, 0), module_type_store, '_ufunc_reduce')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
