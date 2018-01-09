
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import math
4: 
5: from .info import __doc__
6: from numpy.version import version as __version__
7: 
8: from .type_check import *
9: from .index_tricks import *
10: from .function_base import *
11: from .nanfunctions import *
12: from .shape_base import *
13: from .stride_tricks import *
14: from .twodim_base import *
15: from .ufunclike import *
16: 
17: from . import scimath as emath
18: from .polynomial import *
19: #import convertcode
20: from .utils import *
21: from .arraysetops import *
22: from .npyio import *
23: from .financial import *
24: from .arrayterator import Arrayterator
25: from .arraypad import *
26: from ._version import *
27: 
28: __all__ = ['emath', 'math']
29: __all__ += type_check.__all__
30: __all__ += index_tricks.__all__
31: __all__ += function_base.__all__
32: __all__ += shape_base.__all__
33: __all__ += stride_tricks.__all__
34: __all__ += twodim_base.__all__
35: __all__ += ufunclike.__all__
36: __all__ += arraypad.__all__
37: __all__ += polynomial.__all__
38: __all__ += utils.__all__
39: __all__ += arraysetops.__all__
40: __all__ += npyio.__all__
41: __all__ += financial.__all__
42: __all__ += nanfunctions.__all__
43: 
44: from numpy.testing.nosetester import _numpy_tester
45: test = _numpy_tester().test
46: bench = _numpy_tester().bench
47: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import math' statement (line 3)
import math

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.lib.info import __doc__' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134157 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.lib.info')

if (type(import_134157) is not StypyTypeError):

    if (import_134157 != 'pyd_module'):
        __import__(import_134157)
        sys_modules_134158 = sys.modules[import_134157]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.lib.info', sys_modules_134158.module_type_store, module_type_store, ['__doc__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_134158, sys_modules_134158.module_type_store, module_type_store)
    else:
        from numpy.lib.info import __doc__

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.lib.info', None, module_type_store, ['__doc__'], [__doc__])

else:
    # Assigning a type to the variable 'numpy.lib.info' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.lib.info', import_134157)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.version import __version__' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134159 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.version')

if (type(import_134159) is not StypyTypeError):

    if (import_134159 != 'pyd_module'):
        __import__(import_134159)
        sys_modules_134160 = sys.modules[import_134159]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.version', sys_modules_134160.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_134160, sys_modules_134160.module_type_store, module_type_store)
    else:
        from numpy.version import version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.version', None, module_type_store, ['version'], [__version__])

else:
    # Assigning a type to the variable 'numpy.version' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.version', import_134159)

# Adding an alias
module_type_store.add_alias('__version__', 'version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.lib.type_check import ' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.lib.type_check')

if (type(import_134161) is not StypyTypeError):

    if (import_134161 != 'pyd_module'):
        __import__(import_134161)
        sys_modules_134162 = sys.modules[import_134161]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.lib.type_check', sys_modules_134162.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_134162, sys_modules_134162.module_type_store, module_type_store)
    else:
        from numpy.lib.type_check import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.lib.type_check', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.type_check' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.lib.type_check', import_134161)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.lib.index_tricks import ' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.lib.index_tricks')

if (type(import_134163) is not StypyTypeError):

    if (import_134163 != 'pyd_module'):
        __import__(import_134163)
        sys_modules_134164 = sys.modules[import_134163]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.lib.index_tricks', sys_modules_134164.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_134164, sys_modules_134164.module_type_store, module_type_store)
    else:
        from numpy.lib.index_tricks import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.lib.index_tricks', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.index_tricks' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.lib.index_tricks', import_134163)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.lib.function_base import ' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.lib.function_base')

if (type(import_134165) is not StypyTypeError):

    if (import_134165 != 'pyd_module'):
        __import__(import_134165)
        sys_modules_134166 = sys.modules[import_134165]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.lib.function_base', sys_modules_134166.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_134166, sys_modules_134166.module_type_store, module_type_store)
    else:
        from numpy.lib.function_base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.lib.function_base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.function_base' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.lib.function_base', import_134165)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.lib.nanfunctions import ' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134167 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.lib.nanfunctions')

if (type(import_134167) is not StypyTypeError):

    if (import_134167 != 'pyd_module'):
        __import__(import_134167)
        sys_modules_134168 = sys.modules[import_134167]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.lib.nanfunctions', sys_modules_134168.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_134168, sys_modules_134168.module_type_store, module_type_store)
    else:
        from numpy.lib.nanfunctions import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.lib.nanfunctions', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.nanfunctions' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.lib.nanfunctions', import_134167)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.lib.shape_base import ' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134169 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib.shape_base')

if (type(import_134169) is not StypyTypeError):

    if (import_134169 != 'pyd_module'):
        __import__(import_134169)
        sys_modules_134170 = sys.modules[import_134169]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib.shape_base', sys_modules_134170.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_134170, sys_modules_134170.module_type_store, module_type_store)
    else:
        from numpy.lib.shape_base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib.shape_base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.shape_base' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.lib.shape_base', import_134169)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.lib.stride_tricks import ' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib.stride_tricks')

if (type(import_134171) is not StypyTypeError):

    if (import_134171 != 'pyd_module'):
        __import__(import_134171)
        sys_modules_134172 = sys.modules[import_134171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib.stride_tricks', sys_modules_134172.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_134172, sys_modules_134172.module_type_store, module_type_store)
    else:
        from numpy.lib.stride_tricks import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib.stride_tricks', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.stride_tricks' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.lib.stride_tricks', import_134171)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.lib.twodim_base import ' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.twodim_base')

if (type(import_134173) is not StypyTypeError):

    if (import_134173 != 'pyd_module'):
        __import__(import_134173)
        sys_modules_134174 = sys.modules[import_134173]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.twodim_base', sys_modules_134174.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_134174, sys_modules_134174.module_type_store, module_type_store)
    else:
        from numpy.lib.twodim_base import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.twodim_base', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.twodim_base' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.lib.twodim_base', import_134173)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.lib.ufunclike import ' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.lib.ufunclike')

if (type(import_134175) is not StypyTypeError):

    if (import_134175 != 'pyd_module'):
        __import__(import_134175)
        sys_modules_134176 = sys.modules[import_134175]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.lib.ufunclike', sys_modules_134176.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_134176, sys_modules_134176.module_type_store, module_type_store)
    else:
        from numpy.lib.ufunclike import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.lib.ufunclike', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.ufunclike' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.lib.ufunclike', import_134175)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from numpy.lib import emath' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134177 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib')

if (type(import_134177) is not StypyTypeError):

    if (import_134177 != 'pyd_module'):
        __import__(import_134177)
        sys_modules_134178 = sys.modules[import_134177]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib', sys_modules_134178.module_type_store, module_type_store, ['scimath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_134178, sys_modules_134178.module_type_store, module_type_store)
    else:
        from numpy.lib import scimath as emath

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib', None, module_type_store, ['scimath'], [emath])

else:
    # Assigning a type to the variable 'numpy.lib' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'numpy.lib', import_134177)

# Adding an alias
module_type_store.add_alias('emath', 'scimath')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from numpy.lib.polynomial import ' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134179 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.polynomial')

if (type(import_134179) is not StypyTypeError):

    if (import_134179 != 'pyd_module'):
        __import__(import_134179)
        sys_modules_134180 = sys.modules[import_134179]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.polynomial', sys_modules_134180.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_134180, sys_modules_134180.module_type_store, module_type_store)
    else:
        from numpy.lib.polynomial import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.polynomial', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.polynomial' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'numpy.lib.polynomial', import_134179)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from numpy.lib.utils import ' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134181 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.lib.utils')

if (type(import_134181) is not StypyTypeError):

    if (import_134181 != 'pyd_module'):
        __import__(import_134181)
        sys_modules_134182 = sys.modules[import_134181]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.lib.utils', sys_modules_134182.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_134182, sys_modules_134182.module_type_store, module_type_store)
    else:
        from numpy.lib.utils import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.lib.utils', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.utils' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy.lib.utils', import_134181)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy.lib.arraysetops import ' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134183 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.lib.arraysetops')

if (type(import_134183) is not StypyTypeError):

    if (import_134183 != 'pyd_module'):
        __import__(import_134183)
        sys_modules_134184 = sys.modules[import_134183]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.lib.arraysetops', sys_modules_134184.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_134184, sys_modules_134184.module_type_store, module_type_store)
    else:
        from numpy.lib.arraysetops import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.lib.arraysetops', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.arraysetops' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy.lib.arraysetops', import_134183)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.lib.npyio import ' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.lib.npyio')

if (type(import_134185) is not StypyTypeError):

    if (import_134185 != 'pyd_module'):
        __import__(import_134185)
        sys_modules_134186 = sys.modules[import_134185]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.lib.npyio', sys_modules_134186.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_134186, sys_modules_134186.module_type_store, module_type_store)
    else:
        from numpy.lib.npyio import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.lib.npyio', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.npyio' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.lib.npyio', import_134185)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from numpy.lib.financial import ' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.financial')

if (type(import_134187) is not StypyTypeError):

    if (import_134187 != 'pyd_module'):
        __import__(import_134187)
        sys_modules_134188 = sys.modules[import_134187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.financial', sys_modules_134188.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_134188, sys_modules_134188.module_type_store, module_type_store)
    else:
        from numpy.lib.financial import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.financial', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.financial' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy.lib.financial', import_134187)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.lib.arrayterator import Arrayterator' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.arrayterator')

if (type(import_134189) is not StypyTypeError):

    if (import_134189 != 'pyd_module'):
        __import__(import_134189)
        sys_modules_134190 = sys.modules[import_134189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.arrayterator', sys_modules_134190.module_type_store, module_type_store, ['Arrayterator'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_134190, sys_modules_134190.module_type_store, module_type_store)
    else:
        from numpy.lib.arrayterator import Arrayterator

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.arrayterator', None, module_type_store, ['Arrayterator'], [Arrayterator])

else:
    # Assigning a type to the variable 'numpy.lib.arrayterator' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.arrayterator', import_134189)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.lib.arraypad import ' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.lib.arraypad')

if (type(import_134191) is not StypyTypeError):

    if (import_134191 != 'pyd_module'):
        __import__(import_134191)
        sys_modules_134192 = sys.modules[import_134191]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.lib.arraypad', sys_modules_134192.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_134192, sys_modules_134192.module_type_store, module_type_store)
    else:
        from numpy.lib.arraypad import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.lib.arraypad', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib.arraypad' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.lib.arraypad', import_134191)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.lib._version import ' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134193 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib._version')

if (type(import_134193) is not StypyTypeError):

    if (import_134193 != 'pyd_module'):
        __import__(import_134193)
        sys_modules_134194 = sys.modules[import_134193]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib._version', sys_modules_134194.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_134194, sys_modules_134194.module_type_store, module_type_store)
    else:
        from numpy.lib._version import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib._version', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib._version' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.lib._version', import_134193)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 28):
__all__ = ['emath', 'math']
module_type_store.set_exportable_members(['emath', 'math'])

# Obtaining an instance of the builtin type 'list' (line 28)
list_134195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_134196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', 'emath')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_134195, str_134196)
# Adding element type (line 28)
str_134197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'str', 'math')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 10), list_134195, str_134197)

# Assigning a type to the variable '__all__' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '__all__', list_134195)

# Getting the type of '__all__' (line 29)
all___134198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__')
# Getting the type of 'type_check' (line 29)
type_check_134199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'type_check')
# Obtaining the member '__all__' of a type (line 29)
all___134200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), type_check_134199, '__all__')
# Applying the binary operator '+=' (line 29)
result_iadd_134201 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 0), '+=', all___134198, all___134200)
# Assigning a type to the variable '__all__' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__', result_iadd_134201)


# Getting the type of '__all__' (line 30)
all___134202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '__all__')
# Getting the type of 'index_tricks' (line 30)
index_tricks_134203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'index_tricks')
# Obtaining the member '__all__' of a type (line 30)
all___134204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 11), index_tricks_134203, '__all__')
# Applying the binary operator '+=' (line 30)
result_iadd_134205 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 0), '+=', all___134202, all___134204)
# Assigning a type to the variable '__all__' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '__all__', result_iadd_134205)


# Getting the type of '__all__' (line 31)
all___134206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '__all__')
# Getting the type of 'function_base' (line 31)
function_base_134207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'function_base')
# Obtaining the member '__all__' of a type (line 31)
all___134208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), function_base_134207, '__all__')
# Applying the binary operator '+=' (line 31)
result_iadd_134209 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 0), '+=', all___134206, all___134208)
# Assigning a type to the variable '__all__' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '__all__', result_iadd_134209)


# Getting the type of '__all__' (line 32)
all___134210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '__all__')
# Getting the type of 'shape_base' (line 32)
shape_base_134211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'shape_base')
# Obtaining the member '__all__' of a type (line 32)
all___134212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), shape_base_134211, '__all__')
# Applying the binary operator '+=' (line 32)
result_iadd_134213 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 0), '+=', all___134210, all___134212)
# Assigning a type to the variable '__all__' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '__all__', result_iadd_134213)


# Getting the type of '__all__' (line 33)
all___134214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '__all__')
# Getting the type of 'stride_tricks' (line 33)
stride_tricks_134215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'stride_tricks')
# Obtaining the member '__all__' of a type (line 33)
all___134216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), stride_tricks_134215, '__all__')
# Applying the binary operator '+=' (line 33)
result_iadd_134217 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 0), '+=', all___134214, all___134216)
# Assigning a type to the variable '__all__' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '__all__', result_iadd_134217)


# Getting the type of '__all__' (line 34)
all___134218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '__all__')
# Getting the type of 'twodim_base' (line 34)
twodim_base_134219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'twodim_base')
# Obtaining the member '__all__' of a type (line 34)
all___134220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), twodim_base_134219, '__all__')
# Applying the binary operator '+=' (line 34)
result_iadd_134221 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 0), '+=', all___134218, all___134220)
# Assigning a type to the variable '__all__' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), '__all__', result_iadd_134221)


# Getting the type of '__all__' (line 35)
all___134222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '__all__')
# Getting the type of 'ufunclike' (line 35)
ufunclike_134223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'ufunclike')
# Obtaining the member '__all__' of a type (line 35)
all___134224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), ufunclike_134223, '__all__')
# Applying the binary operator '+=' (line 35)
result_iadd_134225 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 0), '+=', all___134222, all___134224)
# Assigning a type to the variable '__all__' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '__all__', result_iadd_134225)


# Getting the type of '__all__' (line 36)
all___134226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '__all__')
# Getting the type of 'arraypad' (line 36)
arraypad_134227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'arraypad')
# Obtaining the member '__all__' of a type (line 36)
all___134228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), arraypad_134227, '__all__')
# Applying the binary operator '+=' (line 36)
result_iadd_134229 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 0), '+=', all___134226, all___134228)
# Assigning a type to the variable '__all__' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '__all__', result_iadd_134229)


# Getting the type of '__all__' (line 37)
all___134230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '__all__')
# Getting the type of 'polynomial' (line 37)
polynomial_134231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'polynomial')
# Obtaining the member '__all__' of a type (line 37)
all___134232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 11), polynomial_134231, '__all__')
# Applying the binary operator '+=' (line 37)
result_iadd_134233 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 0), '+=', all___134230, all___134232)
# Assigning a type to the variable '__all__' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '__all__', result_iadd_134233)


# Getting the type of '__all__' (line 38)
all___134234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '__all__')
# Getting the type of 'utils' (line 38)
utils_134235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'utils')
# Obtaining the member '__all__' of a type (line 38)
all___134236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), utils_134235, '__all__')
# Applying the binary operator '+=' (line 38)
result_iadd_134237 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 0), '+=', all___134234, all___134236)
# Assigning a type to the variable '__all__' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), '__all__', result_iadd_134237)


# Getting the type of '__all__' (line 39)
all___134238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '__all__')
# Getting the type of 'arraysetops' (line 39)
arraysetops_134239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'arraysetops')
# Obtaining the member '__all__' of a type (line 39)
all___134240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), arraysetops_134239, '__all__')
# Applying the binary operator '+=' (line 39)
result_iadd_134241 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 0), '+=', all___134238, all___134240)
# Assigning a type to the variable '__all__' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '__all__', result_iadd_134241)


# Getting the type of '__all__' (line 40)
all___134242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '__all__')
# Getting the type of 'npyio' (line 40)
npyio_134243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'npyio')
# Obtaining the member '__all__' of a type (line 40)
all___134244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), npyio_134243, '__all__')
# Applying the binary operator '+=' (line 40)
result_iadd_134245 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 0), '+=', all___134242, all___134244)
# Assigning a type to the variable '__all__' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '__all__', result_iadd_134245)


# Getting the type of '__all__' (line 41)
all___134246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '__all__')
# Getting the type of 'financial' (line 41)
financial_134247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'financial')
# Obtaining the member '__all__' of a type (line 41)
all___134248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 11), financial_134247, '__all__')
# Applying the binary operator '+=' (line 41)
result_iadd_134249 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 0), '+=', all___134246, all___134248)
# Assigning a type to the variable '__all__' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '__all__', result_iadd_134249)


# Getting the type of '__all__' (line 42)
all___134250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '__all__')
# Getting the type of 'nanfunctions' (line 42)
nanfunctions_134251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'nanfunctions')
# Obtaining the member '__all__' of a type (line 42)
all___134252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), nanfunctions_134251, '__all__')
# Applying the binary operator '+=' (line 42)
result_iadd_134253 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 0), '+=', all___134250, all___134252)
# Assigning a type to the variable '__all__' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '__all__', result_iadd_134253)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from numpy.testing.nosetester import _numpy_tester' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_134254 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.testing.nosetester')

if (type(import_134254) is not StypyTypeError):

    if (import_134254 != 'pyd_module'):
        __import__(import_134254)
        sys_modules_134255 = sys.modules[import_134254]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.testing.nosetester', sys_modules_134255.module_type_store, module_type_store, ['_numpy_tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_134255, sys_modules_134255.module_type_store, module_type_store)
    else:
        from numpy.testing.nosetester import _numpy_tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.testing.nosetester', None, module_type_store, ['_numpy_tester'], [_numpy_tester])

else:
    # Assigning a type to the variable 'numpy.testing.nosetester' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.testing.nosetester', import_134254)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a Attribute to a Name (line 45):

# Call to _numpy_tester(...): (line 45)
# Processing the call keyword arguments (line 45)
kwargs_134257 = {}
# Getting the type of '_numpy_tester' (line 45)
_numpy_tester_134256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 45)
_numpy_tester_call_result_134258 = invoke(stypy.reporting.localization.Localization(__file__, 45, 7), _numpy_tester_134256, *[], **kwargs_134257)

# Obtaining the member 'test' of a type (line 45)
test_134259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 7), _numpy_tester_call_result_134258, 'test')
# Assigning a type to the variable 'test' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'test', test_134259)

# Assigning a Attribute to a Name (line 46):

# Call to _numpy_tester(...): (line 46)
# Processing the call keyword arguments (line 46)
kwargs_134261 = {}
# Getting the type of '_numpy_tester' (line 46)
_numpy_tester_134260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), '_numpy_tester', False)
# Calling _numpy_tester(args, kwargs) (line 46)
_numpy_tester_call_result_134262 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), _numpy_tester_134260, *[], **kwargs_134261)

# Obtaining the member 'bench' of a type (line 46)
bench_134263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), _numpy_tester_call_result_134262, 'bench')
# Assigning a type to the variable 'bench' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'bench', bench_134263)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
