
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ==========================================
3: Miscellaneous routines (:mod:`scipy.misc`)
4: ==========================================
5: 
6: .. currentmodule:: scipy.misc
7: 
8: Various utilities that don't have another home.
9: 
10: Note that Pillow (https://python-pillow.org/) is not a dependency
11: of SciPy, but the image manipulation functions indicated in the list
12: below are not available without it.
13: 
14: .. autosummary::
15:    :toctree: generated/
16: 
17:    ascent - Get example image for processing
18:    central_diff_weights - Weights for an n-point central m-th derivative
19:    derivative - Find the n-th derivative of a function at a point
20:    face - Get example image for processing
21: 
22: Deprecated functions:
23: 
24: .. autosummary::
25:    :toctree: generated/
26: 
27:    bytescale - Byte scales an array (image) [requires Pillow]
28:    fromimage - Return a copy of a PIL image as a numpy array [requires Pillow]
29:    imfilter - Simple filtering of an image [requires Pillow]
30:    imread - Read an image file from a filename [requires Pillow]
31:    imresize - Resize an image [requires Pillow]
32:    imrotate - Rotate an image counter-clockwise [requires Pillow]
33:    imsave - Save an array to an image file [requires Pillow]
34:    imshow - Simple showing of an image through an external viewer [requires Pillow]
35:    toimage - Takes a numpy array and returns a PIL image [requires Pillow]
36: 
37: 
38: Deprecated aliases:
39: 
40: .. autosummary::
41:    :toctree: generated/
42: 
43:    comb - Combinations of N things taken k at a time, "N choose k" (imported from `scipy.special`)
44:    factorial  - The factorial function, ``n! = special.gamma(n+1)``
45:                 (imported from `scipy.special`)
46:    factorial2 - Double factorial, ``(n!)!`` (imported from `scipy.special`)
47:    factorialk - ``(...((n!)!)!...)!`` where there are k '!' (imported from `scipy.special`)
48:    logsumexp - Compute the log of the sum of exponentials of input elements
49:                (imported from `scipy.special`)
50:    pade - Pade approximation to function as the ratio of two polynomials.
51:           (imported from `scipy.interpolate`)
52:    info - Get help information for a function, class, or module. (imported from `numpy`)
53:    source - Print function source code. (imported from `numpy`)
54:    who - Print the Numpy arrays in the given dictionary. (imported from `numpy`)
55: 
56: '''
57: 
58: from __future__ import division, print_function, absolute_import
59: 
60: __all__ = ['who', 'source', 'info', 'doccer', 'pade',
61:            'comb', 'factorial', 'factorial2', 'factorialk', 'logsumexp']
62: 
63: from . import doccer
64: from .common import *
65: from numpy import who as _who, source as _source, info as _info
66: import numpy as np
67: from scipy.interpolate._pade import pade as _pade
68: from scipy.special import (comb as _comb, logsumexp as _lsm,
69:         factorial as _fact, factorial2 as _fact2, factorialk as _factk)
70: 
71: import sys
72: 
73: _msg = ("Importing `%(name)s` from scipy.misc is deprecated in scipy 1.0.0. Use "
74:         "`scipy.special.%(name)s` instead.")
75: comb = np.deprecate(_comb, message=_msg % {"name": _comb.__name__})
76: logsumexp = np.deprecate(_lsm, message=_msg % {"name": _lsm.__name__})
77: factorial = np.deprecate(_fact, message=_msg % {"name": _fact.__name__})
78: factorial2 = np.deprecate(_fact2, message=_msg % {"name": _fact2.__name__})
79: factorialk = np.deprecate(_factk, message=_msg % {"name": _factk.__name__})
80: 
81: _msg = ("Importing `pade` from scipy.misc is deprecated in scipy 1.0.0. Use "
82:         "`scipy.interpolate.pade` instead.")
83: pade = np.deprecate(_pade, message=_msg)
84: 
85: _msg = ("Importing `%(name)s` from scipy.misc is deprecated in scipy 1.0.0. Use "
86:         "`numpy.%(name)s` instead.")
87: who = np.deprecate(_who, message=_msg % {"name": "who"})
88: source = np.deprecate(_source, message=_msg % {"name": "source"})
89: 
90: @np.deprecate(message=_msg % {"name": "info.(..., toplevel='scipy')"})
91: def info(object=None,maxwidth=76,output=sys.stdout,toplevel='scipy'):
92:     return _info(object, maxwidth, output, toplevel)
93: info.__doc__ = _info.__doc__
94: del sys
95: 
96: try:
97:     from .pilutil import *
98:     from . import pilutil
99:     __all__ += pilutil.__all__
100:     del pilutil
101: except ImportError:
102:     pass
103: 
104: from . import common
105: __all__ += common.__all__
106: del common
107: 
108: from scipy._lib._testutils import PytestTester
109: test = PytestTester(__name__)
110: del PytestTester
111: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_115224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n==========================================\nMiscellaneous routines (:mod:`scipy.misc`)\n==========================================\n\n.. currentmodule:: scipy.misc\n\nVarious utilities that don\'t have another home.\n\nNote that Pillow (https://python-pillow.org/) is not a dependency\nof SciPy, but the image manipulation functions indicated in the list\nbelow are not available without it.\n\n.. autosummary::\n   :toctree: generated/\n\n   ascent - Get example image for processing\n   central_diff_weights - Weights for an n-point central m-th derivative\n   derivative - Find the n-th derivative of a function at a point\n   face - Get example image for processing\n\nDeprecated functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   bytescale - Byte scales an array (image) [requires Pillow]\n   fromimage - Return a copy of a PIL image as a numpy array [requires Pillow]\n   imfilter - Simple filtering of an image [requires Pillow]\n   imread - Read an image file from a filename [requires Pillow]\n   imresize - Resize an image [requires Pillow]\n   imrotate - Rotate an image counter-clockwise [requires Pillow]\n   imsave - Save an array to an image file [requires Pillow]\n   imshow - Simple showing of an image through an external viewer [requires Pillow]\n   toimage - Takes a numpy array and returns a PIL image [requires Pillow]\n\n\nDeprecated aliases:\n\n.. autosummary::\n   :toctree: generated/\n\n   comb - Combinations of N things taken k at a time, "N choose k" (imported from `scipy.special`)\n   factorial  - The factorial function, ``n! = special.gamma(n+1)``\n                (imported from `scipy.special`)\n   factorial2 - Double factorial, ``(n!)!`` (imported from `scipy.special`)\n   factorialk - ``(...((n!)!)!...)!`` where there are k \'!\' (imported from `scipy.special`)\n   logsumexp - Compute the log of the sum of exponentials of input elements\n               (imported from `scipy.special`)\n   pade - Pade approximation to function as the ratio of two polynomials.\n          (imported from `scipy.interpolate`)\n   info - Get help information for a function, class, or module. (imported from `numpy`)\n   source - Print function source code. (imported from `numpy`)\n   who - Print the Numpy arrays in the given dictionary. (imported from `numpy`)\n\n')

# Assigning a List to a Name (line 60):
__all__ = ['who', 'source', 'info', 'doccer', 'pade', 'comb', 'factorial', 'factorial2', 'factorialk', 'logsumexp']
module_type_store.set_exportable_members(['who', 'source', 'info', 'doccer', 'pade', 'comb', 'factorial', 'factorial2', 'factorialk', 'logsumexp'])

# Obtaining an instance of the builtin type 'list' (line 60)
list_115225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
str_115226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'str', 'who')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115226)
# Adding element type (line 60)
str_115227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'str', 'source')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115227)
# Adding element type (line 60)
str_115228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'str', 'info')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115228)
# Adding element type (line 60)
str_115229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'str', 'doccer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115229)
# Adding element type (line 60)
str_115230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'str', 'pade')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115230)
# Adding element type (line 60)
str_115231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 11), 'str', 'comb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115231)
# Adding element type (line 60)
str_115232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'factorial')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115232)
# Adding element type (line 60)
str_115233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'str', 'factorial2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115233)
# Adding element type (line 60)
str_115234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 46), 'str', 'factorialk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115234)
# Adding element type (line 60)
str_115235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 60), 'str', 'logsumexp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 10), list_115225, str_115235)

# Assigning a type to the variable '__all__' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), '__all__', list_115225)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 0))

# 'from scipy.misc import doccer' statement (line 63)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115236 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'scipy.misc')

if (type(import_115236) is not StypyTypeError):

    if (import_115236 != 'pyd_module'):
        __import__(import_115236)
        sys_modules_115237 = sys.modules[import_115236]
        import_from_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'scipy.misc', sys_modules_115237.module_type_store, module_type_store, ['doccer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 63, 0), __file__, sys_modules_115237, sys_modules_115237.module_type_store, module_type_store)
    else:
        from scipy.misc import doccer

        import_from_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'scipy.misc', None, module_type_store, ['doccer'], [doccer])

else:
    # Assigning a type to the variable 'scipy.misc' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'scipy.misc', import_115236)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 0))

# 'from scipy.misc.common import ' statement (line 64)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115238 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'scipy.misc.common')

if (type(import_115238) is not StypyTypeError):

    if (import_115238 != 'pyd_module'):
        __import__(import_115238)
        sys_modules_115239 = sys.modules[import_115238]
        import_from_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'scipy.misc.common', sys_modules_115239.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 64, 0), __file__, sys_modules_115239, sys_modules_115239.module_type_store, module_type_store)
    else:
        from scipy.misc.common import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'scipy.misc.common', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.misc.common' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'scipy.misc.common', import_115238)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 65, 0))

# 'from numpy import _who, _source, _info' statement (line 65)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115240 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy')

if (type(import_115240) is not StypyTypeError):

    if (import_115240 != 'pyd_module'):
        __import__(import_115240)
        sys_modules_115241 = sys.modules[import_115240]
        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', sys_modules_115241.module_type_store, module_type_store, ['who', 'source', 'info'])
        nest_module(stypy.reporting.localization.Localization(__file__, 65, 0), __file__, sys_modules_115241, sys_modules_115241.module_type_store, module_type_store)
    else:
        from numpy import who as _who, source as _source, info as _info

        import_from_module(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', None, module_type_store, ['who', 'source', 'info'], [_who, _source, _info])

else:
    # Assigning a type to the variable 'numpy' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'numpy', import_115240)

# Adding an alias
module_type_store.add_alias('_info', 'info')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'import numpy' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115242 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy')

if (type(import_115242) is not StypyTypeError):

    if (import_115242 != 'pyd_module'):
        __import__(import_115242)
        sys_modules_115243 = sys.modules[import_115242]
        import_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'np', sys_modules_115243.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy', import_115242)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 67, 0))

# 'from scipy.interpolate._pade import _pade' statement (line 67)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115244 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'scipy.interpolate._pade')

if (type(import_115244) is not StypyTypeError):

    if (import_115244 != 'pyd_module'):
        __import__(import_115244)
        sys_modules_115245 = sys.modules[import_115244]
        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'scipy.interpolate._pade', sys_modules_115245.module_type_store, module_type_store, ['pade'])
        nest_module(stypy.reporting.localization.Localization(__file__, 67, 0), __file__, sys_modules_115245, sys_modules_115245.module_type_store, module_type_store)
    else:
        from scipy.interpolate._pade import pade as _pade

        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'scipy.interpolate._pade', None, module_type_store, ['pade'], [_pade])

else:
    # Assigning a type to the variable 'scipy.interpolate._pade' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'scipy.interpolate._pade', import_115244)

# Adding an alias
module_type_store.add_alias('_pade', 'pade')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 68, 0))

# 'from scipy.special import _comb, _lsm, _fact, _fact2, _factk' statement (line 68)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115246 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'scipy.special')

if (type(import_115246) is not StypyTypeError):

    if (import_115246 != 'pyd_module'):
        __import__(import_115246)
        sys_modules_115247 = sys.modules[import_115246]
        import_from_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'scipy.special', sys_modules_115247.module_type_store, module_type_store, ['comb', 'logsumexp', 'factorial', 'factorial2', 'factorialk'])
        nest_module(stypy.reporting.localization.Localization(__file__, 68, 0), __file__, sys_modules_115247, sys_modules_115247.module_type_store, module_type_store)
    else:
        from scipy.special import comb as _comb, logsumexp as _lsm, factorial as _fact, factorial2 as _fact2, factorialk as _factk

        import_from_module(stypy.reporting.localization.Localization(__file__, 68, 0), 'scipy.special', None, module_type_store, ['comb', 'logsumexp', 'factorial', 'factorial2', 'factorialk'], [_comb, _lsm, _fact, _fact2, _factk])

else:
    # Assigning a type to the variable 'scipy.special' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'scipy.special', import_115246)

# Adding an alias
module_type_store.add_alias('_factk', 'factorialk')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 71, 0))

# 'import sys' statement (line 71)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'sys', sys, module_type_store)


# Assigning a Str to a Name (line 73):
str_115248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'str', 'Importing `%(name)s` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.special.%(name)s` instead.')
# Assigning a type to the variable '_msg' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_msg', str_115248)

# Assigning a Call to a Name (line 75):

# Call to deprecate(...): (line 75)
# Processing the call arguments (line 75)
# Getting the type of '_comb' (line 75)
_comb_115251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), '_comb', False)
# Processing the call keyword arguments (line 75)
# Getting the type of '_msg' (line 75)
_msg_115252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 75)
dict_115253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 75)
# Adding element type (key, value) (line 75)
str_115254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 43), 'str', 'name')
# Getting the type of '_comb' (line 75)
_comb_115255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 51), '_comb', False)
# Obtaining the member '__name__' of a type (line 75)
name___115256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 51), _comb_115255, '__name__')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 42), dict_115253, (str_115254, name___115256))

# Applying the binary operator '%' (line 75)
result_mod_115257 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 35), '%', _msg_115252, dict_115253)

keyword_115258 = result_mod_115257
kwargs_115259 = {'message': keyword_115258}
# Getting the type of 'np' (line 75)
np_115249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'np', False)
# Obtaining the member 'deprecate' of a type (line 75)
deprecate_115250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 7), np_115249, 'deprecate')
# Calling deprecate(args, kwargs) (line 75)
deprecate_call_result_115260 = invoke(stypy.reporting.localization.Localization(__file__, 75, 7), deprecate_115250, *[_comb_115251], **kwargs_115259)

# Assigning a type to the variable 'comb' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'comb', deprecate_call_result_115260)

# Assigning a Call to a Name (line 76):

# Call to deprecate(...): (line 76)
# Processing the call arguments (line 76)
# Getting the type of '_lsm' (line 76)
_lsm_115263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), '_lsm', False)
# Processing the call keyword arguments (line 76)
# Getting the type of '_msg' (line 76)
_msg_115264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 39), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 76)
dict_115265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 46), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 76)
# Adding element type (key, value) (line 76)
str_115266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 47), 'str', 'name')
# Getting the type of '_lsm' (line 76)
_lsm_115267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 55), '_lsm', False)
# Obtaining the member '__name__' of a type (line 76)
name___115268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 55), _lsm_115267, '__name__')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 46), dict_115265, (str_115266, name___115268))

# Applying the binary operator '%' (line 76)
result_mod_115269 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 39), '%', _msg_115264, dict_115265)

keyword_115270 = result_mod_115269
kwargs_115271 = {'message': keyword_115270}
# Getting the type of 'np' (line 76)
np_115261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'np', False)
# Obtaining the member 'deprecate' of a type (line 76)
deprecate_115262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), np_115261, 'deprecate')
# Calling deprecate(args, kwargs) (line 76)
deprecate_call_result_115272 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), deprecate_115262, *[_lsm_115263], **kwargs_115271)

# Assigning a type to the variable 'logsumexp' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'logsumexp', deprecate_call_result_115272)

# Assigning a Call to a Name (line 77):

# Call to deprecate(...): (line 77)
# Processing the call arguments (line 77)
# Getting the type of '_fact' (line 77)
_fact_115275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), '_fact', False)
# Processing the call keyword arguments (line 77)
# Getting the type of '_msg' (line 77)
_msg_115276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 40), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 77)
dict_115277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 47), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 77)
# Adding element type (key, value) (line 77)
str_115278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 48), 'str', 'name')
# Getting the type of '_fact' (line 77)
_fact_115279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 56), '_fact', False)
# Obtaining the member '__name__' of a type (line 77)
name___115280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 56), _fact_115279, '__name__')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 47), dict_115277, (str_115278, name___115280))

# Applying the binary operator '%' (line 77)
result_mod_115281 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 40), '%', _msg_115276, dict_115277)

keyword_115282 = result_mod_115281
kwargs_115283 = {'message': keyword_115282}
# Getting the type of 'np' (line 77)
np_115273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'np', False)
# Obtaining the member 'deprecate' of a type (line 77)
deprecate_115274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), np_115273, 'deprecate')
# Calling deprecate(args, kwargs) (line 77)
deprecate_call_result_115284 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), deprecate_115274, *[_fact_115275], **kwargs_115283)

# Assigning a type to the variable 'factorial' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'factorial', deprecate_call_result_115284)

# Assigning a Call to a Name (line 78):

# Call to deprecate(...): (line 78)
# Processing the call arguments (line 78)
# Getting the type of '_fact2' (line 78)
_fact2_115287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), '_fact2', False)
# Processing the call keyword arguments (line 78)
# Getting the type of '_msg' (line 78)
_msg_115288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 78)
dict_115289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 49), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 78)
# Adding element type (key, value) (line 78)
str_115290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'str', 'name')
# Getting the type of '_fact2' (line 78)
_fact2_115291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 58), '_fact2', False)
# Obtaining the member '__name__' of a type (line 78)
name___115292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 58), _fact2_115291, '__name__')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 49), dict_115289, (str_115290, name___115292))

# Applying the binary operator '%' (line 78)
result_mod_115293 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 42), '%', _msg_115288, dict_115289)

keyword_115294 = result_mod_115293
kwargs_115295 = {'message': keyword_115294}
# Getting the type of 'np' (line 78)
np_115285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 13), 'np', False)
# Obtaining the member 'deprecate' of a type (line 78)
deprecate_115286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 13), np_115285, 'deprecate')
# Calling deprecate(args, kwargs) (line 78)
deprecate_call_result_115296 = invoke(stypy.reporting.localization.Localization(__file__, 78, 13), deprecate_115286, *[_fact2_115287], **kwargs_115295)

# Assigning a type to the variable 'factorial2' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'factorial2', deprecate_call_result_115296)

# Assigning a Call to a Name (line 79):

# Call to deprecate(...): (line 79)
# Processing the call arguments (line 79)
# Getting the type of '_factk' (line 79)
_factk_115299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), '_factk', False)
# Processing the call keyword arguments (line 79)
# Getting the type of '_msg' (line 79)
_msg_115300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 79)
dict_115301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 49), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 79)
# Adding element type (key, value) (line 79)
str_115302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 50), 'str', 'name')
# Getting the type of '_factk' (line 79)
_factk_115303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 58), '_factk', False)
# Obtaining the member '__name__' of a type (line 79)
name___115304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 58), _factk_115303, '__name__')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 49), dict_115301, (str_115302, name___115304))

# Applying the binary operator '%' (line 79)
result_mod_115305 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 42), '%', _msg_115300, dict_115301)

keyword_115306 = result_mod_115305
kwargs_115307 = {'message': keyword_115306}
# Getting the type of 'np' (line 79)
np_115297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'np', False)
# Obtaining the member 'deprecate' of a type (line 79)
deprecate_115298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), np_115297, 'deprecate')
# Calling deprecate(args, kwargs) (line 79)
deprecate_call_result_115308 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), deprecate_115298, *[_factk_115299], **kwargs_115307)

# Assigning a type to the variable 'factorialk' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'factorialk', deprecate_call_result_115308)

# Assigning a Str to a Name (line 81):
str_115309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'str', 'Importing `pade` from scipy.misc is deprecated in scipy 1.0.0. Use `scipy.interpolate.pade` instead.')
# Assigning a type to the variable '_msg' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), '_msg', str_115309)

# Assigning a Call to a Name (line 83):

# Call to deprecate(...): (line 83)
# Processing the call arguments (line 83)
# Getting the type of '_pade' (line 83)
_pade_115312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), '_pade', False)
# Processing the call keyword arguments (line 83)
# Getting the type of '_msg' (line 83)
_msg_115313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 35), '_msg', False)
keyword_115314 = _msg_115313
kwargs_115315 = {'message': keyword_115314}
# Getting the type of 'np' (line 83)
np_115310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'np', False)
# Obtaining the member 'deprecate' of a type (line 83)
deprecate_115311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 7), np_115310, 'deprecate')
# Calling deprecate(args, kwargs) (line 83)
deprecate_call_result_115316 = invoke(stypy.reporting.localization.Localization(__file__, 83, 7), deprecate_115311, *[_pade_115312], **kwargs_115315)

# Assigning a type to the variable 'pade' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'pade', deprecate_call_result_115316)

# Assigning a Str to a Name (line 85):
str_115317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'str', 'Importing `%(name)s` from scipy.misc is deprecated in scipy 1.0.0. Use `numpy.%(name)s` instead.')
# Assigning a type to the variable '_msg' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '_msg', str_115317)

# Assigning a Call to a Name (line 87):

# Call to deprecate(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of '_who' (line 87)
_who_115320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), '_who', False)
# Processing the call keyword arguments (line 87)
# Getting the type of '_msg' (line 87)
_msg_115321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 87)
dict_115322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 40), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 87)
# Adding element type (key, value) (line 87)
str_115323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 41), 'str', 'name')
str_115324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 49), 'str', 'who')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 40), dict_115322, (str_115323, str_115324))

# Applying the binary operator '%' (line 87)
result_mod_115325 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 33), '%', _msg_115321, dict_115322)

keyword_115326 = result_mod_115325
kwargs_115327 = {'message': keyword_115326}
# Getting the type of 'np' (line 87)
np_115318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 6), 'np', False)
# Obtaining the member 'deprecate' of a type (line 87)
deprecate_115319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 6), np_115318, 'deprecate')
# Calling deprecate(args, kwargs) (line 87)
deprecate_call_result_115328 = invoke(stypy.reporting.localization.Localization(__file__, 87, 6), deprecate_115319, *[_who_115320], **kwargs_115327)

# Assigning a type to the variable 'who' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'who', deprecate_call_result_115328)

# Assigning a Call to a Name (line 88):

# Call to deprecate(...): (line 88)
# Processing the call arguments (line 88)
# Getting the type of '_source' (line 88)
_source_115331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), '_source', False)
# Processing the call keyword arguments (line 88)
# Getting the type of '_msg' (line 88)
_msg_115332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 39), '_msg', False)

# Obtaining an instance of the builtin type 'dict' (line 88)
dict_115333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 46), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 88)
# Adding element type (key, value) (line 88)
str_115334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 47), 'str', 'name')
str_115335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 55), 'str', 'source')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 46), dict_115333, (str_115334, str_115335))

# Applying the binary operator '%' (line 88)
result_mod_115336 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 39), '%', _msg_115332, dict_115333)

keyword_115337 = result_mod_115336
kwargs_115338 = {'message': keyword_115337}
# Getting the type of 'np' (line 88)
np_115329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 9), 'np', False)
# Obtaining the member 'deprecate' of a type (line 88)
deprecate_115330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 9), np_115329, 'deprecate')
# Calling deprecate(args, kwargs) (line 88)
deprecate_call_result_115339 = invoke(stypy.reporting.localization.Localization(__file__, 88, 9), deprecate_115330, *[_source_115331], **kwargs_115338)

# Assigning a type to the variable 'source' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'source', deprecate_call_result_115339)

@norecursion
def info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 91)
    None_115340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'None')
    int_115341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'int')
    # Getting the type of 'sys' (line 91)
    sys_115342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'sys')
    # Obtaining the member 'stdout' of a type (line 91)
    stdout_115343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 40), sys_115342, 'stdout')
    str_115344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 60), 'str', 'scipy')
    defaults = [None_115340, int_115341, stdout_115343, str_115344]
    # Create a new context for function 'info'
    module_type_store = module_type_store.open_function_context('info', 90, 0, False)
    
    # Passed parameters checking function
    info.stypy_localization = localization
    info.stypy_type_of_self = None
    info.stypy_type_store = module_type_store
    info.stypy_function_name = 'info'
    info.stypy_param_names_list = ['object', 'maxwidth', 'output', 'toplevel']
    info.stypy_varargs_param_name = None
    info.stypy_kwargs_param_name = None
    info.stypy_call_defaults = defaults
    info.stypy_call_varargs = varargs
    info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'info', ['object', 'maxwidth', 'output', 'toplevel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'info', localization, ['object', 'maxwidth', 'output', 'toplevel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'info(...)' code ##################

    
    # Call to _info(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'object' (line 92)
    object_115346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'object', False)
    # Getting the type of 'maxwidth' (line 92)
    maxwidth_115347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'maxwidth', False)
    # Getting the type of 'output' (line 92)
    output_115348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'output', False)
    # Getting the type of 'toplevel' (line 92)
    toplevel_115349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'toplevel', False)
    # Processing the call keyword arguments (line 92)
    kwargs_115350 = {}
    # Getting the type of '_info' (line 92)
    _info_115345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), '_info', False)
    # Calling _info(args, kwargs) (line 92)
    _info_call_result_115351 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), _info_115345, *[object_115346, maxwidth_115347, output_115348, toplevel_115349], **kwargs_115350)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', _info_call_result_115351)
    
    # ################# End of 'info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'info' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_115352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'info'
    return stypy_return_type_115352

# Assigning a type to the variable 'info' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'info', info)

# Assigning a Attribute to a Attribute (line 93):
# Getting the type of '_info' (line 93)
_info_115353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), '_info')
# Obtaining the member '__doc__' of a type (line 93)
doc___115354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), _info_115353, '__doc__')
# Getting the type of 'info' (line 93)
info_115355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'info')
# Setting the type of the member '__doc__' of a type (line 93)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 0), info_115355, '__doc__', doc___115354)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 94, 0), module_type_store, 'sys')


# SSA begins for try-except statement (line 96)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 97, 4))

# 'from scipy.misc.pilutil import ' statement (line 97)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 97, 4), 'scipy.misc.pilutil')

if (type(import_115356) is not StypyTypeError):

    if (import_115356 != 'pyd_module'):
        __import__(import_115356)
        sys_modules_115357 = sys.modules[import_115356]
        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 4), 'scipy.misc.pilutil', sys_modules_115357.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 97, 4), __file__, sys_modules_115357, sys_modules_115357.module_type_store, module_type_store)
    else:
        from scipy.misc.pilutil import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 97, 4), 'scipy.misc.pilutil', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.misc.pilutil' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'scipy.misc.pilutil', import_115356)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 98, 4))

# 'from scipy.misc import pilutil' statement (line 98)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115358 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 98, 4), 'scipy.misc')

if (type(import_115358) is not StypyTypeError):

    if (import_115358 != 'pyd_module'):
        __import__(import_115358)
        sys_modules_115359 = sys.modules[import_115358]
        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 4), 'scipy.misc', sys_modules_115359.module_type_store, module_type_store, ['pilutil'])
        nest_module(stypy.reporting.localization.Localization(__file__, 98, 4), __file__, sys_modules_115359, sys_modules_115359.module_type_store, module_type_store)
    else:
        from scipy.misc import pilutil

        import_from_module(stypy.reporting.localization.Localization(__file__, 98, 4), 'scipy.misc', None, module_type_store, ['pilutil'], [pilutil])

else:
    # Assigning a type to the variable 'scipy.misc' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'scipy.misc', import_115358)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')


# Getting the type of '__all__' (line 99)
all___115360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), '__all__')
# Getting the type of 'pilutil' (line 99)
pilutil_115361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'pilutil')
# Obtaining the member '__all__' of a type (line 99)
all___115362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 15), pilutil_115361, '__all__')
# Applying the binary operator '+=' (line 99)
result_iadd_115363 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 4), '+=', all___115360, all___115362)
# Assigning a type to the variable '__all__' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), '__all__', result_iadd_115363)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 100, 4), module_type_store, 'pilutil')
# SSA branch for the except part of a try statement (line 96)
# SSA branch for the except 'ImportError' branch of a try statement (line 96)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 96)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 104, 0))

# 'from scipy.misc import common' statement (line 104)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 104, 0), 'scipy.misc')

if (type(import_115364) is not StypyTypeError):

    if (import_115364 != 'pyd_module'):
        __import__(import_115364)
        sys_modules_115365 = sys.modules[import_115364]
        import_from_module(stypy.reporting.localization.Localization(__file__, 104, 0), 'scipy.misc', sys_modules_115365.module_type_store, module_type_store, ['common'])
        nest_module(stypy.reporting.localization.Localization(__file__, 104, 0), __file__, sys_modules_115365, sys_modules_115365.module_type_store, module_type_store)
    else:
        from scipy.misc import common

        import_from_module(stypy.reporting.localization.Localization(__file__, 104, 0), 'scipy.misc', None, module_type_store, ['common'], [common])

else:
    # Assigning a type to the variable 'scipy.misc' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'scipy.misc', import_115364)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')


# Getting the type of '__all__' (line 105)
all___115366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '__all__')
# Getting the type of 'common' (line 105)
common_115367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'common')
# Obtaining the member '__all__' of a type (line 105)
all___115368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), common_115367, '__all__')
# Applying the binary operator '+=' (line 105)
result_iadd_115369 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 0), '+=', all___115366, all___115368)
# Assigning a type to the variable '__all__' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '__all__', result_iadd_115369)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 106, 0), module_type_store, 'common')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 108, 0))

# 'from scipy._lib._testutils import PytestTester' statement (line 108)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
import_115370 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy._lib._testutils')

if (type(import_115370) is not StypyTypeError):

    if (import_115370 != 'pyd_module'):
        __import__(import_115370)
        sys_modules_115371 = sys.modules[import_115370]
        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy._lib._testutils', sys_modules_115371.module_type_store, module_type_store, ['PytestTester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 108, 0), __file__, sys_modules_115371, sys_modules_115371.module_type_store, module_type_store)
    else:
        from scipy._lib._testutils import PytestTester

        import_from_module(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy._lib._testutils', None, module_type_store, ['PytestTester'], [PytestTester])

else:
    # Assigning a type to the variable 'scipy._lib._testutils' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'scipy._lib._testutils', import_115370)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')


# Assigning a Call to a Name (line 109):

# Call to PytestTester(...): (line 109)
# Processing the call arguments (line 109)
# Getting the type of '__name__' (line 109)
name___115373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), '__name__', False)
# Processing the call keyword arguments (line 109)
kwargs_115374 = {}
# Getting the type of 'PytestTester' (line 109)
PytestTester_115372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'PytestTester', False)
# Calling PytestTester(args, kwargs) (line 109)
PytestTester_call_result_115375 = invoke(stypy.reporting.localization.Localization(__file__, 109, 7), PytestTester_115372, *[name___115373], **kwargs_115374)

# Assigning a type to the variable 'test' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'test', PytestTester_call_result_115375)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 110, 0), module_type_store, 'PytestTester')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
