
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Statistics-related constants.
3: 
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import numpy as np
8: 
9: 
10: # The smallest representable positive number such that 1.0 + _EPS != 1.0.
11: _EPS = np.finfo(float).eps
12: 
13: # The largest [in magnitude] usable floating value.
14: _XMAX = np.finfo(float).max
15: 
16: # The log of the largest usable floating value; useful for knowing
17: # when exp(something) will overflow
18: _LOGXMAX = np.log(_XMAX)
19: 
20: # The smallest [in magnitude] usable floating value.
21: _XMIN = np.finfo(float).tiny
22: 
23: # -special.psi(1)
24: _EULER = 0.577215664901532860606512090082402431042
25: 
26: # special.zeta(3, 1)  Apery's constant
27: _ZETA3 = 1.202056903159594285399738161511449990765
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_590327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nStatistics-related constants.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_590328 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_590328) is not StypyTypeError):

    if (import_590328 != 'pyd_module'):
        __import__(import_590328)
        sys_modules_590329 = sys.modules[import_590328]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_590329.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_590328)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a Attribute to a Name (line 11):

# Call to finfo(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'float' (line 11)
float_590332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'float', False)
# Processing the call keyword arguments (line 11)
kwargs_590333 = {}
# Getting the type of 'np' (line 11)
np_590330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'np', False)
# Obtaining the member 'finfo' of a type (line 11)
finfo_590331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), np_590330, 'finfo')
# Calling finfo(args, kwargs) (line 11)
finfo_call_result_590334 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), finfo_590331, *[float_590332], **kwargs_590333)

# Obtaining the member 'eps' of a type (line 11)
eps_590335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), finfo_call_result_590334, 'eps')
# Assigning a type to the variable '_EPS' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '_EPS', eps_590335)

# Assigning a Attribute to a Name (line 14):

# Call to finfo(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'float' (line 14)
float_590338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'float', False)
# Processing the call keyword arguments (line 14)
kwargs_590339 = {}
# Getting the type of 'np' (line 14)
np_590336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'np', False)
# Obtaining the member 'finfo' of a type (line 14)
finfo_590337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), np_590336, 'finfo')
# Calling finfo(args, kwargs) (line 14)
finfo_call_result_590340 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), finfo_590337, *[float_590338], **kwargs_590339)

# Obtaining the member 'max' of a type (line 14)
max_590341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), finfo_call_result_590340, 'max')
# Assigning a type to the variable '_XMAX' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_XMAX', max_590341)

# Assigning a Call to a Name (line 18):

# Call to log(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of '_XMAX' (line 18)
_XMAX_590344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), '_XMAX', False)
# Processing the call keyword arguments (line 18)
kwargs_590345 = {}
# Getting the type of 'np' (line 18)
np_590342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'np', False)
# Obtaining the member 'log' of a type (line 18)
log_590343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), np_590342, 'log')
# Calling log(args, kwargs) (line 18)
log_call_result_590346 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), log_590343, *[_XMAX_590344], **kwargs_590345)

# Assigning a type to the variable '_LOGXMAX' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_LOGXMAX', log_call_result_590346)

# Assigning a Attribute to a Name (line 21):

# Call to finfo(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'float' (line 21)
float_590349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'float', False)
# Processing the call keyword arguments (line 21)
kwargs_590350 = {}
# Getting the type of 'np' (line 21)
np_590347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'np', False)
# Obtaining the member 'finfo' of a type (line 21)
finfo_590348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), np_590347, 'finfo')
# Calling finfo(args, kwargs) (line 21)
finfo_call_result_590351 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), finfo_590348, *[float_590349], **kwargs_590350)

# Obtaining the member 'tiny' of a type (line 21)
tiny_590352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), finfo_call_result_590351, 'tiny')
# Assigning a type to the variable '_XMIN' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '_XMIN', tiny_590352)

# Assigning a Num to a Name (line 24):
float_590353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'float')
# Assigning a type to the variable '_EULER' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_EULER', float_590353)

# Assigning a Num to a Name (line 27):
float_590354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'float')
# Assigning a type to the variable '_ZETA3' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '_ZETA3', float_590354)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
