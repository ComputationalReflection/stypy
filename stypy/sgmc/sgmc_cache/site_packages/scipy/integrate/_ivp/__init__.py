
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Suite of ODE solvers implemented in Python.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: from .ivp import solve_ivp
5: from .rk import RK23, RK45
6: from .radau import Radau
7: from .bdf import BDF
8: from .lsoda import LSODA
9: from .common import OdeSolution
10: from .base import DenseOutput, OdeSolver
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_59061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Suite of ODE solvers implemented in Python.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.integrate._ivp.ivp import solve_ivp' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59062 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.ivp')

if (type(import_59062) is not StypyTypeError):

    if (import_59062 != 'pyd_module'):
        __import__(import_59062)
        sys_modules_59063 = sys.modules[import_59062]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.ivp', sys_modules_59063.module_type_store, module_type_store, ['solve_ivp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_59063, sys_modules_59063.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.ivp import solve_ivp

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.ivp', None, module_type_store, ['solve_ivp'], [solve_ivp])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.ivp' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.integrate._ivp.ivp', import_59062)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.integrate._ivp.rk import RK23, RK45' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.rk')

if (type(import_59064) is not StypyTypeError):

    if (import_59064 != 'pyd_module'):
        __import__(import_59064)
        sys_modules_59065 = sys.modules[import_59064]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.rk', sys_modules_59065.module_type_store, module_type_store, ['RK23', 'RK45'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_59065, sys_modules_59065.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.rk import RK23, RK45

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.rk', None, module_type_store, ['RK23', 'RK45'], [RK23, RK45])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.rk' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.integrate._ivp.rk', import_59064)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.integrate._ivp.radau import Radau' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.radau')

if (type(import_59066) is not StypyTypeError):

    if (import_59066 != 'pyd_module'):
        __import__(import_59066)
        sys_modules_59067 = sys.modules[import_59066]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.radau', sys_modules_59067.module_type_store, module_type_store, ['Radau'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_59067, sys_modules_59067.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.radau import Radau

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.radau', None, module_type_store, ['Radau'], [Radau])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.radau' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.integrate._ivp.radau', import_59066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.integrate._ivp.bdf import BDF' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.bdf')

if (type(import_59068) is not StypyTypeError):

    if (import_59068 != 'pyd_module'):
        __import__(import_59068)
        sys_modules_59069 = sys.modules[import_59068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.bdf', sys_modules_59069.module_type_store, module_type_store, ['BDF'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_59069, sys_modules_59069.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.bdf import BDF

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.bdf', None, module_type_store, ['BDF'], [BDF])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.bdf' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.integrate._ivp.bdf', import_59068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.integrate._ivp.lsoda import LSODA' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate._ivp.lsoda')

if (type(import_59070) is not StypyTypeError):

    if (import_59070 != 'pyd_module'):
        __import__(import_59070)
        sys_modules_59071 = sys.modules[import_59070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate._ivp.lsoda', sys_modules_59071.module_type_store, module_type_store, ['LSODA'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_59071, sys_modules_59071.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.lsoda import LSODA

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate._ivp.lsoda', None, module_type_store, ['LSODA'], [LSODA])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.lsoda' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.integrate._ivp.lsoda', import_59070)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.integrate._ivp.common import OdeSolution' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common')

if (type(import_59072) is not StypyTypeError):

    if (import_59072 != 'pyd_module'):
        __import__(import_59072)
        sys_modules_59073 = sys.modules[import_59072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', sys_modules_59073.module_type_store, module_type_store, ['OdeSolution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_59073, sys_modules_59073.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.common import OdeSolution

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', None, module_type_store, ['OdeSolution'], [OdeSolution])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.common' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.integrate._ivp.common', import_59072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.integrate._ivp.base import DenseOutput, OdeSolver' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')
import_59074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base')

if (type(import_59074) is not StypyTypeError):

    if (import_59074 != 'pyd_module'):
        __import__(import_59074)
        sys_modules_59075 = sys.modules[import_59074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', sys_modules_59075.module_type_store, module_type_store, ['DenseOutput', 'OdeSolver'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_59075, sys_modules_59075.module_type_store, module_type_store)
    else:
        from scipy.integrate._ivp.base import DenseOutput, OdeSolver

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', None, module_type_store, ['DenseOutput', 'OdeSolver'], [DenseOutput, OdeSolver])

else:
    # Assigning a type to the variable 'scipy.integrate._ivp.base' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.integrate._ivp.base', import_59074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/_ivp/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
