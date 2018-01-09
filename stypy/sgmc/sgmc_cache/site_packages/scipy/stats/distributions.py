
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author:  Travis Oliphant  2002-2011 with contributions from
3: #          SciPy Developers 2004-2011
4: #
5: # NOTE: To look at history using `git blame`, use `git blame -M -C -C`
6: #       instead of `git blame -Lxxx,+x`.
7: #
8: from __future__ import division, print_function, absolute_import
9: 
10: from ._distn_infrastructure import (entropy, rv_discrete, rv_continuous,
11:                                     rv_frozen)
12: 
13: from . import _continuous_distns
14: from . import _discrete_distns
15: 
16: from ._continuous_distns import *
17: from ._discrete_distns import *
18: 
19: # For backwards compatibility e.g. pymc expects distributions.__all__.
20: __all__ = ['entropy', 'rv_discrete', 'rv_continuous', 'rv_histogram']
21: 
22: # Add only the distribution names, not the *_gen names.
23: __all__ += _continuous_distns._distn_names
24: __all__ += _discrete_distns._distn_names
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.stats._distn_infrastructure import entropy, rv_discrete, rv_continuous, rv_frozen' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565051 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats._distn_infrastructure')

if (type(import_565051) is not StypyTypeError):

    if (import_565051 != 'pyd_module'):
        __import__(import_565051)
        sys_modules_565052 = sys.modules[import_565051]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats._distn_infrastructure', sys_modules_565052.module_type_store, module_type_store, ['entropy', 'rv_discrete', 'rv_continuous', 'rv_frozen'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_565052, sys_modules_565052.module_type_store, module_type_store)
    else:
        from scipy.stats._distn_infrastructure import entropy, rv_discrete, rv_continuous, rv_frozen

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats._distn_infrastructure', None, module_type_store, ['entropy', 'rv_discrete', 'rv_continuous', 'rv_frozen'], [entropy, rv_discrete, rv_continuous, rv_frozen])

else:
    # Assigning a type to the variable 'scipy.stats._distn_infrastructure' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.stats._distn_infrastructure', import_565051)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.stats import _continuous_distns' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.stats')

if (type(import_565053) is not StypyTypeError):

    if (import_565053 != 'pyd_module'):
        __import__(import_565053)
        sys_modules_565054 = sys.modules[import_565053]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.stats', sys_modules_565054.module_type_store, module_type_store, ['_continuous_distns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_565054, sys_modules_565054.module_type_store, module_type_store)
    else:
        from scipy.stats import _continuous_distns

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.stats', None, module_type_store, ['_continuous_distns'], [_continuous_distns])

else:
    # Assigning a type to the variable 'scipy.stats' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.stats', import_565053)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.stats import _discrete_distns' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.stats')

if (type(import_565055) is not StypyTypeError):

    if (import_565055 != 'pyd_module'):
        __import__(import_565055)
        sys_modules_565056 = sys.modules[import_565055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.stats', sys_modules_565056.module_type_store, module_type_store, ['_discrete_distns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_565056, sys_modules_565056.module_type_store, module_type_store)
    else:
        from scipy.stats import _discrete_distns

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.stats', None, module_type_store, ['_discrete_distns'], [_discrete_distns])

else:
    # Assigning a type to the variable 'scipy.stats' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.stats', import_565055)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.stats._continuous_distns import ' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.stats._continuous_distns')

if (type(import_565057) is not StypyTypeError):

    if (import_565057 != 'pyd_module'):
        __import__(import_565057)
        sys_modules_565058 = sys.modules[import_565057]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.stats._continuous_distns', sys_modules_565058.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_565058, sys_modules_565058.module_type_store, module_type_store)
    else:
        from scipy.stats._continuous_distns import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.stats._continuous_distns', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats._continuous_distns' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.stats._continuous_distns', import_565057)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.stats._discrete_distns import ' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_565059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.stats._discrete_distns')

if (type(import_565059) is not StypyTypeError):

    if (import_565059 != 'pyd_module'):
        __import__(import_565059)
        sys_modules_565060 = sys.modules[import_565059]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.stats._discrete_distns', sys_modules_565060.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_565060, sys_modules_565060.module_type_store, module_type_store)
    else:
        from scipy.stats._discrete_distns import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.stats._discrete_distns', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.stats._discrete_distns' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.stats._discrete_distns', import_565059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a List to a Name (line 20):
__all__ = ['entropy', 'rv_discrete', 'rv_continuous', 'rv_histogram']
module_type_store.set_exportable_members(['entropy', 'rv_discrete', 'rv_continuous', 'rv_histogram'])

# Obtaining an instance of the builtin type 'list' (line 20)
list_565061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_565062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'entropy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_565061, str_565062)
# Adding element type (line 20)
str_565063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'str', 'rv_discrete')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_565061, str_565063)
# Adding element type (line 20)
str_565064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'str', 'rv_continuous')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_565061, str_565064)
# Adding element type (line 20)
str_565065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 54), 'str', 'rv_histogram')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_565061, str_565065)

# Assigning a type to the variable '__all__' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '__all__', list_565061)

# Getting the type of '__all__' (line 23)
all___565066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '__all__')
# Getting the type of '_continuous_distns' (line 23)
_continuous_distns_565067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), '_continuous_distns')
# Obtaining the member '_distn_names' of a type (line 23)
_distn_names_565068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 11), _continuous_distns_565067, '_distn_names')
# Applying the binary operator '+=' (line 23)
result_iadd_565069 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 0), '+=', all___565066, _distn_names_565068)
# Assigning a type to the variable '__all__' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '__all__', result_iadd_565069)


# Getting the type of '__all__' (line 24)
all___565070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__all__')
# Getting the type of '_discrete_distns' (line 24)
_discrete_distns_565071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), '_discrete_distns')
# Obtaining the member '_distn_names' of a type (line 24)
_distn_names_565072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 11), _discrete_distns_565071, '_distn_names')
# Applying the binary operator '+=' (line 24)
result_iadd_565073 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 0), '+=', all___565070, _distn_names_565072)
# Assigning a type to the variable '__all__' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__all__', result_iadd_565073)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
