
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: n = 10
6: p = 3
7: Z = np.zeros((n,n))
8: np.put(Z, np.random.choice(range(n*n), p, replace=False), 1)
9: 
10: # l = globals().copy()
11: # for v in l:
12: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2112 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2112) is not StypyTypeError):

    if (import_2112 != 'pyd_module'):
        __import__(import_2112)
        sys_modules_2113 = sys.modules[import_2112]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2113.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2112)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Num to a Name (line 5):
int_2114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'n' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'n', int_2114)

# Assigning a Num to a Name (line 6):
int_2115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')
# Assigning a type to the variable 'p' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'p', int_2115)

# Assigning a Call to a Name (line 7):

# Call to zeros(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_2118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'n' (line 7)
n_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'n', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), tuple_2118, n_2119)
# Adding element type (line 7)
# Getting the type of 'n' (line 7)
n_2120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'n', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), tuple_2118, n_2120)

# Processing the call keyword arguments (line 7)
kwargs_2121 = {}
# Getting the type of 'np' (line 7)
np_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 7)
zeros_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_2116, 'zeros')
# Calling zeros(args, kwargs) (line 7)
zeros_call_result_2122 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), zeros_2117, *[tuple_2118], **kwargs_2121)

# Assigning a type to the variable 'Z' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'Z', zeros_call_result_2122)

# Call to put(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'Z' (line 8)
Z_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'Z', False)

# Call to choice(...): (line 8)
# Processing the call arguments (line 8)

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'n' (line 8)
n_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 33), 'n', False)
# Getting the type of 'n' (line 8)
n_2131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 35), 'n', False)
# Applying the binary operator '*' (line 8)
result_mul_2132 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 33), '*', n_2130, n_2131)

# Processing the call keyword arguments (line 8)
kwargs_2133 = {}
# Getting the type of 'range' (line 8)
range_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 27), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_2134 = invoke(stypy.reporting.localization.Localization(__file__, 8, 27), range_2129, *[result_mul_2132], **kwargs_2133)

# Getting the type of 'p' (line 8)
p_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 39), 'p', False)
# Processing the call keyword arguments (line 8)
# Getting the type of 'False' (line 8)
False_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 50), 'False', False)
keyword_2137 = False_2136
kwargs_2138 = {'replace': keyword_2137}
# Getting the type of 'np' (line 8)
np_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'np', False)
# Obtaining the member 'random' of a type (line 8)
random_2127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), np_2126, 'random')
# Obtaining the member 'choice' of a type (line 8)
choice_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), random_2127, 'choice')
# Calling choice(args, kwargs) (line 8)
choice_call_result_2139 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), choice_2128, *[range_call_result_2134, p_2135], **kwargs_2138)

int_2140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 58), 'int')
# Processing the call keyword arguments (line 8)
kwargs_2141 = {}
# Getting the type of 'np' (line 8)
np_2123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', False)
# Obtaining the member 'put' of a type (line 8)
put_2124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 0), np_2123, 'put')
# Calling put(args, kwargs) (line 8)
put_call_result_2142 = invoke(stypy.reporting.localization.Localization(__file__, 8, 0), put_2124, *[Z_2125, choice_call_result_2139, int_2140], **kwargs_2141)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
