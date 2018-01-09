
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: import sys
5: if sys.version_info[0] >= 3:
6:     from io import StringIO
7: else:
8:     from StringIO import StringIO
9: import tempfile
10: 
11: import numpy as np
12: 
13: from numpy.testing import assert_equal, \
14:     assert_array_almost_equal_nulp
15: 
16: from scipy.sparse import coo_matrix, csc_matrix, rand
17: 
18: from scipy.io import hb_read, hb_write
19: from scipy.io.harwell_boeing import HBFile, HBInfo
20: 
21: 
22: SIMPLE = '''\
23: No Title                                                                |No Key
24:              9             4             1             4
25: RUA                      100           100            10             0
26: (26I3)          (26I3)          (3E23.15)
27: 1  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
28: 3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
29: 3  3  3  3  3  3  3  4  4  4  6  6  6  6  6  6  6  6  6  6  6  8  9  9  9  9
30: 9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9 11
31: 37 71 89 18 30 45 70 19 25 52
32: 2.971243799687726e-01  3.662366682877375e-01  4.786962174699534e-01
33: 6.490068647991184e-01  6.617490424831662e-02  8.870370343191623e-01
34: 4.196478590163001e-01  5.649603072111251e-01  9.934423887087086e-01
35: 6.912334991524289e-01
36: '''
37: 
38: SIMPLE_MATRIX = coo_matrix(
39:         (
40:             (0.297124379969, 0.366236668288, 0.47869621747, 0.649006864799,
41:              0.0661749042483, 0.887037034319, 0.419647859016,
42:              0.564960307211, 0.993442388709, 0.691233499152,),
43:             (np.array([[36, 70, 88, 17, 29, 44, 69, 18, 24, 51],
44:                        [0, 4, 58, 61, 61, 72, 72, 73, 99, 99]]))))
45: 
46: 
47: def assert_csc_almost_equal(r, l):
48:     r = csc_matrix(r)
49:     l = csc_matrix(l)
50:     assert_equal(r.indptr, l.indptr)
51:     assert_equal(r.indices, l.indices)
52:     assert_array_almost_equal_nulp(r.data, l.data, 10000)
53: 
54: 
55: class TestHBReader(object):
56:     def test_simple(self):
57:         m = hb_read(StringIO(SIMPLE))
58:         assert_csc_almost_equal(m, SIMPLE_MATRIX)
59: 
60: 
61: class TestRBRoundtrip(object):
62:     def test_simple(self):
63:         rm = rand(100, 1000, 0.05).tocsc()
64:         fd, filename = tempfile.mkstemp(suffix="rb")
65:         try:
66:             hb_write(filename, rm, HBInfo.from_data(rm))
67:             m = hb_read(filename)
68:         finally:
69:             os.close(fd)
70:             os.remove(filename)
71: 
72:         assert_csc_almost_equal(m, rm)
73: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)




# Obtaining the type of the subscript
int_133196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
# Getting the type of 'sys' (line 5)
sys_133197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 5)
version_info_133198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 3), sys_133197, 'version_info')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___133199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 3), version_info_133198, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_133200 = invoke(stypy.reporting.localization.Localization(__file__, 5, 3), getitem___133199, int_133196)

int_133201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
# Applying the binary operator '>=' (line 5)
result_ge_133202 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 3), '>=', subscript_call_result_133200, int_133201)

# Testing the type of an if condition (line 5)
if_condition_133203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), result_ge_133202)
# Assigning a type to the variable 'if_condition_133203' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_133203', if_condition_133203)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from io import StringIO' statement (line 6)
try:
    from io import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'io', None, module_type_store, ['StringIO'], [StringIO])

# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from StringIO import StringIO' statement (line 8)
try:
    from StringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import tempfile' statement (line 9)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_133204 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_133204) is not StypyTypeError):

    if (import_133204 != 'pyd_module'):
        __import__(import_133204)
        sys_modules_133205 = sys.modules[import_133204]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_133205.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_133204)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.testing import assert_equal, assert_array_almost_equal_nulp' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_133206 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing')

if (type(import_133206) is not StypyTypeError):

    if (import_133206 != 'pyd_module'):
        __import__(import_133206)
        sys_modules_133207 = sys.modules[import_133206]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', sys_modules_133207.module_type_store, module_type_store, ['assert_equal', 'assert_array_almost_equal_nulp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_133207, sys_modules_133207.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_almost_equal_nulp

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_almost_equal_nulp'], [assert_equal, assert_array_almost_equal_nulp])

else:
    # Assigning a type to the variable 'numpy.testing' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.testing', import_133206)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse import coo_matrix, csc_matrix, rand' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_133208 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse')

if (type(import_133208) is not StypyTypeError):

    if (import_133208 != 'pyd_module'):
        __import__(import_133208)
        sys_modules_133209 = sys.modules[import_133208]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', sys_modules_133209.module_type_store, module_type_store, ['coo_matrix', 'csc_matrix', 'rand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_133209, sys_modules_133209.module_type_store, module_type_store)
    else:
        from scipy.sparse import coo_matrix, csc_matrix, rand

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', None, module_type_store, ['coo_matrix', 'csc_matrix', 'rand'], [coo_matrix, csc_matrix, rand])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', import_133208)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.io import hb_read, hb_write' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_133210 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io')

if (type(import_133210) is not StypyTypeError):

    if (import_133210 != 'pyd_module'):
        __import__(import_133210)
        sys_modules_133211 = sys.modules[import_133210]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io', sys_modules_133211.module_type_store, module_type_store, ['hb_read', 'hb_write'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_133211, sys_modules_133211.module_type_store, module_type_store)
    else:
        from scipy.io import hb_read, hb_write

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io', None, module_type_store, ['hb_read', 'hb_write'], [hb_read, hb_write])

else:
    # Assigning a type to the variable 'scipy.io' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.io', import_133210)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from scipy.io.harwell_boeing import HBFile, HBInfo' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')
import_133212 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.harwell_boeing')

if (type(import_133212) is not StypyTypeError):

    if (import_133212 != 'pyd_module'):
        __import__(import_133212)
        sys_modules_133213 = sys.modules[import_133212]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.harwell_boeing', sys_modules_133213.module_type_store, module_type_store, ['HBFile', 'HBInfo'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_133213, sys_modules_133213.module_type_store, module_type_store)
    else:
        from scipy.io.harwell_boeing import HBFile, HBInfo

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.harwell_boeing', None, module_type_store, ['HBFile', 'HBInfo'], [HBFile, HBInfo])

else:
    # Assigning a type to the variable 'scipy.io.harwell_boeing' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'scipy.io.harwell_boeing', import_133212)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/tests/')


# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_133214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'No Title                                                                |No Key\n             9             4             1             4\nRUA                      100           100            10             0\n(26I3)          (26I3)          (3E23.15)\n1  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3\n3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3\n3  3  3  3  3  3  3  4  4  4  6  6  6  6  6  6  6  6  6  6  6  8  9  9  9  9\n9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9 11\n37 71 89 18 30 45 70 19 25 52\n2.971243799687726e-01  3.662366682877375e-01  4.786962174699534e-01\n6.490068647991184e-01  6.617490424831662e-02  8.870370343191623e-01\n4.196478590163001e-01  5.649603072111251e-01  9.934423887087086e-01\n6.912334991524289e-01\n')
# Assigning a type to the variable 'SIMPLE' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'SIMPLE', str_133214)

# Assigning a Call to a Name (line 38):

# Assigning a Call to a Name (line 38):

# Call to coo_matrix(...): (line 38)
# Processing the call arguments (line 38)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_133216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_133217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
float_133218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133218)
# Adding element type (line 40)
float_133219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133219)
# Adding element type (line 40)
float_133220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133220)
# Adding element type (line 40)
float_133221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 60), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133221)
# Adding element type (line 40)
float_133222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133222)
# Adding element type (line 40)
float_133223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133223)
# Adding element type (line 40)
float_133224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 46), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133224)
# Adding element type (line 40)
float_133225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133225)
# Adding element type (line 40)
float_133226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133226)
# Adding element type (line 40)
float_133227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), tuple_133217, float_133227)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_133216, tuple_133217)
# Adding element type (line 40)

# Call to array(...): (line 43)
# Processing the call arguments (line 43)

# Obtaining an instance of the builtin type 'list' (line 43)
list_133230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)

# Obtaining an instance of the builtin type 'list' (line 43)
list_133231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
int_133232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133232)
# Adding element type (line 43)
int_133233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133233)
# Adding element type (line 43)
int_133234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133234)
# Adding element type (line 43)
int_133235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133235)
# Adding element type (line 43)
int_133236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133236)
# Adding element type (line 43)
int_133237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133237)
# Adding element type (line 43)
int_133238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133238)
# Adding element type (line 43)
int_133239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133239)
# Adding element type (line 43)
int_133240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133240)
# Adding element type (line 43)
int_133241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 60), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 23), list_133231, int_133241)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 22), list_133230, list_133231)
# Adding element type (line 43)

# Obtaining an instance of the builtin type 'list' (line 44)
list_133242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
int_133243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133243)
# Adding element type (line 44)
int_133244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133244)
# Adding element type (line 44)
int_133245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133245)
# Adding element type (line 44)
int_133246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133246)
# Adding element type (line 44)
int_133247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133247)
# Adding element type (line 44)
int_133248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133248)
# Adding element type (line 44)
int_133249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133249)
# Adding element type (line 44)
int_133250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133250)
# Adding element type (line 44)
int_133251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 54), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133251)
# Adding element type (line 44)
int_133252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 58), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 23), list_133242, int_133252)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 22), list_133230, list_133242)

# Processing the call keyword arguments (line 43)
kwargs_133253 = {}
# Getting the type of 'np' (line 43)
np_133228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'np', False)
# Obtaining the member 'array' of a type (line 43)
array_133229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 13), np_133228, 'array')
# Calling array(args, kwargs) (line 43)
array_call_result_133254 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), array_133229, *[list_133230], **kwargs_133253)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_133216, array_call_result_133254)

# Processing the call keyword arguments (line 38)
kwargs_133255 = {}
# Getting the type of 'coo_matrix' (line 38)
coo_matrix_133215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'coo_matrix', False)
# Calling coo_matrix(args, kwargs) (line 38)
coo_matrix_call_result_133256 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), coo_matrix_133215, *[tuple_133216], **kwargs_133255)

# Assigning a type to the variable 'SIMPLE_MATRIX' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'SIMPLE_MATRIX', coo_matrix_call_result_133256)

@norecursion
def assert_csc_almost_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_csc_almost_equal'
    module_type_store = module_type_store.open_function_context('assert_csc_almost_equal', 47, 0, False)
    
    # Passed parameters checking function
    assert_csc_almost_equal.stypy_localization = localization
    assert_csc_almost_equal.stypy_type_of_self = None
    assert_csc_almost_equal.stypy_type_store = module_type_store
    assert_csc_almost_equal.stypy_function_name = 'assert_csc_almost_equal'
    assert_csc_almost_equal.stypy_param_names_list = ['r', 'l']
    assert_csc_almost_equal.stypy_varargs_param_name = None
    assert_csc_almost_equal.stypy_kwargs_param_name = None
    assert_csc_almost_equal.stypy_call_defaults = defaults
    assert_csc_almost_equal.stypy_call_varargs = varargs
    assert_csc_almost_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_csc_almost_equal', ['r', 'l'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_csc_almost_equal', localization, ['r', 'l'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_csc_almost_equal(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to csc_matrix(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'r' (line 48)
    r_133258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'r', False)
    # Processing the call keyword arguments (line 48)
    kwargs_133259 = {}
    # Getting the type of 'csc_matrix' (line 48)
    csc_matrix_133257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 48)
    csc_matrix_call_result_133260 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), csc_matrix_133257, *[r_133258], **kwargs_133259)
    
    # Assigning a type to the variable 'r' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'r', csc_matrix_call_result_133260)
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to csc_matrix(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'l' (line 49)
    l_133262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'l', False)
    # Processing the call keyword arguments (line 49)
    kwargs_133263 = {}
    # Getting the type of 'csc_matrix' (line 49)
    csc_matrix_133261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'csc_matrix', False)
    # Calling csc_matrix(args, kwargs) (line 49)
    csc_matrix_call_result_133264 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), csc_matrix_133261, *[l_133262], **kwargs_133263)
    
    # Assigning a type to the variable 'l' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'l', csc_matrix_call_result_133264)
    
    # Call to assert_equal(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'r' (line 50)
    r_133266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'r', False)
    # Obtaining the member 'indptr' of a type (line 50)
    indptr_133267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), r_133266, 'indptr')
    # Getting the type of 'l' (line 50)
    l_133268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'l', False)
    # Obtaining the member 'indptr' of a type (line 50)
    indptr_133269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 27), l_133268, 'indptr')
    # Processing the call keyword arguments (line 50)
    kwargs_133270 = {}
    # Getting the type of 'assert_equal' (line 50)
    assert_equal_133265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 50)
    assert_equal_call_result_133271 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), assert_equal_133265, *[indptr_133267, indptr_133269], **kwargs_133270)
    
    
    # Call to assert_equal(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'r' (line 51)
    r_133273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'r', False)
    # Obtaining the member 'indices' of a type (line 51)
    indices_133274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), r_133273, 'indices')
    # Getting the type of 'l' (line 51)
    l_133275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'l', False)
    # Obtaining the member 'indices' of a type (line 51)
    indices_133276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), l_133275, 'indices')
    # Processing the call keyword arguments (line 51)
    kwargs_133277 = {}
    # Getting the type of 'assert_equal' (line 51)
    assert_equal_133272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 51)
    assert_equal_call_result_133278 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), assert_equal_133272, *[indices_133274, indices_133276], **kwargs_133277)
    
    
    # Call to assert_array_almost_equal_nulp(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'r' (line 52)
    r_133280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'r', False)
    # Obtaining the member 'data' of a type (line 52)
    data_133281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 35), r_133280, 'data')
    # Getting the type of 'l' (line 52)
    l_133282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'l', False)
    # Obtaining the member 'data' of a type (line 52)
    data_133283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), l_133282, 'data')
    int_133284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 51), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_133285 = {}
    # Getting the type of 'assert_array_almost_equal_nulp' (line 52)
    assert_array_almost_equal_nulp_133279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'assert_array_almost_equal_nulp', False)
    # Calling assert_array_almost_equal_nulp(args, kwargs) (line 52)
    assert_array_almost_equal_nulp_call_result_133286 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), assert_array_almost_equal_nulp_133279, *[data_133281, data_133283, int_133284], **kwargs_133285)
    
    
    # ################# End of 'assert_csc_almost_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_csc_almost_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_133287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_133287)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_csc_almost_equal'
    return stypy_return_type_133287

# Assigning a type to the variable 'assert_csc_almost_equal' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'assert_csc_almost_equal', assert_csc_almost_equal)
# Declaration of the 'TestHBReader' class

class TestHBReader(object, ):

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHBReader.test_simple.__dict__.__setitem__('stypy_localization', localization)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_function_name', 'TestHBReader.test_simple')
        TestHBReader.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestHBReader.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHBReader.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHBReader.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to hb_read(...): (line 57)
        # Processing the call arguments (line 57)
        
        # Call to StringIO(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'SIMPLE' (line 57)
        SIMPLE_133290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'SIMPLE', False)
        # Processing the call keyword arguments (line 57)
        kwargs_133291 = {}
        # Getting the type of 'StringIO' (line 57)
        StringIO_133289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 20), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 57)
        StringIO_call_result_133292 = invoke(stypy.reporting.localization.Localization(__file__, 57, 20), StringIO_133289, *[SIMPLE_133290], **kwargs_133291)
        
        # Processing the call keyword arguments (line 57)
        kwargs_133293 = {}
        # Getting the type of 'hb_read' (line 57)
        hb_read_133288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'hb_read', False)
        # Calling hb_read(args, kwargs) (line 57)
        hb_read_call_result_133294 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), hb_read_133288, *[StringIO_call_result_133292], **kwargs_133293)
        
        # Assigning a type to the variable 'm' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'm', hb_read_call_result_133294)
        
        # Call to assert_csc_almost_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'm' (line 58)
        m_133296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'm', False)
        # Getting the type of 'SIMPLE_MATRIX' (line 58)
        SIMPLE_MATRIX_133297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'SIMPLE_MATRIX', False)
        # Processing the call keyword arguments (line 58)
        kwargs_133298 = {}
        # Getting the type of 'assert_csc_almost_equal' (line 58)
        assert_csc_almost_equal_133295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_csc_almost_equal', False)
        # Calling assert_csc_almost_equal(args, kwargs) (line 58)
        assert_csc_almost_equal_call_result_133299 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_csc_almost_equal_133295, *[m_133296, SIMPLE_MATRIX_133297], **kwargs_133298)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_133300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133300)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_133300


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 0, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHBReader.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestHBReader' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'TestHBReader', TestHBReader)
# Declaration of the 'TestRBRoundtrip' class

class TestRBRoundtrip(object, ):

    @norecursion
    def test_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simple'
        module_type_store = module_type_store.open_function_context('test_simple', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_localization', localization)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_function_name', 'TestRBRoundtrip.test_simple')
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRBRoundtrip.test_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRBRoundtrip.test_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simple(...)' code ##################

        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to tocsc(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_133308 = {}
        
        # Call to rand(...): (line 63)
        # Processing the call arguments (line 63)
        int_133302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'int')
        int_133303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 23), 'int')
        float_133304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'float')
        # Processing the call keyword arguments (line 63)
        kwargs_133305 = {}
        # Getting the type of 'rand' (line 63)
        rand_133301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'rand', False)
        # Calling rand(args, kwargs) (line 63)
        rand_call_result_133306 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), rand_133301, *[int_133302, int_133303, float_133304], **kwargs_133305)
        
        # Obtaining the member 'tocsc' of a type (line 63)
        tocsc_133307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 13), rand_call_result_133306, 'tocsc')
        # Calling tocsc(args, kwargs) (line 63)
        tocsc_call_result_133309 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), tocsc_133307, *[], **kwargs_133308)
        
        # Assigning a type to the variable 'rm' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'rm', tocsc_call_result_133309)
        
        # Assigning a Call to a Tuple (line 64):
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_133310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to mkstemp(...): (line 64)
        # Processing the call keyword arguments (line 64)
        str_133313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'str', 'rb')
        keyword_133314 = str_133313
        kwargs_133315 = {'suffix': keyword_133314}
        # Getting the type of 'tempfile' (line 64)
        tempfile_133311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'tempfile', False)
        # Obtaining the member 'mkstemp' of a type (line 64)
        mkstemp_133312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), tempfile_133311, 'mkstemp')
        # Calling mkstemp(args, kwargs) (line 64)
        mkstemp_call_result_133316 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), mkstemp_133312, *[], **kwargs_133315)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___133317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), mkstemp_call_result_133316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_133318 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___133317, int_133310)
        
        # Assigning a type to the variable 'tuple_var_assignment_133194' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_133194', subscript_call_result_133318)
        
        # Assigning a Subscript to a Name (line 64):
        
        # Obtaining the type of the subscript
        int_133319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 8), 'int')
        
        # Call to mkstemp(...): (line 64)
        # Processing the call keyword arguments (line 64)
        str_133322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 47), 'str', 'rb')
        keyword_133323 = str_133322
        kwargs_133324 = {'suffix': keyword_133323}
        # Getting the type of 'tempfile' (line 64)
        tempfile_133320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'tempfile', False)
        # Obtaining the member 'mkstemp' of a type (line 64)
        mkstemp_133321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), tempfile_133320, 'mkstemp')
        # Calling mkstemp(args, kwargs) (line 64)
        mkstemp_call_result_133325 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), mkstemp_133321, *[], **kwargs_133324)
        
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___133326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), mkstemp_call_result_133325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_133327 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), getitem___133326, int_133319)
        
        # Assigning a type to the variable 'tuple_var_assignment_133195' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_133195', subscript_call_result_133327)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_133194' (line 64)
        tuple_var_assignment_133194_133328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_133194')
        # Assigning a type to the variable 'fd' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'fd', tuple_var_assignment_133194_133328)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_var_assignment_133195' (line 64)
        tuple_var_assignment_133195_133329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_var_assignment_133195')
        # Assigning a type to the variable 'filename' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'filename', tuple_var_assignment_133195_133329)
        
        # Try-finally block (line 65)
        
        # Call to hb_write(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'filename' (line 66)
        filename_133331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'filename', False)
        # Getting the type of 'rm' (line 66)
        rm_133332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 'rm', False)
        
        # Call to from_data(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'rm' (line 66)
        rm_133335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 52), 'rm', False)
        # Processing the call keyword arguments (line 66)
        kwargs_133336 = {}
        # Getting the type of 'HBInfo' (line 66)
        HBInfo_133333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'HBInfo', False)
        # Obtaining the member 'from_data' of a type (line 66)
        from_data_133334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 35), HBInfo_133333, 'from_data')
        # Calling from_data(args, kwargs) (line 66)
        from_data_call_result_133337 = invoke(stypy.reporting.localization.Localization(__file__, 66, 35), from_data_133334, *[rm_133335], **kwargs_133336)
        
        # Processing the call keyword arguments (line 66)
        kwargs_133338 = {}
        # Getting the type of 'hb_write' (line 66)
        hb_write_133330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'hb_write', False)
        # Calling hb_write(args, kwargs) (line 66)
        hb_write_call_result_133339 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), hb_write_133330, *[filename_133331, rm_133332, from_data_call_result_133337], **kwargs_133338)
        
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to hb_read(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'filename' (line 67)
        filename_133341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'filename', False)
        # Processing the call keyword arguments (line 67)
        kwargs_133342 = {}
        # Getting the type of 'hb_read' (line 67)
        hb_read_133340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'hb_read', False)
        # Calling hb_read(args, kwargs) (line 67)
        hb_read_call_result_133343 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), hb_read_133340, *[filename_133341], **kwargs_133342)
        
        # Assigning a type to the variable 'm' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'm', hb_read_call_result_133343)
        
        # finally branch of the try-finally block (line 65)
        
        # Call to close(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'fd' (line 69)
        fd_133346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'fd', False)
        # Processing the call keyword arguments (line 69)
        kwargs_133347 = {}
        # Getting the type of 'os' (line 69)
        os_133344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'os', False)
        # Obtaining the member 'close' of a type (line 69)
        close_133345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), os_133344, 'close')
        # Calling close(args, kwargs) (line 69)
        close_call_result_133348 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), close_133345, *[fd_133346], **kwargs_133347)
        
        
        # Call to remove(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'filename' (line 70)
        filename_133351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 22), 'filename', False)
        # Processing the call keyword arguments (line 70)
        kwargs_133352 = {}
        # Getting the type of 'os' (line 70)
        os_133349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'os', False)
        # Obtaining the member 'remove' of a type (line 70)
        remove_133350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), os_133349, 'remove')
        # Calling remove(args, kwargs) (line 70)
        remove_call_result_133353 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), remove_133350, *[filename_133351], **kwargs_133352)
        
        
        
        # Call to assert_csc_almost_equal(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'm' (line 72)
        m_133355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'm', False)
        # Getting the type of 'rm' (line 72)
        rm_133356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'rm', False)
        # Processing the call keyword arguments (line 72)
        kwargs_133357 = {}
        # Getting the type of 'assert_csc_almost_equal' (line 72)
        assert_csc_almost_equal_133354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_csc_almost_equal', False)
        # Calling assert_csc_almost_equal(args, kwargs) (line 72)
        assert_csc_almost_equal_call_result_133358 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_csc_almost_equal_133354, *[m_133355, rm_133356], **kwargs_133357)
        
        
        # ################# End of 'test_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_133359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133359)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simple'
        return stypy_return_type_133359


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 0, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRBRoundtrip.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestRBRoundtrip' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'TestRBRoundtrip', TestRBRoundtrip)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
