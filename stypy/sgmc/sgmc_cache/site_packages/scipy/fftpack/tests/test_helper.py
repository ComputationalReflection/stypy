
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Created by Pearu Peterson, September 2002
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __usage__ = '''
6: Build fftpack:
7:   python setup_fftpack.py build
8: Run tests if scipy is installed:
9:   python -c 'import scipy;scipy.fftpack.test(<level>)'
10: Run tests if fftpack is not installed:
11:   python tests/test_helper.py [<level>]
12: '''
13: 
14: from numpy.testing import (assert_array_almost_equal,
15:                            assert_equal, assert_)
16: from scipy.fftpack import fftshift,ifftshift,fftfreq,rfftfreq
17: from scipy.fftpack.helper import next_fast_len
18: 
19: from numpy import pi, random
20: 
21: 
22: class TestFFTShift(object):
23: 
24:     def test_definition(self):
25:         x = [0,1,2,3,4,-4,-3,-2,-1]
26:         y = [-4,-3,-2,-1,0,1,2,3,4]
27:         assert_array_almost_equal(fftshift(x),y)
28:         assert_array_almost_equal(ifftshift(y),x)
29:         x = [0,1,2,3,4,-5,-4,-3,-2,-1]
30:         y = [-5,-4,-3,-2,-1,0,1,2,3,4]
31:         assert_array_almost_equal(fftshift(x),y)
32:         assert_array_almost_equal(ifftshift(y),x)
33: 
34:     def test_inverse(self):
35:         for n in [1,4,9,100,211]:
36:             x = random.random((n,))
37:             assert_array_almost_equal(ifftshift(fftshift(x)),x)
38: 
39: 
40: class TestFFTFreq(object):
41: 
42:     def test_definition(self):
43:         x = [0,1,2,3,4,-4,-3,-2,-1]
44:         assert_array_almost_equal(9*fftfreq(9),x)
45:         assert_array_almost_equal(9*pi*fftfreq(9,pi),x)
46:         x = [0,1,2,3,4,-5,-4,-3,-2,-1]
47:         assert_array_almost_equal(10*fftfreq(10),x)
48:         assert_array_almost_equal(10*pi*fftfreq(10,pi),x)
49: 
50: 
51: class TestRFFTFreq(object):
52: 
53:     def test_definition(self):
54:         x = [0,1,1,2,2,3,3,4,4]
55:         assert_array_almost_equal(9*rfftfreq(9),x)
56:         assert_array_almost_equal(9*pi*rfftfreq(9,pi),x)
57:         x = [0,1,1,2,2,3,3,4,4,5]
58:         assert_array_almost_equal(10*rfftfreq(10),x)
59:         assert_array_almost_equal(10*pi*rfftfreq(10,pi),x)
60: 
61: 
62: class TestNextOptLen(object):
63: 
64:     def test_next_opt_len(self):
65:         random.seed(1234)
66: 
67:         def nums():
68:             for j in range(1, 1000):
69:                 yield j
70:             yield 2**5 * 3**5 * 4**5 + 1
71: 
72:         for n in nums():
73:             m = next_fast_len(n)
74:             msg = "n=%d, m=%d" % (n, m)
75: 
76:             assert_(m >= n, msg)
77: 
78:             # check regularity
79:             k = m
80:             for d in [2, 3, 5]:
81:                 while True:
82:                     a, b = divmod(k, d)
83:                     if b == 0:
84:                         k = a
85:                     else:
86:                         break
87:             assert_equal(k, 1, err_msg=msg)
88: 
89:     def test_next_opt_len_strict(self):
90:         hams = {
91:             1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8, 8: 8, 14: 15, 15: 15,
92:             16: 16, 17: 18, 1021: 1024, 1536: 1536, 51200000: 51200000,
93:             510183360: 510183360, 510183360 + 1: 512000000,
94:             511000000: 512000000,
95:             854296875: 854296875, 854296875 + 1: 859963392,
96:             196608000000: 196608000000, 196608000000 + 1: 196830000000,
97:             8789062500000: 8789062500000, 8789062500000 + 1: 8796093022208,
98:             206391214080000: 206391214080000,
99:             206391214080000 + 1: 206624260800000,
100:             470184984576000: 470184984576000,
101:             470184984576000 + 1: 470715894135000,
102:             7222041363087360: 7222041363087360,
103:             7222041363087360 + 1: 7230196133913600,
104:             # power of 5    5**23
105:             11920928955078125: 11920928955078125,
106:             11920928955078125 - 1: 11920928955078125,
107:             # power of 3    3**34
108:             16677181699666569: 16677181699666569,
109:             16677181699666569 - 1: 16677181699666569,
110:             # power of 2   2**54
111:             18014398509481984: 18014398509481984,
112:             18014398509481984 - 1: 18014398509481984,
113:             # above this, int(ceil(n)) == int(ceil(n+1))
114:             19200000000000000: 19200000000000000,
115:             19200000000000000 + 1: 19221679687500000,
116:             288230376151711744: 288230376151711744,
117:             288230376151711744 + 1: 288325195312500000,
118:             288325195312500000 - 1: 288325195312500000,
119:             288325195312500000: 288325195312500000,
120:             288325195312500000 + 1: 288555831593533440,
121:             # power of 3    3**83
122:             3990838394187339929534246675572349035227 - 1:
123:                 3990838394187339929534246675572349035227,
124:             3990838394187339929534246675572349035227:
125:                 3990838394187339929534246675572349035227,
126:             # power of 2     2**135
127:             43556142965880123323311949751266331066368 - 1:
128:                 43556142965880123323311949751266331066368,
129:             43556142965880123323311949751266331066368:
130:                 43556142965880123323311949751266331066368,
131:             # power of 5      5**57
132:             6938893903907228377647697925567626953125 - 1:
133:                 6938893903907228377647697925567626953125,
134:             6938893903907228377647697925567626953125:
135:                 6938893903907228377647697925567626953125,
136:             # http://www.drdobbs.com/228700538
137:             # 2**96 * 3**1 * 5**13
138:             290142196707511001929482240000000000000 - 1:
139:                 290142196707511001929482240000000000000,
140:             290142196707511001929482240000000000000:
141:                 290142196707511001929482240000000000000,
142:             290142196707511001929482240000000000000 + 1:
143:                 290237644800000000000000000000000000000,
144:             # 2**36 * 3**69 * 5**7
145:             4479571262811807241115438439905203543080960000000 - 1:
146:                 4479571262811807241115438439905203543080960000000,
147:             4479571262811807241115438439905203543080960000000:
148:                 4479571262811807241115438439905203543080960000000,
149:             4479571262811807241115438439905203543080960000000 + 1:
150:                 4480327901140333639941336854183943340032000000000,
151:             # 2**37 * 3**44 * 5**42
152:             30774090693237851027531250000000000000000000000000000000000000 - 1:
153:                 30774090693237851027531250000000000000000000000000000000000000,
154:             30774090693237851027531250000000000000000000000000000000000000:
155:                 30774090693237851027531250000000000000000000000000000000000000,
156:             30774090693237851027531250000000000000000000000000000000000000 + 1:
157:                 30778180617309082445871527002041377406962596539492679680000000,
158:         }
159:         for x, y in hams.items():
160:             assert_equal(next_fast_len(x), y)
161: 
162: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_23683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', "\nBuild fftpack:\n  python setup_fftpack.py build\nRun tests if scipy is installed:\n  python -c 'import scipy;scipy.fftpack.test(<level>)'\nRun tests if fftpack is not installed:\n  python tests/test_helper.py [<level>]\n")
# Assigning a type to the variable '__usage__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__usage__', str_23683)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_equal, assert_' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_23684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_23684) is not StypyTypeError):

    if (import_23684 != 'pyd_module'):
        __import__(import_23684)
        sys_modules_23685 = sys.modules[import_23684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_23685.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_23685, sys_modules_23685.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_equal', 'assert_'], [assert_array_almost_equal, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_23684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.fftpack import fftshift, ifftshift, fftfreq, rfftfreq' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_23686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack')

if (type(import_23686) is not StypyTypeError):

    if (import_23686 != 'pyd_module'):
        __import__(import_23686)
        sys_modules_23687 = sys.modules[import_23686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', sys_modules_23687.module_type_store, module_type_store, ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_23687, sys_modules_23687.module_type_store, module_type_store)
    else:
        from scipy.fftpack import fftshift, ifftshift, fftfreq, rfftfreq

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', None, module_type_store, ['fftshift', 'ifftshift', 'fftfreq', 'rfftfreq'], [fftshift, ifftshift, fftfreq, rfftfreq])

else:
    # Assigning a type to the variable 'scipy.fftpack' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', import_23686)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.fftpack.helper import next_fast_len' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_23688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.fftpack.helper')

if (type(import_23688) is not StypyTypeError):

    if (import_23688 != 'pyd_module'):
        __import__(import_23688)
        sys_modules_23689 = sys.modules[import_23688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.fftpack.helper', sys_modules_23689.module_type_store, module_type_store, ['next_fast_len'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_23689, sys_modules_23689.module_type_store, module_type_store)
    else:
        from scipy.fftpack.helper import next_fast_len

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.fftpack.helper', None, module_type_store, ['next_fast_len'], [next_fast_len])

else:
    # Assigning a type to the variable 'scipy.fftpack.helper' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.fftpack.helper', import_23688)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from numpy import pi, random' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_23690 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy')

if (type(import_23690) is not StypyTypeError):

    if (import_23690 != 'pyd_module'):
        __import__(import_23690)
        sys_modules_23691 = sys.modules[import_23690]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy', sys_modules_23691.module_type_store, module_type_store, ['pi', 'random'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_23691, sys_modules_23691.module_type_store, module_type_store)
    else:
        from numpy import pi, random

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy', None, module_type_store, ['pi', 'random'], [pi, random])

else:
    # Assigning a type to the variable 'numpy' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'numpy', import_23690)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

# Declaration of the 'TestFFTShift' class

class TestFFTShift(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_function_name', 'TestFFTShift.test_definition')
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFFTShift.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTShift.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Assigning a List to a Name (line 25):
        
        # Assigning a List to a Name (line 25):
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_23692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        int_23693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23693)
        # Adding element type (line 25)
        int_23694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23694)
        # Adding element type (line 25)
        int_23695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23695)
        # Adding element type (line 25)
        int_23696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23696)
        # Adding element type (line 25)
        int_23697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23697)
        # Adding element type (line 25)
        int_23698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23698)
        # Adding element type (line 25)
        int_23699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23699)
        # Adding element type (line 25)
        int_23700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23700)
        # Adding element type (line 25)
        int_23701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), list_23692, int_23701)
        
        # Assigning a type to the variable 'x' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'x', list_23692)
        
        # Assigning a List to a Name (line 26):
        
        # Assigning a List to a Name (line 26):
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_23702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_23703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23703)
        # Adding element type (line 26)
        int_23704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23704)
        # Adding element type (line 26)
        int_23705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23705)
        # Adding element type (line 26)
        int_23706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23706)
        # Adding element type (line 26)
        int_23707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23707)
        # Adding element type (line 26)
        int_23708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23708)
        # Adding element type (line 26)
        int_23709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23709)
        # Adding element type (line 26)
        int_23710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23710)
        # Adding element type (line 26)
        int_23711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 12), list_23702, int_23711)
        
        # Assigning a type to the variable 'y' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'y', list_23702)
        
        # Call to assert_array_almost_equal(...): (line 27)
        # Processing the call arguments (line 27)
        
        # Call to fftshift(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'x' (line 27)
        x_23714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'x', False)
        # Processing the call keyword arguments (line 27)
        kwargs_23715 = {}
        # Getting the type of 'fftshift' (line 27)
        fftshift_23713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'fftshift', False)
        # Calling fftshift(args, kwargs) (line 27)
        fftshift_call_result_23716 = invoke(stypy.reporting.localization.Localization(__file__, 27, 34), fftshift_23713, *[x_23714], **kwargs_23715)
        
        # Getting the type of 'y' (line 27)
        y_23717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 46), 'y', False)
        # Processing the call keyword arguments (line 27)
        kwargs_23718 = {}
        # Getting the type of 'assert_array_almost_equal' (line 27)
        assert_array_almost_equal_23712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 27)
        assert_array_almost_equal_call_result_23719 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), assert_array_almost_equal_23712, *[fftshift_call_result_23716, y_23717], **kwargs_23718)
        
        
        # Call to assert_array_almost_equal(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to ifftshift(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'y' (line 28)
        y_23722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'y', False)
        # Processing the call keyword arguments (line 28)
        kwargs_23723 = {}
        # Getting the type of 'ifftshift' (line 28)
        ifftshift_23721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ifftshift', False)
        # Calling ifftshift(args, kwargs) (line 28)
        ifftshift_call_result_23724 = invoke(stypy.reporting.localization.Localization(__file__, 28, 34), ifftshift_23721, *[y_23722], **kwargs_23723)
        
        # Getting the type of 'x' (line 28)
        x_23725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 47), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_23726 = {}
        # Getting the type of 'assert_array_almost_equal' (line 28)
        assert_array_almost_equal_23720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 28)
        assert_array_almost_equal_call_result_23727 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert_array_almost_equal_23720, *[ifftshift_call_result_23724, x_23725], **kwargs_23726)
        
        
        # Assigning a List to a Name (line 29):
        
        # Assigning a List to a Name (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_23728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        int_23729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23729)
        # Adding element type (line 29)
        int_23730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23730)
        # Adding element type (line 29)
        int_23731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23731)
        # Adding element type (line 29)
        int_23732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23732)
        # Adding element type (line 29)
        int_23733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23733)
        # Adding element type (line 29)
        int_23734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23734)
        # Adding element type (line 29)
        int_23735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23735)
        # Adding element type (line 29)
        int_23736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23736)
        # Adding element type (line 29)
        int_23737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23737)
        # Adding element type (line 29)
        int_23738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 12), list_23728, int_23738)
        
        # Assigning a type to the variable 'x' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'x', list_23728)
        
        # Assigning a List to a Name (line 30):
        
        # Assigning a List to a Name (line 30):
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_23739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        int_23740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23740)
        # Adding element type (line 30)
        int_23741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23741)
        # Adding element type (line 30)
        int_23742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23742)
        # Adding element type (line 30)
        int_23743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23743)
        # Adding element type (line 30)
        int_23744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23744)
        # Adding element type (line 30)
        int_23745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23745)
        # Adding element type (line 30)
        int_23746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23746)
        # Adding element type (line 30)
        int_23747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23747)
        # Adding element type (line 30)
        int_23748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23748)
        # Adding element type (line 30)
        int_23749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 12), list_23739, int_23749)
        
        # Assigning a type to the variable 'y' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'y', list_23739)
        
        # Call to assert_array_almost_equal(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Call to fftshift(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'x' (line 31)
        x_23752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'x', False)
        # Processing the call keyword arguments (line 31)
        kwargs_23753 = {}
        # Getting the type of 'fftshift' (line 31)
        fftshift_23751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'fftshift', False)
        # Calling fftshift(args, kwargs) (line 31)
        fftshift_call_result_23754 = invoke(stypy.reporting.localization.Localization(__file__, 31, 34), fftshift_23751, *[x_23752], **kwargs_23753)
        
        # Getting the type of 'y' (line 31)
        y_23755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 46), 'y', False)
        # Processing the call keyword arguments (line 31)
        kwargs_23756 = {}
        # Getting the type of 'assert_array_almost_equal' (line 31)
        assert_array_almost_equal_23750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 31)
        assert_array_almost_equal_call_result_23757 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_array_almost_equal_23750, *[fftshift_call_result_23754, y_23755], **kwargs_23756)
        
        
        # Call to assert_array_almost_equal(...): (line 32)
        # Processing the call arguments (line 32)
        
        # Call to ifftshift(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'y' (line 32)
        y_23760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 44), 'y', False)
        # Processing the call keyword arguments (line 32)
        kwargs_23761 = {}
        # Getting the type of 'ifftshift' (line 32)
        ifftshift_23759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'ifftshift', False)
        # Calling ifftshift(args, kwargs) (line 32)
        ifftshift_call_result_23762 = invoke(stypy.reporting.localization.Localization(__file__, 32, 34), ifftshift_23759, *[y_23760], **kwargs_23761)
        
        # Getting the type of 'x' (line 32)
        x_23763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 47), 'x', False)
        # Processing the call keyword arguments (line 32)
        kwargs_23764 = {}
        # Getting the type of 'assert_array_almost_equal' (line 32)
        assert_array_almost_equal_23758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 32)
        assert_array_almost_equal_call_result_23765 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_array_almost_equal_23758, *[ifftshift_call_result_23762, x_23763], **kwargs_23764)
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_23766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_23766


    @norecursion
    def test_inverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inverse'
        module_type_store = module_type_store.open_function_context('test_inverse', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_localization', localization)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_function_name', 'TestFFTShift.test_inverse')
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFFTShift.test_inverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTShift.test_inverse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inverse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inverse(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_23767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        int_23768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), list_23767, int_23768)
        # Adding element type (line 35)
        int_23769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), list_23767, int_23769)
        # Adding element type (line 35)
        int_23770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), list_23767, int_23770)
        # Adding element type (line 35)
        int_23771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), list_23767, int_23771)
        # Adding element type (line 35)
        int_23772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 17), list_23767, int_23772)
        
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), list_23767)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_23773 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), list_23767)
        # Assigning a type to the variable 'n' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'n', for_loop_var_23773)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to random(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_23776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        # Getting the type of 'n' (line 36)
        n_23777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 31), tuple_23776, n_23777)
        
        # Processing the call keyword arguments (line 36)
        kwargs_23778 = {}
        # Getting the type of 'random' (line 36)
        random_23774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'random', False)
        # Obtaining the member 'random' of a type (line 36)
        random_23775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), random_23774, 'random')
        # Calling random(args, kwargs) (line 36)
        random_call_result_23779 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), random_23775, *[tuple_23776], **kwargs_23778)
        
        # Assigning a type to the variable 'x' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'x', random_call_result_23779)
        
        # Call to assert_array_almost_equal(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to ifftshift(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to fftshift(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'x' (line 37)
        x_23783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 57), 'x', False)
        # Processing the call keyword arguments (line 37)
        kwargs_23784 = {}
        # Getting the type of 'fftshift' (line 37)
        fftshift_23782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'fftshift', False)
        # Calling fftshift(args, kwargs) (line 37)
        fftshift_call_result_23785 = invoke(stypy.reporting.localization.Localization(__file__, 37, 48), fftshift_23782, *[x_23783], **kwargs_23784)
        
        # Processing the call keyword arguments (line 37)
        kwargs_23786 = {}
        # Getting the type of 'ifftshift' (line 37)
        ifftshift_23781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'ifftshift', False)
        # Calling ifftshift(args, kwargs) (line 37)
        ifftshift_call_result_23787 = invoke(stypy.reporting.localization.Localization(__file__, 37, 38), ifftshift_23781, *[fftshift_call_result_23785], **kwargs_23786)
        
        # Getting the type of 'x' (line 37)
        x_23788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 61), 'x', False)
        # Processing the call keyword arguments (line 37)
        kwargs_23789 = {}
        # Getting the type of 'assert_array_almost_equal' (line 37)
        assert_array_almost_equal_23780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 37)
        assert_array_almost_equal_call_result_23790 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), assert_array_almost_equal_23780, *[ifftshift_call_result_23787, x_23788], **kwargs_23789)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_inverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inverse' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_23791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inverse'
        return stypy_return_type_23791


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 0, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTShift.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFFTShift' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'TestFFTShift', TestFFTShift)
# Declaration of the 'TestFFTFreq' class

class TestFFTFreq(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_function_name', 'TestFFTFreq.test_definition')
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFFTFreq.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTFreq.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Assigning a List to a Name (line 43):
        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_23792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        int_23793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23793)
        # Adding element type (line 43)
        int_23794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23794)
        # Adding element type (line 43)
        int_23795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23795)
        # Adding element type (line 43)
        int_23796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23796)
        # Adding element type (line 43)
        int_23797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23797)
        # Adding element type (line 43)
        int_23798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23798)
        # Adding element type (line 43)
        int_23799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23799)
        # Adding element type (line 43)
        int_23800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23800)
        # Adding element type (line 43)
        int_23801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 12), list_23792, int_23801)
        
        # Assigning a type to the variable 'x' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'x', list_23792)
        
        # Call to assert_array_almost_equal(...): (line 44)
        # Processing the call arguments (line 44)
        int_23803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
        
        # Call to fftfreq(...): (line 44)
        # Processing the call arguments (line 44)
        int_23805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 44), 'int')
        # Processing the call keyword arguments (line 44)
        kwargs_23806 = {}
        # Getting the type of 'fftfreq' (line 44)
        fftfreq_23804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 44)
        fftfreq_call_result_23807 = invoke(stypy.reporting.localization.Localization(__file__, 44, 36), fftfreq_23804, *[int_23805], **kwargs_23806)
        
        # Applying the binary operator '*' (line 44)
        result_mul_23808 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 34), '*', int_23803, fftfreq_call_result_23807)
        
        # Getting the type of 'x' (line 44)
        x_23809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 47), 'x', False)
        # Processing the call keyword arguments (line 44)
        kwargs_23810 = {}
        # Getting the type of 'assert_array_almost_equal' (line 44)
        assert_array_almost_equal_23802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 44)
        assert_array_almost_equal_call_result_23811 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), assert_array_almost_equal_23802, *[result_mul_23808, x_23809], **kwargs_23810)
        
        
        # Call to assert_array_almost_equal(...): (line 45)
        # Processing the call arguments (line 45)
        int_23813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'int')
        # Getting the type of 'pi' (line 45)
        pi_23814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'pi', False)
        # Applying the binary operator '*' (line 45)
        result_mul_23815 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '*', int_23813, pi_23814)
        
        
        # Call to fftfreq(...): (line 45)
        # Processing the call arguments (line 45)
        int_23817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 47), 'int')
        # Getting the type of 'pi' (line 45)
        pi_23818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 49), 'pi', False)
        # Processing the call keyword arguments (line 45)
        kwargs_23819 = {}
        # Getting the type of 'fftfreq' (line 45)
        fftfreq_23816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 45)
        fftfreq_call_result_23820 = invoke(stypy.reporting.localization.Localization(__file__, 45, 39), fftfreq_23816, *[int_23817, pi_23818], **kwargs_23819)
        
        # Applying the binary operator '*' (line 45)
        result_mul_23821 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 38), '*', result_mul_23815, fftfreq_call_result_23820)
        
        # Getting the type of 'x' (line 45)
        x_23822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 53), 'x', False)
        # Processing the call keyword arguments (line 45)
        kwargs_23823 = {}
        # Getting the type of 'assert_array_almost_equal' (line 45)
        assert_array_almost_equal_23812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 45)
        assert_array_almost_equal_call_result_23824 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_array_almost_equal_23812, *[result_mul_23821, x_23822], **kwargs_23823)
        
        
        # Assigning a List to a Name (line 46):
        
        # Assigning a List to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_23825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_23826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23826)
        # Adding element type (line 46)
        int_23827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23827)
        # Adding element type (line 46)
        int_23828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23828)
        # Adding element type (line 46)
        int_23829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23829)
        # Adding element type (line 46)
        int_23830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23830)
        # Adding element type (line 46)
        int_23831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23831)
        # Adding element type (line 46)
        int_23832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23832)
        # Adding element type (line 46)
        int_23833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23833)
        # Adding element type (line 46)
        int_23834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23834)
        # Adding element type (line 46)
        int_23835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_23825, int_23835)
        
        # Assigning a type to the variable 'x' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'x', list_23825)
        
        # Call to assert_array_almost_equal(...): (line 47)
        # Processing the call arguments (line 47)
        int_23837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 34), 'int')
        
        # Call to fftfreq(...): (line 47)
        # Processing the call arguments (line 47)
        int_23839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 45), 'int')
        # Processing the call keyword arguments (line 47)
        kwargs_23840 = {}
        # Getting the type of 'fftfreq' (line 47)
        fftfreq_23838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 47)
        fftfreq_call_result_23841 = invoke(stypy.reporting.localization.Localization(__file__, 47, 37), fftfreq_23838, *[int_23839], **kwargs_23840)
        
        # Applying the binary operator '*' (line 47)
        result_mul_23842 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 34), '*', int_23837, fftfreq_call_result_23841)
        
        # Getting the type of 'x' (line 47)
        x_23843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 49), 'x', False)
        # Processing the call keyword arguments (line 47)
        kwargs_23844 = {}
        # Getting the type of 'assert_array_almost_equal' (line 47)
        assert_array_almost_equal_23836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 47)
        assert_array_almost_equal_call_result_23845 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_array_almost_equal_23836, *[result_mul_23842, x_23843], **kwargs_23844)
        
        
        # Call to assert_array_almost_equal(...): (line 48)
        # Processing the call arguments (line 48)
        int_23847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'int')
        # Getting the type of 'pi' (line 48)
        pi_23848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'pi', False)
        # Applying the binary operator '*' (line 48)
        result_mul_23849 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 34), '*', int_23847, pi_23848)
        
        
        # Call to fftfreq(...): (line 48)
        # Processing the call arguments (line 48)
        int_23851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 48), 'int')
        # Getting the type of 'pi' (line 48)
        pi_23852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 51), 'pi', False)
        # Processing the call keyword arguments (line 48)
        kwargs_23853 = {}
        # Getting the type of 'fftfreq' (line 48)
        fftfreq_23850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 48)
        fftfreq_call_result_23854 = invoke(stypy.reporting.localization.Localization(__file__, 48, 40), fftfreq_23850, *[int_23851, pi_23852], **kwargs_23853)
        
        # Applying the binary operator '*' (line 48)
        result_mul_23855 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 39), '*', result_mul_23849, fftfreq_call_result_23854)
        
        # Getting the type of 'x' (line 48)
        x_23856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 55), 'x', False)
        # Processing the call keyword arguments (line 48)
        kwargs_23857 = {}
        # Getting the type of 'assert_array_almost_equal' (line 48)
        assert_array_almost_equal_23846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 48)
        assert_array_almost_equal_call_result_23858 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_array_almost_equal_23846, *[result_mul_23855, x_23856], **kwargs_23857)
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_23859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_23859


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 0, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFFTFreq.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFFTFreq' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'TestFFTFreq', TestFFTFreq)
# Declaration of the 'TestRFFTFreq' class

class TestRFFTFreq(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_function_name', 'TestRFFTFreq.test_definition')
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRFFTFreq.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRFFTFreq.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        # Assigning a List to a Name (line 54):
        
        # Assigning a List to a Name (line 54):
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_23860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_23861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23861)
        # Adding element type (line 54)
        int_23862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23862)
        # Adding element type (line 54)
        int_23863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23863)
        # Adding element type (line 54)
        int_23864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23864)
        # Adding element type (line 54)
        int_23865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23865)
        # Adding element type (line 54)
        int_23866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23866)
        # Adding element type (line 54)
        int_23867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23867)
        # Adding element type (line 54)
        int_23868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23868)
        # Adding element type (line 54)
        int_23869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), list_23860, int_23869)
        
        # Assigning a type to the variable 'x' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'x', list_23860)
        
        # Call to assert_array_almost_equal(...): (line 55)
        # Processing the call arguments (line 55)
        int_23871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
        
        # Call to rfftfreq(...): (line 55)
        # Processing the call arguments (line 55)
        int_23873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'int')
        # Processing the call keyword arguments (line 55)
        kwargs_23874 = {}
        # Getting the type of 'rfftfreq' (line 55)
        rfftfreq_23872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'rfftfreq', False)
        # Calling rfftfreq(args, kwargs) (line 55)
        rfftfreq_call_result_23875 = invoke(stypy.reporting.localization.Localization(__file__, 55, 36), rfftfreq_23872, *[int_23873], **kwargs_23874)
        
        # Applying the binary operator '*' (line 55)
        result_mul_23876 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 34), '*', int_23871, rfftfreq_call_result_23875)
        
        # Getting the type of 'x' (line 55)
        x_23877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 48), 'x', False)
        # Processing the call keyword arguments (line 55)
        kwargs_23878 = {}
        # Getting the type of 'assert_array_almost_equal' (line 55)
        assert_array_almost_equal_23870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 55)
        assert_array_almost_equal_call_result_23879 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_array_almost_equal_23870, *[result_mul_23876, x_23877], **kwargs_23878)
        
        
        # Call to assert_array_almost_equal(...): (line 56)
        # Processing the call arguments (line 56)
        int_23881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 34), 'int')
        # Getting the type of 'pi' (line 56)
        pi_23882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'pi', False)
        # Applying the binary operator '*' (line 56)
        result_mul_23883 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 34), '*', int_23881, pi_23882)
        
        
        # Call to rfftfreq(...): (line 56)
        # Processing the call arguments (line 56)
        int_23885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 48), 'int')
        # Getting the type of 'pi' (line 56)
        pi_23886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 50), 'pi', False)
        # Processing the call keyword arguments (line 56)
        kwargs_23887 = {}
        # Getting the type of 'rfftfreq' (line 56)
        rfftfreq_23884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'rfftfreq', False)
        # Calling rfftfreq(args, kwargs) (line 56)
        rfftfreq_call_result_23888 = invoke(stypy.reporting.localization.Localization(__file__, 56, 39), rfftfreq_23884, *[int_23885, pi_23886], **kwargs_23887)
        
        # Applying the binary operator '*' (line 56)
        result_mul_23889 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 38), '*', result_mul_23883, rfftfreq_call_result_23888)
        
        # Getting the type of 'x' (line 56)
        x_23890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'x', False)
        # Processing the call keyword arguments (line 56)
        kwargs_23891 = {}
        # Getting the type of 'assert_array_almost_equal' (line 56)
        assert_array_almost_equal_23880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 56)
        assert_array_almost_equal_call_result_23892 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_array_almost_equal_23880, *[result_mul_23889, x_23890], **kwargs_23891)
        
        
        # Assigning a List to a Name (line 57):
        
        # Assigning a List to a Name (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_23893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_23894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23894)
        # Adding element type (line 57)
        int_23895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23895)
        # Adding element type (line 57)
        int_23896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23896)
        # Adding element type (line 57)
        int_23897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23897)
        # Adding element type (line 57)
        int_23898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23898)
        # Adding element type (line 57)
        int_23899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23899)
        # Adding element type (line 57)
        int_23900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23900)
        # Adding element type (line 57)
        int_23901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23901)
        # Adding element type (line 57)
        int_23902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23902)
        # Adding element type (line 57)
        int_23903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), list_23893, int_23903)
        
        # Assigning a type to the variable 'x' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'x', list_23893)
        
        # Call to assert_array_almost_equal(...): (line 58)
        # Processing the call arguments (line 58)
        int_23905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 34), 'int')
        
        # Call to rfftfreq(...): (line 58)
        # Processing the call arguments (line 58)
        int_23907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 46), 'int')
        # Processing the call keyword arguments (line 58)
        kwargs_23908 = {}
        # Getting the type of 'rfftfreq' (line 58)
        rfftfreq_23906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'rfftfreq', False)
        # Calling rfftfreq(args, kwargs) (line 58)
        rfftfreq_call_result_23909 = invoke(stypy.reporting.localization.Localization(__file__, 58, 37), rfftfreq_23906, *[int_23907], **kwargs_23908)
        
        # Applying the binary operator '*' (line 58)
        result_mul_23910 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 34), '*', int_23905, rfftfreq_call_result_23909)
        
        # Getting the type of 'x' (line 58)
        x_23911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 50), 'x', False)
        # Processing the call keyword arguments (line 58)
        kwargs_23912 = {}
        # Getting the type of 'assert_array_almost_equal' (line 58)
        assert_array_almost_equal_23904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 58)
        assert_array_almost_equal_call_result_23913 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_array_almost_equal_23904, *[result_mul_23910, x_23911], **kwargs_23912)
        
        
        # Call to assert_array_almost_equal(...): (line 59)
        # Processing the call arguments (line 59)
        int_23915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'int')
        # Getting the type of 'pi' (line 59)
        pi_23916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'pi', False)
        # Applying the binary operator '*' (line 59)
        result_mul_23917 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 34), '*', int_23915, pi_23916)
        
        
        # Call to rfftfreq(...): (line 59)
        # Processing the call arguments (line 59)
        int_23919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 49), 'int')
        # Getting the type of 'pi' (line 59)
        pi_23920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'pi', False)
        # Processing the call keyword arguments (line 59)
        kwargs_23921 = {}
        # Getting the type of 'rfftfreq' (line 59)
        rfftfreq_23918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'rfftfreq', False)
        # Calling rfftfreq(args, kwargs) (line 59)
        rfftfreq_call_result_23922 = invoke(stypy.reporting.localization.Localization(__file__, 59, 40), rfftfreq_23918, *[int_23919, pi_23920], **kwargs_23921)
        
        # Applying the binary operator '*' (line 59)
        result_mul_23923 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 39), '*', result_mul_23917, rfftfreq_call_result_23922)
        
        # Getting the type of 'x' (line 59)
        x_23924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'x', False)
        # Processing the call keyword arguments (line 59)
        kwargs_23925 = {}
        # Getting the type of 'assert_array_almost_equal' (line 59)
        assert_array_almost_equal_23914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 59)
        assert_array_almost_equal_call_result_23926 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_array_almost_equal_23914, *[result_mul_23923, x_23924], **kwargs_23925)
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_23927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23927)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_23927


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRFFTFreq.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRFFTFreq' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'TestRFFTFreq', TestRFFTFreq)
# Declaration of the 'TestNextOptLen' class

class TestNextOptLen(object, ):

    @norecursion
    def test_next_opt_len(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_next_opt_len'
        module_type_store = module_type_store.open_function_context('test_next_opt_len', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_localization', localization)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_function_name', 'TestNextOptLen.test_next_opt_len')
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_param_names_list', [])
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNextOptLen.test_next_opt_len.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNextOptLen.test_next_opt_len', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_next_opt_len', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_next_opt_len(...)' code ##################

        
        # Call to seed(...): (line 65)
        # Processing the call arguments (line 65)
        int_23930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'int')
        # Processing the call keyword arguments (line 65)
        kwargs_23931 = {}
        # Getting the type of 'random' (line 65)
        random_23928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'random', False)
        # Obtaining the member 'seed' of a type (line 65)
        seed_23929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), random_23928, 'seed')
        # Calling seed(args, kwargs) (line 65)
        seed_call_result_23932 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), seed_23929, *[int_23930], **kwargs_23931)
        

        @norecursion
        def nums(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'nums'
            module_type_store = module_type_store.open_function_context('nums', 67, 8, False)
            
            # Passed parameters checking function
            nums.stypy_localization = localization
            nums.stypy_type_of_self = None
            nums.stypy_type_store = module_type_store
            nums.stypy_function_name = 'nums'
            nums.stypy_param_names_list = []
            nums.stypy_varargs_param_name = None
            nums.stypy_kwargs_param_name = None
            nums.stypy_call_defaults = defaults
            nums.stypy_call_varargs = varargs
            nums.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'nums', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'nums', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'nums(...)' code ##################

            
            
            # Call to range(...): (line 68)
            # Processing the call arguments (line 68)
            int_23934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
            int_23935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 30), 'int')
            # Processing the call keyword arguments (line 68)
            kwargs_23936 = {}
            # Getting the type of 'range' (line 68)
            range_23933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'range', False)
            # Calling range(args, kwargs) (line 68)
            range_call_result_23937 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), range_23933, *[int_23934, int_23935], **kwargs_23936)
            
            # Testing the type of a for loop iterable (line 68)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 12), range_call_result_23937)
            # Getting the type of the for loop variable (line 68)
            for_loop_var_23938 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 12), range_call_result_23937)
            # Assigning a type to the variable 'j' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'j', for_loop_var_23938)
            # SSA begins for a for statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'j' (line 69)
            j_23939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'j')
            GeneratorType_23940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), GeneratorType_23940, j_23939)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'stypy_return_type', GeneratorType_23940)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Creating a generator
            int_23941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'int')
            int_23942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 21), 'int')
            # Applying the binary operator '**' (line 70)
            result_pow_23943 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 18), '**', int_23941, int_23942)
            
            int_23944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'int')
            int_23945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'int')
            # Applying the binary operator '**' (line 70)
            result_pow_23946 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '**', int_23944, int_23945)
            
            # Applying the binary operator '*' (line 70)
            result_mul_23947 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 18), '*', result_pow_23943, result_pow_23946)
            
            int_23948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'int')
            int_23949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 35), 'int')
            # Applying the binary operator '**' (line 70)
            result_pow_23950 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 32), '**', int_23948, int_23949)
            
            # Applying the binary operator '*' (line 70)
            result_mul_23951 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 30), '*', result_mul_23947, result_pow_23950)
            
            int_23952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'int')
            # Applying the binary operator '+' (line 70)
            result_add_23953 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 18), '+', result_mul_23951, int_23952)
            
            GeneratorType_23954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), GeneratorType_23954, result_add_23953)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', GeneratorType_23954)
            
            # ################# End of 'nums(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'nums' in the type store
            # Getting the type of 'stypy_return_type' (line 67)
            stypy_return_type_23955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_23955)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'nums'
            return stypy_return_type_23955

        # Assigning a type to the variable 'nums' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'nums', nums)
        
        
        # Call to nums(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_23957 = {}
        # Getting the type of 'nums' (line 72)
        nums_23956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'nums', False)
        # Calling nums(args, kwargs) (line 72)
        nums_call_result_23958 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), nums_23956, *[], **kwargs_23957)
        
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 8), nums_call_result_23958)
        # Getting the type of the for loop variable (line 72)
        for_loop_var_23959 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 8), nums_call_result_23958)
        # Assigning a type to the variable 'n' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'n', for_loop_var_23959)
        # SSA begins for a for statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 73):
        
        # Assigning a Call to a Name (line 73):
        
        # Call to next_fast_len(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'n' (line 73)
        n_23961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'n', False)
        # Processing the call keyword arguments (line 73)
        kwargs_23962 = {}
        # Getting the type of 'next_fast_len' (line 73)
        next_fast_len_23960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'next_fast_len', False)
        # Calling next_fast_len(args, kwargs) (line 73)
        next_fast_len_call_result_23963 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), next_fast_len_23960, *[n_23961], **kwargs_23962)
        
        # Assigning a type to the variable 'm' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'm', next_fast_len_call_result_23963)
        
        # Assigning a BinOp to a Name (line 74):
        
        # Assigning a BinOp to a Name (line 74):
        str_23964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'str', 'n=%d, m=%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_23965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        # Getting the type of 'n' (line 74)
        n_23966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 34), tuple_23965, n_23966)
        # Adding element type (line 74)
        # Getting the type of 'm' (line 74)
        m_23967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 37), 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 34), tuple_23965, m_23967)
        
        # Applying the binary operator '%' (line 74)
        result_mod_23968 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 18), '%', str_23964, tuple_23965)
        
        # Assigning a type to the variable 'msg' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'msg', result_mod_23968)
        
        # Call to assert_(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Getting the type of 'm' (line 76)
        m_23970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'm', False)
        # Getting the type of 'n' (line 76)
        n_23971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'n', False)
        # Applying the binary operator '>=' (line 76)
        result_ge_23972 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 20), '>=', m_23970, n_23971)
        
        # Getting the type of 'msg' (line 76)
        msg_23973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'msg', False)
        # Processing the call keyword arguments (line 76)
        kwargs_23974 = {}
        # Getting the type of 'assert_' (line 76)
        assert__23969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 76)
        assert__call_result_23975 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), assert__23969, *[result_ge_23972, msg_23973], **kwargs_23974)
        
        
        # Assigning a Name to a Name (line 79):
        
        # Assigning a Name to a Name (line 79):
        # Getting the type of 'm' (line 79)
        m_23976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'm')
        # Assigning a type to the variable 'k' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'k', m_23976)
        
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_23977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        int_23978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_23977, int_23978)
        # Adding element type (line 80)
        int_23979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_23977, int_23979)
        # Adding element type (line 80)
        int_23980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 21), list_23977, int_23980)
        
        # Testing the type of a for loop iterable (line 80)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 12), list_23977)
        # Getting the type of the for loop variable (line 80)
        for_loop_var_23981 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 12), list_23977)
        # Assigning a type to the variable 'd' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'd', for_loop_var_23981)
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'True' (line 81)
        True_23982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'True')
        # Testing the type of an if condition (line 81)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 16), True_23982)
        # SSA begins for while statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Tuple (line 82):
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_23983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
        
        # Call to divmod(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'k' (line 82)
        k_23985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'k', False)
        # Getting the type of 'd' (line 82)
        d_23986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'd', False)
        # Processing the call keyword arguments (line 82)
        kwargs_23987 = {}
        # Getting the type of 'divmod' (line 82)
        divmod_23984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'divmod', False)
        # Calling divmod(args, kwargs) (line 82)
        divmod_call_result_23988 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), divmod_23984, *[k_23985, d_23986], **kwargs_23987)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___23989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), divmod_call_result_23988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_23990 = invoke(stypy.reporting.localization.Localization(__file__, 82, 20), getitem___23989, int_23983)
        
        # Assigning a type to the variable 'tuple_var_assignment_23681' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple_var_assignment_23681', subscript_call_result_23990)
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        int_23991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 20), 'int')
        
        # Call to divmod(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'k' (line 82)
        k_23993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'k', False)
        # Getting the type of 'd' (line 82)
        d_23994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'd', False)
        # Processing the call keyword arguments (line 82)
        kwargs_23995 = {}
        # Getting the type of 'divmod' (line 82)
        divmod_23992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'divmod', False)
        # Calling divmod(args, kwargs) (line 82)
        divmod_call_result_23996 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), divmod_23992, *[k_23993, d_23994], **kwargs_23995)
        
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___23997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), divmod_call_result_23996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_23998 = invoke(stypy.reporting.localization.Localization(__file__, 82, 20), getitem___23997, int_23991)
        
        # Assigning a type to the variable 'tuple_var_assignment_23682' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple_var_assignment_23682', subscript_call_result_23998)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_23681' (line 82)
        tuple_var_assignment_23681_23999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple_var_assignment_23681')
        # Assigning a type to the variable 'a' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'a', tuple_var_assignment_23681_23999)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_var_assignment_23682' (line 82)
        tuple_var_assignment_23682_24000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'tuple_var_assignment_23682')
        # Assigning a type to the variable 'b' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'b', tuple_var_assignment_23682_24000)
        
        
        # Getting the type of 'b' (line 83)
        b_24001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'b')
        int_24002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'int')
        # Applying the binary operator '==' (line 83)
        result_eq_24003 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 23), '==', b_24001, int_24002)
        
        # Testing the type of an if condition (line 83)
        if_condition_24004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 20), result_eq_24003)
        # Assigning a type to the variable 'if_condition_24004' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'if_condition_24004', if_condition_24004)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 84):
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'a' (line 84)
        a_24005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'a')
        # Assigning a type to the variable 'k' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'k', a_24005)
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'k' (line 87)
        k_24007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'k', False)
        int_24008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'int')
        # Processing the call keyword arguments (line 87)
        # Getting the type of 'msg' (line 87)
        msg_24009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'msg', False)
        keyword_24010 = msg_24009
        kwargs_24011 = {'err_msg': keyword_24010}
        # Getting the type of 'assert_equal' (line 87)
        assert_equal_24006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 87)
        assert_equal_call_result_24012 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), assert_equal_24006, *[k_24007, int_24008], **kwargs_24011)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_next_opt_len(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_next_opt_len' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_24013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_next_opt_len'
        return stypy_return_type_24013


    @norecursion
    def test_next_opt_len_strict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_next_opt_len_strict'
        module_type_store = module_type_store.open_function_context('test_next_opt_len_strict', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_localization', localization)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_function_name', 'TestNextOptLen.test_next_opt_len_strict')
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_param_names_list', [])
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNextOptLen.test_next_opt_len_strict.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNextOptLen.test_next_opt_len_strict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_next_opt_len_strict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_next_opt_len_strict(...)' code ##################

        
        # Assigning a Dict to a Name (line 90):
        
        # Assigning a Dict to a Name (line 90):
        
        # Obtaining an instance of the builtin type 'dict' (line 90)
        dict_24014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 90)
        # Adding element type (key, value) (line 90)
        int_24015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 12), 'int')
        int_24016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24015, int_24016))
        # Adding element type (key, value) (line 90)
        int_24017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 18), 'int')
        int_24018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24017, int_24018))
        # Adding element type (key, value) (line 90)
        int_24019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'int')
        int_24020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24019, int_24020))
        # Adding element type (key, value) (line 90)
        int_24021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 30), 'int')
        int_24022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 33), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24021, int_24022))
        # Adding element type (key, value) (line 90)
        int_24023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 36), 'int')
        int_24024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24023, int_24024))
        # Adding element type (key, value) (line 90)
        int_24025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'int')
        int_24026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 45), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24025, int_24026))
        # Adding element type (key, value) (line 90)
        int_24027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 48), 'int')
        int_24028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 51), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24027, int_24028))
        # Adding element type (key, value) (line 90)
        int_24029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 54), 'int')
        int_24030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 57), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24029, int_24030))
        # Adding element type (key, value) (line 90)
        int_24031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 60), 'int')
        int_24032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 64), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24031, int_24032))
        # Adding element type (key, value) (line 90)
        int_24033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 68), 'int')
        int_24034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 72), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24033, int_24034))
        # Adding element type (key, value) (line 90)
        int_24035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
        int_24036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24035, int_24036))
        # Adding element type (key, value) (line 90)
        int_24037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 20), 'int')
        int_24038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24037, int_24038))
        # Adding element type (key, value) (line 90)
        int_24039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'int')
        int_24040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 34), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24039, int_24040))
        # Adding element type (key, value) (line 90)
        int_24041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 40), 'int')
        int_24042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 46), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24041, int_24042))
        # Adding element type (key, value) (line 90)
        int_24043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'int')
        int_24044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 62), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24043, int_24044))
        # Adding element type (key, value) (line 90)
        int_24045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'int')
        int_24046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24045, int_24046))
        # Adding element type (key, value) (line 90)
        int_24047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'int')
        int_24048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 46), 'int')
        # Applying the binary operator '+' (line 93)
        result_add_24049 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '+', int_24047, int_24048)
        
        int_24050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24049, int_24050))
        # Adding element type (key, value) (line 90)
        int_24051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 12), 'int')
        int_24052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24051, int_24052))
        # Adding element type (key, value) (line 90)
        int_24053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'int')
        int_24054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (int_24053, int_24054))
        # Adding element type (key, value) (line 90)
        int_24055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 34), 'int')
        int_24056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 46), 'int')
        # Applying the binary operator '+' (line 95)
        result_add_24057 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 34), '+', int_24055, int_24056)
        
        int_24058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 49), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24057, int_24058))
        # Adding element type (key, value) (line 90)
        long_24059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 12), 'long')
        long_24060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 26), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24059, long_24060))
        # Adding element type (key, value) (line 90)
        long_24061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 40), 'long')
        int_24062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 55), 'int')
        # Applying the binary operator '+' (line 96)
        result_add_24063 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 40), '+', long_24061, int_24062)
        
        long_24064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 58), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24063, long_24064))
        # Adding element type (key, value) (line 90)
        long_24065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'long')
        long_24066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 27), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24065, long_24066))
        # Adding element type (key, value) (line 90)
        long_24067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 42), 'long')
        int_24068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 58), 'int')
        # Applying the binary operator '+' (line 97)
        result_add_24069 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 42), '+', long_24067, int_24068)
        
        long_24070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 61), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24069, long_24070))
        # Adding element type (key, value) (line 90)
        long_24071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'long')
        long_24072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24071, long_24072))
        # Adding element type (key, value) (line 90)
        long_24073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 12), 'long')
        int_24074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_24075 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), '+', long_24073, int_24074)
        
        long_24076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 33), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24075, long_24076))
        # Adding element type (key, value) (line 90)
        long_24077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 12), 'long')
        long_24078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24077, long_24078))
        # Adding element type (key, value) (line 90)
        long_24079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'long')
        int_24080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'int')
        # Applying the binary operator '+' (line 101)
        result_add_24081 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 12), '+', long_24079, int_24080)
        
        long_24082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 33), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24081, long_24082))
        # Adding element type (key, value) (line 90)
        long_24083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'long')
        long_24084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24083, long_24084))
        # Adding element type (key, value) (line 90)
        long_24085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 12), 'long')
        int_24086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'int')
        # Applying the binary operator '+' (line 103)
        result_add_24087 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 12), '+', long_24085, int_24086)
        
        long_24088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24087, long_24088))
        # Adding element type (key, value) (line 90)
        long_24089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'long')
        long_24090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 31), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24089, long_24090))
        # Adding element type (key, value) (line 90)
        long_24091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'long')
        int_24092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 32), 'int')
        # Applying the binary operator '-' (line 106)
        result_sub_24093 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '-', long_24091, int_24092)
        
        long_24094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24093, long_24094))
        # Adding element type (key, value) (line 90)
        long_24095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'long')
        long_24096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24095, long_24096))
        # Adding element type (key, value) (line 90)
        long_24097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'long')
        int_24098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'int')
        # Applying the binary operator '-' (line 109)
        result_sub_24099 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '-', long_24097, int_24098)
        
        long_24100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24099, long_24100))
        # Adding element type (key, value) (line 90)
        long_24101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'long')
        long_24102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24101, long_24102))
        # Adding element type (key, value) (line 90)
        long_24103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'long')
        int_24104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'int')
        # Applying the binary operator '-' (line 112)
        result_sub_24105 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 12), '-', long_24103, int_24104)
        
        long_24106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24105, long_24106))
        # Adding element type (key, value) (line 90)
        long_24107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'long')
        long_24108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24107, long_24108))
        # Adding element type (key, value) (line 90)
        long_24109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'long')
        int_24110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 32), 'int')
        # Applying the binary operator '+' (line 115)
        result_add_24111 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 12), '+', long_24109, int_24110)
        
        long_24112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24111, long_24112))
        # Adding element type (key, value) (line 90)
        long_24113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 12), 'long')
        long_24114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24113, long_24114))
        # Adding element type (key, value) (line 90)
        long_24115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 12), 'long')
        int_24116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'int')
        # Applying the binary operator '+' (line 117)
        result_add_24117 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '+', long_24115, int_24116)
        
        long_24118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 36), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24117, long_24118))
        # Adding element type (key, value) (line 90)
        long_24119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'long')
        int_24120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 33), 'int')
        # Applying the binary operator '-' (line 118)
        result_sub_24121 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 12), '-', long_24119, int_24120)
        
        long_24122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 36), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24121, long_24122))
        # Adding element type (key, value) (line 90)
        long_24123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'long')
        long_24124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24123, long_24124))
        # Adding element type (key, value) (line 90)
        long_24125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'long')
        int_24126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'int')
        # Applying the binary operator '+' (line 120)
        result_add_24127 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '+', long_24125, int_24126)
        
        long_24128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24127, long_24128))
        # Adding element type (key, value) (line 90)
        long_24129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 12), 'long')
        int_24130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 55), 'int')
        # Applying the binary operator '-' (line 122)
        result_sub_24131 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 12), '-', long_24129, int_24130)
        
        long_24132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24131, long_24132))
        # Adding element type (key, value) (line 90)
        long_24133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 12), 'long')
        long_24134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24133, long_24134))
        # Adding element type (key, value) (line 90)
        long_24135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'long')
        int_24136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 56), 'int')
        # Applying the binary operator '-' (line 127)
        result_sub_24137 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 12), '-', long_24135, int_24136)
        
        long_24138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24137, long_24138))
        # Adding element type (key, value) (line 90)
        long_24139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 12), 'long')
        long_24140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24139, long_24140))
        # Adding element type (key, value) (line 90)
        long_24141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 12), 'long')
        int_24142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 55), 'int')
        # Applying the binary operator '-' (line 132)
        result_sub_24143 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 12), '-', long_24141, int_24142)
        
        long_24144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24143, long_24144))
        # Adding element type (key, value) (line 90)
        long_24145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 12), 'long')
        long_24146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24145, long_24146))
        # Adding element type (key, value) (line 90)
        long_24147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'long')
        int_24148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 54), 'int')
        # Applying the binary operator '-' (line 138)
        result_sub_24149 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '-', long_24147, int_24148)
        
        long_24150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24149, long_24150))
        # Adding element type (key, value) (line 90)
        long_24151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 12), 'long')
        long_24152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24151, long_24152))
        # Adding element type (key, value) (line 90)
        long_24153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 12), 'long')
        int_24154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 54), 'int')
        # Applying the binary operator '+' (line 142)
        result_add_24155 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 12), '+', long_24153, int_24154)
        
        long_24156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24155, long_24156))
        # Adding element type (key, value) (line 90)
        long_24157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 12), 'long')
        int_24158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 64), 'int')
        # Applying the binary operator '-' (line 145)
        result_sub_24159 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 12), '-', long_24157, int_24158)
        
        long_24160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24159, long_24160))
        # Adding element type (key, value) (line 90)
        long_24161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'long')
        long_24162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24161, long_24162))
        # Adding element type (key, value) (line 90)
        long_24163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'long')
        int_24164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 64), 'int')
        # Applying the binary operator '+' (line 149)
        result_add_24165 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 12), '+', long_24163, int_24164)
        
        long_24166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24165, long_24166))
        # Adding element type (key, value) (line 90)
        long_24167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'long')
        int_24168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 77), 'int')
        # Applying the binary operator '-' (line 152)
        result_sub_24169 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '-', long_24167, int_24168)
        
        long_24170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_sub_24169, long_24170))
        # Adding element type (key, value) (line 90)
        long_24171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 12), 'long')
        long_24172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (long_24171, long_24172))
        # Adding element type (key, value) (line 90)
        long_24173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'long')
        int_24174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 77), 'int')
        # Applying the binary operator '+' (line 156)
        result_add_24175 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 12), '+', long_24173, int_24174)
        
        long_24176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'long')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), dict_24014, (result_add_24175, long_24176))
        
        # Assigning a type to the variable 'hams' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'hams', dict_24014)
        
        
        # Call to items(...): (line 159)
        # Processing the call keyword arguments (line 159)
        kwargs_24179 = {}
        # Getting the type of 'hams' (line 159)
        hams_24177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'hams', False)
        # Obtaining the member 'items' of a type (line 159)
        items_24178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), hams_24177, 'items')
        # Calling items(args, kwargs) (line 159)
        items_call_result_24180 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), items_24178, *[], **kwargs_24179)
        
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), items_call_result_24180)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_24181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), items_call_result_24180)
        # Assigning a type to the variable 'x' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_24181))
        # Assigning a type to the variable 'y' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), for_loop_var_24181))
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to next_fast_len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'x' (line 160)
        x_24184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 39), 'x', False)
        # Processing the call keyword arguments (line 160)
        kwargs_24185 = {}
        # Getting the type of 'next_fast_len' (line 160)
        next_fast_len_24183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 25), 'next_fast_len', False)
        # Calling next_fast_len(args, kwargs) (line 160)
        next_fast_len_call_result_24186 = invoke(stypy.reporting.localization.Localization(__file__, 160, 25), next_fast_len_24183, *[x_24184], **kwargs_24185)
        
        # Getting the type of 'y' (line 160)
        y_24187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'y', False)
        # Processing the call keyword arguments (line 160)
        kwargs_24188 = {}
        # Getting the type of 'assert_equal' (line 160)
        assert_equal_24182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 160)
        assert_equal_call_result_24189 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), assert_equal_24182, *[next_fast_len_call_result_24186, y_24187], **kwargs_24188)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_next_opt_len_strict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_next_opt_len_strict' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_24190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24190)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_next_opt_len_strict'
        return stypy_return_type_24190


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 0, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNextOptLen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNextOptLen' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'TestNextOptLen', TestNextOptLen)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
