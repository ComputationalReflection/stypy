
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #! /usr/bin/env python
2: from __future__ import division, print_function
3: 
4: import timeit
5: import numpy
6: 
7: 
8: ###############################################################################
9: #                               Global variables                              #
10: ###############################################################################
11: 
12: 
13: # Small arrays
14: xs = numpy.random.uniform(-1, 1, 6).reshape(2, 3)
15: ys = numpy.random.uniform(-1, 1, 6).reshape(2, 3)
16: zs = xs + 1j * ys
17: m1 = [[True, False, False], [False, False, True]]
18: m2 = [[True, False, True], [False, False, True]]
19: nmxs = numpy.ma.array(xs, mask=m1)
20: nmys = numpy.ma.array(ys, mask=m2)
21: nmzs = numpy.ma.array(zs, mask=m1)
22: 
23: # Big arrays
24: xl = numpy.random.uniform(-1, 1, 100*100).reshape(100, 100)
25: yl = numpy.random.uniform(-1, 1, 100*100).reshape(100, 100)
26: zl = xl + 1j * yl
27: maskx = xl > 0.8
28: masky = yl < -0.8
29: nmxl = numpy.ma.array(xl, mask=maskx)
30: nmyl = numpy.ma.array(yl, mask=masky)
31: nmzl = numpy.ma.array(zl, mask=maskx)
32: 
33: 
34: ###############################################################################
35: #                                 Functions                                   #
36: ###############################################################################
37: 
38: 
39: def timer(s, v='', nloop=500, nrep=3):
40:     units = ["s", "ms", "µs", "ns"]
41:     scaling = [1, 1e3, 1e6, 1e9]
42:     print("%s : %-50s : " % (v, s), end=' ')
43:     varnames = ["%ss,nm%ss,%sl,nm%sl" % tuple(x*4) for x in 'xyz']
44:     setup = 'from __main__ import numpy, ma, %s' % ','.join(varnames)
45:     Timer = timeit.Timer(stmt=s, setup=setup)
46:     best = min(Timer.repeat(nrep, nloop)) / nloop
47:     if best > 0.0:
48:         order = min(-int(numpy.floor(numpy.log10(best)) // 3), 3)
49:     else:
50:         order = 3
51:     print("%d loops, best of %d: %.*g %s per loop" % (nloop, nrep,
52:                                                       3,
53:                                                       best * scaling[order],
54:                                                       units[order]))
55: 
56: 
57: def compare_functions_1v(func, nloop=500,
58:                        xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl):
59:     funcname = func.__name__
60:     print("-"*50)
61:     print("%s on small arrays" % funcname)
62:     module, data = "numpy.ma", "nmxs"
63:     timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
64: 
65:     print("%s on large arrays" % funcname)
66:     module, data = "numpy.ma", "nmxl"
67:     timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
68:     return
69: 
70: def compare_methods(methodname, args, vars='x', nloop=500, test=True,
71:                     xs=xs, nmxs=nmxs, xl=xl, nmxl=nmxl):
72:     print("-"*50)
73:     print("%s on small arrays" % methodname)
74:     data, ver = "nm%ss" % vars, 'numpy.ma'
75:     timer("%(data)s.%(methodname)s(%(args)s)" % locals(), v=ver, nloop=nloop)
76: 
77:     print("%s on large arrays" % methodname)
78:     data, ver = "nm%sl" % vars, 'numpy.ma'
79:     timer("%(data)s.%(methodname)s(%(args)s)" % locals(), v=ver, nloop=nloop)
80:     return
81: 
82: def compare_functions_2v(func, nloop=500, test=True,
83:                        xs=xs, nmxs=nmxs,
84:                        ys=ys, nmys=nmys,
85:                        xl=xl, nmxl=nmxl,
86:                        yl=yl, nmyl=nmyl):
87:     funcname = func.__name__
88:     print("-"*50)
89:     print("%s on small arrays" % funcname)
90:     module, data = "numpy.ma", "nmxs,nmys"
91:     timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
92: 
93:     print("%s on large arrays" % funcname)
94:     module, data = "numpy.ma", "nmxl,nmyl"
95:     timer("%(module)s.%(funcname)s(%(data)s)" % locals(), v="%11s" % module, nloop=nloop)
96:     return
97: 
98: 
99: if __name__ == '__main__':
100:     compare_functions_1v(numpy.sin)
101:     compare_functions_1v(numpy.log)
102:     compare_functions_1v(numpy.sqrt)
103: 
104:     compare_functions_2v(numpy.multiply)
105:     compare_functions_2v(numpy.divide)
106:     compare_functions_2v(numpy.power)
107: 
108:     compare_methods('ravel', '', nloop=1000)
109:     compare_methods('conjugate', '', 'z', nloop=1000)
110:     compare_methods('transpose', '', nloop=1000)
111:     compare_methods('compressed', '', nloop=1000)
112:     compare_methods('__getitem__', '0', nloop=1000)
113:     compare_methods('__getitem__', '(0,0)', nloop=1000)
114:     compare_methods('__getitem__', '[0,-1]', nloop=1000)
115:     compare_methods('__setitem__', '0, 17', nloop=1000, test=False)
116:     compare_methods('__setitem__', '(0,0), 17', nloop=1000, test=False)
117: 
118:     print("-"*50)
119:     print("__setitem__ on small arrays")
120:     timer('nmxs.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ', nloop=10000)
121: 
122:     print("-"*50)
123:     print("__setitem__ on large arrays")
124:     timer('nmxl.__setitem__((-1,0),numpy.ma.masked)', 'numpy.ma   ', nloop=10000)
125: 
126:     print("-"*50)
127:     print("where on small arrays")
128:     timer('numpy.ma.where(nmxs>2,nmxs,nmys)', 'numpy.ma   ', nloop=1000)
129:     print("-"*50)
130:     print("where on large arrays")
131:     timer('numpy.ma.where(nmxl>2,nmxl,nmyl)', 'numpy.ma   ', nloop=100)
132: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import timeit' statement (line 4)
import timeit

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'timeit', timeit, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_138556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_138556) is not StypyTypeError):

    if (import_138556 != 'pyd_module'):
        __import__(import_138556)
        sys_modules_138557 = sys.modules[import_138556]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_138557.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_138556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a Call to a Name (line 14):

# Assigning a Call to a Name (line 14):

# Call to reshape(...): (line 14)
# Processing the call arguments (line 14)
int_138567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 44), 'int')
int_138568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 47), 'int')
# Processing the call keyword arguments (line 14)
kwargs_138569 = {}

# Call to uniform(...): (line 14)
# Processing the call arguments (line 14)
int_138561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'int')
int_138562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'int')
int_138563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'int')
# Processing the call keyword arguments (line 14)
kwargs_138564 = {}
# Getting the type of 'numpy' (line 14)
numpy_138558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'numpy', False)
# Obtaining the member 'random' of a type (line 14)
random_138559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), numpy_138558, 'random')
# Obtaining the member 'uniform' of a type (line 14)
uniform_138560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), random_138559, 'uniform')
# Calling uniform(args, kwargs) (line 14)
uniform_call_result_138565 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), uniform_138560, *[int_138561, int_138562, int_138563], **kwargs_138564)

# Obtaining the member 'reshape' of a type (line 14)
reshape_138566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), uniform_call_result_138565, 'reshape')
# Calling reshape(args, kwargs) (line 14)
reshape_call_result_138570 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), reshape_138566, *[int_138567, int_138568], **kwargs_138569)

# Assigning a type to the variable 'xs' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'xs', reshape_call_result_138570)

# Assigning a Call to a Name (line 15):

# Assigning a Call to a Name (line 15):

# Call to reshape(...): (line 15)
# Processing the call arguments (line 15)
int_138580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 44), 'int')
int_138581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 47), 'int')
# Processing the call keyword arguments (line 15)
kwargs_138582 = {}

# Call to uniform(...): (line 15)
# Processing the call arguments (line 15)
int_138574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
int_138575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
int_138576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'int')
# Processing the call keyword arguments (line 15)
kwargs_138577 = {}
# Getting the type of 'numpy' (line 15)
numpy_138571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'numpy', False)
# Obtaining the member 'random' of a type (line 15)
random_138572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), numpy_138571, 'random')
# Obtaining the member 'uniform' of a type (line 15)
uniform_138573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), random_138572, 'uniform')
# Calling uniform(args, kwargs) (line 15)
uniform_call_result_138578 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), uniform_138573, *[int_138574, int_138575, int_138576], **kwargs_138577)

# Obtaining the member 'reshape' of a type (line 15)
reshape_138579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), uniform_call_result_138578, 'reshape')
# Calling reshape(args, kwargs) (line 15)
reshape_call_result_138583 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), reshape_138579, *[int_138580, int_138581], **kwargs_138582)

# Assigning a type to the variable 'ys' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'ys', reshape_call_result_138583)

# Assigning a BinOp to a Name (line 16):

# Assigning a BinOp to a Name (line 16):
# Getting the type of 'xs' (line 16)
xs_138584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'xs')
complex_138585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'complex')
# Getting the type of 'ys' (line 16)
ys_138586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'ys')
# Applying the binary operator '*' (line 16)
result_mul_138587 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '*', complex_138585, ys_138586)

# Applying the binary operator '+' (line 16)
result_add_138588 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 5), '+', xs_138584, result_mul_138587)

# Assigning a type to the variable 'zs' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'zs', result_add_138588)

# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):

# Obtaining an instance of the builtin type 'list' (line 17)
list_138589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_138590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
# Getting the type of 'True' (line 17)
True_138591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 6), list_138590, True_138591)
# Adding element type (line 17)
# Getting the type of 'False' (line 17)
False_138592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 6), list_138590, False_138592)
# Adding element type (line 17)
# Getting the type of 'False' (line 17)
False_138593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 6), list_138590, False_138593)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), list_138589, list_138590)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_138594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
# Getting the type of 'False' (line 17)
False_138595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 28), list_138594, False_138595)
# Adding element type (line 17)
# Getting the type of 'False' (line 17)
False_138596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 36), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 28), list_138594, False_138596)
# Adding element type (line 17)
# Getting the type of 'True' (line 17)
True_138597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 43), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 28), list_138594, True_138597)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), list_138589, list_138594)

# Assigning a type to the variable 'm1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'm1', list_138589)

# Assigning a List to a Name (line 18):

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_138598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_138599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'True' (line 18)
True_138600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 6), list_138599, True_138600)
# Adding element type (line 18)
# Getting the type of 'False' (line 18)
False_138601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 6), list_138599, False_138601)
# Adding element type (line 18)
# Getting the type of 'True' (line 18)
True_138602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 6), list_138599, True_138602)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), list_138598, list_138599)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_138603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'False' (line 18)
False_138604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 28), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_138603, False_138604)
# Adding element type (line 18)
# Getting the type of 'False' (line 18)
False_138605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 35), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_138603, False_138605)
# Adding element type (line 18)
# Getting the type of 'True' (line 18)
True_138606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 42), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 27), list_138603, True_138606)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), list_138598, list_138603)

# Assigning a type to the variable 'm2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'm2', list_138598)

# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to array(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'xs' (line 19)
xs_138610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'xs', False)
# Processing the call keyword arguments (line 19)
# Getting the type of 'm1' (line 19)
m1_138611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'm1', False)
keyword_138612 = m1_138611
kwargs_138613 = {'mask': keyword_138612}
# Getting the type of 'numpy' (line 19)
numpy_138607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 19)
ma_138608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 7), numpy_138607, 'ma')
# Obtaining the member 'array' of a type (line 19)
array_138609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 7), ma_138608, 'array')
# Calling array(args, kwargs) (line 19)
array_call_result_138614 = invoke(stypy.reporting.localization.Localization(__file__, 19, 7), array_138609, *[xs_138610], **kwargs_138613)

# Assigning a type to the variable 'nmxs' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'nmxs', array_call_result_138614)

# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to array(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'ys' (line 20)
ys_138618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'ys', False)
# Processing the call keyword arguments (line 20)
# Getting the type of 'm2' (line 20)
m2_138619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 31), 'm2', False)
keyword_138620 = m2_138619
kwargs_138621 = {'mask': keyword_138620}
# Getting the type of 'numpy' (line 20)
numpy_138615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 20)
ma_138616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 7), numpy_138615, 'ma')
# Obtaining the member 'array' of a type (line 20)
array_138617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 7), ma_138616, 'array')
# Calling array(args, kwargs) (line 20)
array_call_result_138622 = invoke(stypy.reporting.localization.Localization(__file__, 20, 7), array_138617, *[ys_138618], **kwargs_138621)

# Assigning a type to the variable 'nmys' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'nmys', array_call_result_138622)

# Assigning a Call to a Name (line 21):

# Assigning a Call to a Name (line 21):

# Call to array(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'zs' (line 21)
zs_138626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'zs', False)
# Processing the call keyword arguments (line 21)
# Getting the type of 'm1' (line 21)
m1_138627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'm1', False)
keyword_138628 = m1_138627
kwargs_138629 = {'mask': keyword_138628}
# Getting the type of 'numpy' (line 21)
numpy_138623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 21)
ma_138624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), numpy_138623, 'ma')
# Obtaining the member 'array' of a type (line 21)
array_138625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), ma_138624, 'array')
# Calling array(args, kwargs) (line 21)
array_call_result_138630 = invoke(stypy.reporting.localization.Localization(__file__, 21, 7), array_138625, *[zs_138626], **kwargs_138629)

# Assigning a type to the variable 'nmzs' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'nmzs', array_call_result_138630)

# Assigning a Call to a Name (line 24):

# Assigning a Call to a Name (line 24):

# Call to reshape(...): (line 24)
# Processing the call arguments (line 24)
int_138642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 50), 'int')
int_138643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 55), 'int')
# Processing the call keyword arguments (line 24)
kwargs_138644 = {}

# Call to uniform(...): (line 24)
# Processing the call arguments (line 24)
int_138634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'int')
int_138635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'int')
int_138636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
int_138637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'int')
# Applying the binary operator '*' (line 24)
result_mul_138638 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 33), '*', int_138636, int_138637)

# Processing the call keyword arguments (line 24)
kwargs_138639 = {}
# Getting the type of 'numpy' (line 24)
numpy_138631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'numpy', False)
# Obtaining the member 'random' of a type (line 24)
random_138632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), numpy_138631, 'random')
# Obtaining the member 'uniform' of a type (line 24)
uniform_138633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), random_138632, 'uniform')
# Calling uniform(args, kwargs) (line 24)
uniform_call_result_138640 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), uniform_138633, *[int_138634, int_138635, result_mul_138638], **kwargs_138639)

# Obtaining the member 'reshape' of a type (line 24)
reshape_138641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), uniform_call_result_138640, 'reshape')
# Calling reshape(args, kwargs) (line 24)
reshape_call_result_138645 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), reshape_138641, *[int_138642, int_138643], **kwargs_138644)

# Assigning a type to the variable 'xl' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'xl', reshape_call_result_138645)

# Assigning a Call to a Name (line 25):

# Assigning a Call to a Name (line 25):

# Call to reshape(...): (line 25)
# Processing the call arguments (line 25)
int_138657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 50), 'int')
int_138658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 55), 'int')
# Processing the call keyword arguments (line 25)
kwargs_138659 = {}

# Call to uniform(...): (line 25)
# Processing the call arguments (line 25)
int_138649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'int')
int_138650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
int_138651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'int')
int_138652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'int')
# Applying the binary operator '*' (line 25)
result_mul_138653 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 33), '*', int_138651, int_138652)

# Processing the call keyword arguments (line 25)
kwargs_138654 = {}
# Getting the type of 'numpy' (line 25)
numpy_138646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'numpy', False)
# Obtaining the member 'random' of a type (line 25)
random_138647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), numpy_138646, 'random')
# Obtaining the member 'uniform' of a type (line 25)
uniform_138648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), random_138647, 'uniform')
# Calling uniform(args, kwargs) (line 25)
uniform_call_result_138655 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), uniform_138648, *[int_138649, int_138650, result_mul_138653], **kwargs_138654)

# Obtaining the member 'reshape' of a type (line 25)
reshape_138656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), uniform_call_result_138655, 'reshape')
# Calling reshape(args, kwargs) (line 25)
reshape_call_result_138660 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), reshape_138656, *[int_138657, int_138658], **kwargs_138659)

# Assigning a type to the variable 'yl' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'yl', reshape_call_result_138660)

# Assigning a BinOp to a Name (line 26):

# Assigning a BinOp to a Name (line 26):
# Getting the type of 'xl' (line 26)
xl_138661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'xl')
complex_138662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'complex')
# Getting the type of 'yl' (line 26)
yl_138663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'yl')
# Applying the binary operator '*' (line 26)
result_mul_138664 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 10), '*', complex_138662, yl_138663)

# Applying the binary operator '+' (line 26)
result_add_138665 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 5), '+', xl_138661, result_mul_138664)

# Assigning a type to the variable 'zl' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'zl', result_add_138665)

# Assigning a Compare to a Name (line 27):

# Assigning a Compare to a Name (line 27):

# Getting the type of 'xl' (line 27)
xl_138666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'xl')
float_138667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 13), 'float')
# Applying the binary operator '>' (line 27)
result_gt_138668 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 8), '>', xl_138666, float_138667)

# Assigning a type to the variable 'maskx' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'maskx', result_gt_138668)

# Assigning a Compare to a Name (line 28):

# Assigning a Compare to a Name (line 28):

# Getting the type of 'yl' (line 28)
yl_138669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'yl')
float_138670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 13), 'float')
# Applying the binary operator '<' (line 28)
result_lt_138671 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 8), '<', yl_138669, float_138670)

# Assigning a type to the variable 'masky' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'masky', result_lt_138671)

# Assigning a Call to a Name (line 29):

# Assigning a Call to a Name (line 29):

# Call to array(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'xl' (line 29)
xl_138675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'xl', False)
# Processing the call keyword arguments (line 29)
# Getting the type of 'maskx' (line 29)
maskx_138676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'maskx', False)
keyword_138677 = maskx_138676
kwargs_138678 = {'mask': keyword_138677}
# Getting the type of 'numpy' (line 29)
numpy_138672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 29)
ma_138673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 7), numpy_138672, 'ma')
# Obtaining the member 'array' of a type (line 29)
array_138674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 7), ma_138673, 'array')
# Calling array(args, kwargs) (line 29)
array_call_result_138679 = invoke(stypy.reporting.localization.Localization(__file__, 29, 7), array_138674, *[xl_138675], **kwargs_138678)

# Assigning a type to the variable 'nmxl' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'nmxl', array_call_result_138679)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to array(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'yl' (line 30)
yl_138683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'yl', False)
# Processing the call keyword arguments (line 30)
# Getting the type of 'masky' (line 30)
masky_138684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'masky', False)
keyword_138685 = masky_138684
kwargs_138686 = {'mask': keyword_138685}
# Getting the type of 'numpy' (line 30)
numpy_138680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 30)
ma_138681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), numpy_138680, 'ma')
# Obtaining the member 'array' of a type (line 30)
array_138682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), ma_138681, 'array')
# Calling array(args, kwargs) (line 30)
array_call_result_138687 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), array_138682, *[yl_138683], **kwargs_138686)

# Assigning a type to the variable 'nmyl' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'nmyl', array_call_result_138687)

# Assigning a Call to a Name (line 31):

# Assigning a Call to a Name (line 31):

# Call to array(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'zl' (line 31)
zl_138691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'zl', False)
# Processing the call keyword arguments (line 31)
# Getting the type of 'maskx' (line 31)
maskx_138692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'maskx', False)
keyword_138693 = maskx_138692
kwargs_138694 = {'mask': keyword_138693}
# Getting the type of 'numpy' (line 31)
numpy_138688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'numpy', False)
# Obtaining the member 'ma' of a type (line 31)
ma_138689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), numpy_138688, 'ma')
# Obtaining the member 'array' of a type (line 31)
array_138690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), ma_138689, 'array')
# Calling array(args, kwargs) (line 31)
array_call_result_138695 = invoke(stypy.reporting.localization.Localization(__file__, 31, 7), array_138690, *[zl_138691], **kwargs_138694)

# Assigning a type to the variable 'nmzl' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'nmzl', array_call_result_138695)

@norecursion
def timer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_138696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 15), 'str', '')
    int_138697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
    int_138698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 35), 'int')
    defaults = [str_138696, int_138697, int_138698]
    # Create a new context for function 'timer'
    module_type_store = module_type_store.open_function_context('timer', 39, 0, False)
    
    # Passed parameters checking function
    timer.stypy_localization = localization
    timer.stypy_type_of_self = None
    timer.stypy_type_store = module_type_store
    timer.stypy_function_name = 'timer'
    timer.stypy_param_names_list = ['s', 'v', 'nloop', 'nrep']
    timer.stypy_varargs_param_name = None
    timer.stypy_kwargs_param_name = None
    timer.stypy_call_defaults = defaults
    timer.stypy_call_varargs = varargs
    timer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'timer', ['s', 'v', 'nloop', 'nrep'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'timer', localization, ['s', 'v', 'nloop', 'nrep'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'timer(...)' code ##################

    
    # Assigning a List to a Name (line 40):
    
    # Assigning a List to a Name (line 40):
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_138699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    str_138700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'str', 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), list_138699, str_138700)
    # Adding element type (line 40)
    str_138701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'str', 'ms')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), list_138699, str_138701)
    # Adding element type (line 40)
    str_138702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'str', '\xc2\xb5s')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), list_138699, str_138702)
    # Adding element type (line 40)
    str_138703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'str', 'ns')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), list_138699, str_138703)
    
    # Assigning a type to the variable 'units' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'units', list_138699)
    
    # Assigning a List to a Name (line 41):
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_138704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_138705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_138704, int_138705)
    # Adding element type (line 41)
    float_138706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_138704, float_138706)
    # Adding element type (line 41)
    float_138707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_138704, float_138707)
    # Adding element type (line 41)
    float_138708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 28), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_138704, float_138708)
    
    # Assigning a type to the variable 'scaling' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'scaling', list_138704)
    
    # Call to print(...): (line 42)
    # Processing the call arguments (line 42)
    str_138710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 10), 'str', '%s : %-50s : ')
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_138711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'v' (line 42)
    v_138712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 29), tuple_138711, v_138712)
    # Adding element type (line 42)
    # Getting the type of 's' (line 42)
    s_138713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 29), tuple_138711, s_138713)
    
    # Applying the binary operator '%' (line 42)
    result_mod_138714 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 10), '%', str_138710, tuple_138711)
    
    # Processing the call keyword arguments (line 42)
    str_138715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'str', ' ')
    keyword_138716 = str_138715
    kwargs_138717 = {'end': keyword_138716}
    # Getting the type of 'print' (line 42)
    print_138709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'print', False)
    # Calling print(args, kwargs) (line 42)
    print_call_result_138718 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), print_138709, *[result_mod_138714], **kwargs_138717)
    
    
    # Assigning a ListComp to a Name (line 43):
    
    # Assigning a ListComp to a Name (line 43):
    # Calculating list comprehension
    # Calculating comprehension expression
    str_138727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 60), 'str', 'xyz')
    comprehension_138728 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 16), str_138727)
    # Assigning a type to the variable 'x' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'x', comprehension_138728)
    str_138719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'str', '%ss,nm%ss,%sl,nm%sl')
    
    # Call to tuple(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'x' (line 43)
    x_138721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'x', False)
    int_138722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 48), 'int')
    # Applying the binary operator '*' (line 43)
    result_mul_138723 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 46), '*', x_138721, int_138722)
    
    # Processing the call keyword arguments (line 43)
    kwargs_138724 = {}
    # Getting the type of 'tuple' (line 43)
    tuple_138720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 40), 'tuple', False)
    # Calling tuple(args, kwargs) (line 43)
    tuple_call_result_138725 = invoke(stypy.reporting.localization.Localization(__file__, 43, 40), tuple_138720, *[result_mul_138723], **kwargs_138724)
    
    # Applying the binary operator '%' (line 43)
    result_mod_138726 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 16), '%', str_138719, tuple_call_result_138725)
    
    list_138729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 16), list_138729, result_mod_138726)
    # Assigning a type to the variable 'varnames' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'varnames', list_138729)
    
    # Assigning a BinOp to a Name (line 44):
    
    # Assigning a BinOp to a Name (line 44):
    str_138730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'str', 'from __main__ import numpy, ma, %s')
    
    # Call to join(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'varnames' (line 44)
    varnames_138733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 60), 'varnames', False)
    # Processing the call keyword arguments (line 44)
    kwargs_138734 = {}
    str_138731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 51), 'str', ',')
    # Obtaining the member 'join' of a type (line 44)
    join_138732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 51), str_138731, 'join')
    # Calling join(args, kwargs) (line 44)
    join_call_result_138735 = invoke(stypy.reporting.localization.Localization(__file__, 44, 51), join_138732, *[varnames_138733], **kwargs_138734)
    
    # Applying the binary operator '%' (line 44)
    result_mod_138736 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '%', str_138730, join_call_result_138735)
    
    # Assigning a type to the variable 'setup' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'setup', result_mod_138736)
    
    # Assigning a Call to a Name (line 45):
    
    # Assigning a Call to a Name (line 45):
    
    # Call to Timer(...): (line 45)
    # Processing the call keyword arguments (line 45)
    # Getting the type of 's' (line 45)
    s_138739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 's', False)
    keyword_138740 = s_138739
    # Getting the type of 'setup' (line 45)
    setup_138741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'setup', False)
    keyword_138742 = setup_138741
    kwargs_138743 = {'setup': keyword_138742, 'stmt': keyword_138740}
    # Getting the type of 'timeit' (line 45)
    timeit_138737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'timeit', False)
    # Obtaining the member 'Timer' of a type (line 45)
    Timer_138738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), timeit_138737, 'Timer')
    # Calling Timer(args, kwargs) (line 45)
    Timer_call_result_138744 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), Timer_138738, *[], **kwargs_138743)
    
    # Assigning a type to the variable 'Timer' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'Timer', Timer_call_result_138744)
    
    # Assigning a BinOp to a Name (line 46):
    
    # Assigning a BinOp to a Name (line 46):
    
    # Call to min(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to repeat(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'nrep' (line 46)
    nrep_138748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'nrep', False)
    # Getting the type of 'nloop' (line 46)
    nloop_138749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'nloop', False)
    # Processing the call keyword arguments (line 46)
    kwargs_138750 = {}
    # Getting the type of 'Timer' (line 46)
    Timer_138746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'Timer', False)
    # Obtaining the member 'repeat' of a type (line 46)
    repeat_138747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), Timer_138746, 'repeat')
    # Calling repeat(args, kwargs) (line 46)
    repeat_call_result_138751 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), repeat_138747, *[nrep_138748, nloop_138749], **kwargs_138750)
    
    # Processing the call keyword arguments (line 46)
    kwargs_138752 = {}
    # Getting the type of 'min' (line 46)
    min_138745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'min', False)
    # Calling min(args, kwargs) (line 46)
    min_call_result_138753 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), min_138745, *[repeat_call_result_138751], **kwargs_138752)
    
    # Getting the type of 'nloop' (line 46)
    nloop_138754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'nloop')
    # Applying the binary operator 'div' (line 46)
    result_div_138755 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 11), 'div', min_call_result_138753, nloop_138754)
    
    # Assigning a type to the variable 'best' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'best', result_div_138755)
    
    
    # Getting the type of 'best' (line 47)
    best_138756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'best')
    float_138757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'float')
    # Applying the binary operator '>' (line 47)
    result_gt_138758 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), '>', best_138756, float_138757)
    
    # Testing the type of an if condition (line 47)
    if_condition_138759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_gt_138758)
    # Assigning a type to the variable 'if_condition_138759' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_138759', if_condition_138759)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to min(...): (line 48)
    # Processing the call arguments (line 48)
    
    
    # Call to int(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to floor(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to log10(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'best' (line 48)
    best_138766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'best', False)
    # Processing the call keyword arguments (line 48)
    kwargs_138767 = {}
    # Getting the type of 'numpy' (line 48)
    numpy_138764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'numpy', False)
    # Obtaining the member 'log10' of a type (line 48)
    log10_138765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 37), numpy_138764, 'log10')
    # Calling log10(args, kwargs) (line 48)
    log10_call_result_138768 = invoke(stypy.reporting.localization.Localization(__file__, 48, 37), log10_138765, *[best_138766], **kwargs_138767)
    
    # Processing the call keyword arguments (line 48)
    kwargs_138769 = {}
    # Getting the type of 'numpy' (line 48)
    numpy_138762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'numpy', False)
    # Obtaining the member 'floor' of a type (line 48)
    floor_138763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), numpy_138762, 'floor')
    # Calling floor(args, kwargs) (line 48)
    floor_call_result_138770 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), floor_138763, *[log10_call_result_138768], **kwargs_138769)
    
    int_138771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 59), 'int')
    # Applying the binary operator '//' (line 48)
    result_floordiv_138772 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 25), '//', floor_call_result_138770, int_138771)
    
    # Processing the call keyword arguments (line 48)
    kwargs_138773 = {}
    # Getting the type of 'int' (line 48)
    int_138761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'int', False)
    # Calling int(args, kwargs) (line 48)
    int_call_result_138774 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), int_138761, *[result_floordiv_138772], **kwargs_138773)
    
    # Applying the 'usub' unary operator (line 48)
    result___neg___138775 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 20), 'usub', int_call_result_138774)
    
    int_138776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 63), 'int')
    # Processing the call keyword arguments (line 48)
    kwargs_138777 = {}
    # Getting the type of 'min' (line 48)
    min_138760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'min', False)
    # Calling min(args, kwargs) (line 48)
    min_call_result_138778 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), min_138760, *[result___neg___138775, int_138776], **kwargs_138777)
    
    # Assigning a type to the variable 'order' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'order', min_call_result_138778)
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 50):
    
    # Assigning a Num to a Name (line 50):
    int_138779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
    # Assigning a type to the variable 'order' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'order', int_138779)
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 51)
    # Processing the call arguments (line 51)
    str_138781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 10), 'str', '%d loops, best of %d: %.*g %s per loop')
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_138782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    # Getting the type of 'nloop' (line 51)
    nloop_138783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 54), 'nloop', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 54), tuple_138782, nloop_138783)
    # Adding element type (line 51)
    # Getting the type of 'nrep' (line 51)
    nrep_138784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 61), 'nrep', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 54), tuple_138782, nrep_138784)
    # Adding element type (line 51)
    int_138785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 54), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 54), tuple_138782, int_138785)
    # Adding element type (line 51)
    # Getting the type of 'best' (line 53)
    best_138786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 54), 'best', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 53)
    order_138787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 69), 'order', False)
    # Getting the type of 'scaling' (line 53)
    scaling_138788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 61), 'scaling', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___138789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 61), scaling_138788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_138790 = invoke(stypy.reporting.localization.Localization(__file__, 53, 61), getitem___138789, order_138787)
    
    # Applying the binary operator '*' (line 53)
    result_mul_138791 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 54), '*', best_138786, subscript_call_result_138790)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 54), tuple_138782, result_mul_138791)
    # Adding element type (line 51)
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 54)
    order_138792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 60), 'order', False)
    # Getting the type of 'units' (line 54)
    units_138793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 54), 'units', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___138794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 54), units_138793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_138795 = invoke(stypy.reporting.localization.Localization(__file__, 54, 54), getitem___138794, order_138792)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 54), tuple_138782, subscript_call_result_138795)
    
    # Applying the binary operator '%' (line 51)
    result_mod_138796 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 10), '%', str_138781, tuple_138782)
    
    # Processing the call keyword arguments (line 51)
    kwargs_138797 = {}
    # Getting the type of 'print' (line 51)
    print_138780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'print', False)
    # Calling print(args, kwargs) (line 51)
    print_call_result_138798 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), print_138780, *[result_mod_138796], **kwargs_138797)
    
    
    # ################# End of 'timer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'timer' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_138799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'timer'
    return stypy_return_type_138799

# Assigning a type to the variable 'timer' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'timer', timer)

@norecursion
def compare_functions_1v(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_138800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 37), 'int')
    # Getting the type of 'xs' (line 58)
    xs_138801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'xs')
    # Getting the type of 'nmxs' (line 58)
    nmxs_138802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'nmxs')
    # Getting the type of 'xl' (line 58)
    xl_138803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 44), 'xl')
    # Getting the type of 'nmxl' (line 58)
    nmxl_138804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 53), 'nmxl')
    defaults = [int_138800, xs_138801, nmxs_138802, xl_138803, nmxl_138804]
    # Create a new context for function 'compare_functions_1v'
    module_type_store = module_type_store.open_function_context('compare_functions_1v', 57, 0, False)
    
    # Passed parameters checking function
    compare_functions_1v.stypy_localization = localization
    compare_functions_1v.stypy_type_of_self = None
    compare_functions_1v.stypy_type_store = module_type_store
    compare_functions_1v.stypy_function_name = 'compare_functions_1v'
    compare_functions_1v.stypy_param_names_list = ['func', 'nloop', 'xs', 'nmxs', 'xl', 'nmxl']
    compare_functions_1v.stypy_varargs_param_name = None
    compare_functions_1v.stypy_kwargs_param_name = None
    compare_functions_1v.stypy_call_defaults = defaults
    compare_functions_1v.stypy_call_varargs = varargs
    compare_functions_1v.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_functions_1v', ['func', 'nloop', 'xs', 'nmxs', 'xl', 'nmxl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_functions_1v', localization, ['func', 'nloop', 'xs', 'nmxs', 'xl', 'nmxl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_functions_1v(...)' code ##################

    
    # Assigning a Attribute to a Name (line 59):
    
    # Assigning a Attribute to a Name (line 59):
    # Getting the type of 'func' (line 59)
    func_138805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'func')
    # Obtaining the member '__name__' of a type (line 59)
    name___138806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), func_138805, '__name__')
    # Assigning a type to the variable 'funcname' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'funcname', name___138806)
    
    # Call to print(...): (line 60)
    # Processing the call arguments (line 60)
    str_138808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 10), 'str', '-')
    int_138809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 14), 'int')
    # Applying the binary operator '*' (line 60)
    result_mul_138810 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 10), '*', str_138808, int_138809)
    
    # Processing the call keyword arguments (line 60)
    kwargs_138811 = {}
    # Getting the type of 'print' (line 60)
    print_138807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'print', False)
    # Calling print(args, kwargs) (line 60)
    print_call_result_138812 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), print_138807, *[result_mul_138810], **kwargs_138811)
    
    
    # Call to print(...): (line 61)
    # Processing the call arguments (line 61)
    str_138814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 10), 'str', '%s on small arrays')
    # Getting the type of 'funcname' (line 61)
    funcname_138815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'funcname', False)
    # Applying the binary operator '%' (line 61)
    result_mod_138816 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 10), '%', str_138814, funcname_138815)
    
    # Processing the call keyword arguments (line 61)
    kwargs_138817 = {}
    # Getting the type of 'print' (line 61)
    print_138813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'print', False)
    # Calling print(args, kwargs) (line 61)
    print_call_result_138818 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), print_138813, *[result_mod_138816], **kwargs_138817)
    
    
    # Assigning a Tuple to a Tuple (line 62):
    
    # Assigning a Str to a Name (line 62):
    str_138819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138544' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_assignment_138544', str_138819)
    
    # Assigning a Str to a Name (line 62):
    str_138820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 31), 'str', 'nmxs')
    # Assigning a type to the variable 'tuple_assignment_138545' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_assignment_138545', str_138820)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'tuple_assignment_138544' (line 62)
    tuple_assignment_138544_138821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_assignment_138544')
    # Assigning a type to the variable 'module' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'module', tuple_assignment_138544_138821)
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'tuple_assignment_138545' (line 62)
    tuple_assignment_138545_138822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'tuple_assignment_138545')
    # Assigning a type to the variable 'data' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'data', tuple_assignment_138545_138822)
    
    # Call to timer(...): (line 63)
    # Processing the call arguments (line 63)
    str_138824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 10), 'str', '%(module)s.%(funcname)s(%(data)s)')
    
    # Call to locals(...): (line 63)
    # Processing the call keyword arguments (line 63)
    kwargs_138826 = {}
    # Getting the type of 'locals' (line 63)
    locals_138825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 63)
    locals_call_result_138827 = invoke(stypy.reporting.localization.Localization(__file__, 63, 48), locals_138825, *[], **kwargs_138826)
    
    # Applying the binary operator '%' (line 63)
    result_mod_138828 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 10), '%', str_138824, locals_call_result_138827)
    
    # Processing the call keyword arguments (line 63)
    str_138829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 60), 'str', '%11s')
    # Getting the type of 'module' (line 63)
    module_138830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 69), 'module', False)
    # Applying the binary operator '%' (line 63)
    result_mod_138831 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 60), '%', str_138829, module_138830)
    
    keyword_138832 = result_mod_138831
    # Getting the type of 'nloop' (line 63)
    nloop_138833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 83), 'nloop', False)
    keyword_138834 = nloop_138833
    kwargs_138835 = {'nloop': keyword_138834, 'v': keyword_138832}
    # Getting the type of 'timer' (line 63)
    timer_138823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 63)
    timer_call_result_138836 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), timer_138823, *[result_mod_138828], **kwargs_138835)
    
    
    # Call to print(...): (line 65)
    # Processing the call arguments (line 65)
    str_138838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 10), 'str', '%s on large arrays')
    # Getting the type of 'funcname' (line 65)
    funcname_138839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'funcname', False)
    # Applying the binary operator '%' (line 65)
    result_mod_138840 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 10), '%', str_138838, funcname_138839)
    
    # Processing the call keyword arguments (line 65)
    kwargs_138841 = {}
    # Getting the type of 'print' (line 65)
    print_138837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'print', False)
    # Calling print(args, kwargs) (line 65)
    print_call_result_138842 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), print_138837, *[result_mod_138840], **kwargs_138841)
    
    
    # Assigning a Tuple to a Tuple (line 66):
    
    # Assigning a Str to a Name (line 66):
    str_138843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138546' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_assignment_138546', str_138843)
    
    # Assigning a Str to a Name (line 66):
    str_138844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'str', 'nmxl')
    # Assigning a type to the variable 'tuple_assignment_138547' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_assignment_138547', str_138844)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_assignment_138546' (line 66)
    tuple_assignment_138546_138845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_assignment_138546')
    # Assigning a type to the variable 'module' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'module', tuple_assignment_138546_138845)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_assignment_138547' (line 66)
    tuple_assignment_138547_138846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_assignment_138547')
    # Assigning a type to the variable 'data' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'data', tuple_assignment_138547_138846)
    
    # Call to timer(...): (line 67)
    # Processing the call arguments (line 67)
    str_138848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'str', '%(module)s.%(funcname)s(%(data)s)')
    
    # Call to locals(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_138850 = {}
    # Getting the type of 'locals' (line 67)
    locals_138849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 67)
    locals_call_result_138851 = invoke(stypy.reporting.localization.Localization(__file__, 67, 48), locals_138849, *[], **kwargs_138850)
    
    # Applying the binary operator '%' (line 67)
    result_mod_138852 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 10), '%', str_138848, locals_call_result_138851)
    
    # Processing the call keyword arguments (line 67)
    str_138853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 60), 'str', '%11s')
    # Getting the type of 'module' (line 67)
    module_138854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 69), 'module', False)
    # Applying the binary operator '%' (line 67)
    result_mod_138855 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 60), '%', str_138853, module_138854)
    
    keyword_138856 = result_mod_138855
    # Getting the type of 'nloop' (line 67)
    nloop_138857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 83), 'nloop', False)
    keyword_138858 = nloop_138857
    kwargs_138859 = {'nloop': keyword_138858, 'v': keyword_138856}
    # Getting the type of 'timer' (line 67)
    timer_138847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 67)
    timer_call_result_138860 = invoke(stypy.reporting.localization.Localization(__file__, 67, 4), timer_138847, *[result_mod_138852], **kwargs_138859)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'compare_functions_1v(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_functions_1v' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_138861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138861)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_functions_1v'
    return stypy_return_type_138861

# Assigning a type to the variable 'compare_functions_1v' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'compare_functions_1v', compare_functions_1v)

@norecursion
def compare_methods(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_138862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 43), 'str', 'x')
    int_138863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 54), 'int')
    # Getting the type of 'True' (line 70)
    True_138864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 64), 'True')
    # Getting the type of 'xs' (line 71)
    xs_138865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'xs')
    # Getting the type of 'nmxs' (line 71)
    nmxs_138866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'nmxs')
    # Getting the type of 'xl' (line 71)
    xl_138867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 41), 'xl')
    # Getting the type of 'nmxl' (line 71)
    nmxl_138868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 50), 'nmxl')
    defaults = [str_138862, int_138863, True_138864, xs_138865, nmxs_138866, xl_138867, nmxl_138868]
    # Create a new context for function 'compare_methods'
    module_type_store = module_type_store.open_function_context('compare_methods', 70, 0, False)
    
    # Passed parameters checking function
    compare_methods.stypy_localization = localization
    compare_methods.stypy_type_of_self = None
    compare_methods.stypy_type_store = module_type_store
    compare_methods.stypy_function_name = 'compare_methods'
    compare_methods.stypy_param_names_list = ['methodname', 'args', 'vars', 'nloop', 'test', 'xs', 'nmxs', 'xl', 'nmxl']
    compare_methods.stypy_varargs_param_name = None
    compare_methods.stypy_kwargs_param_name = None
    compare_methods.stypy_call_defaults = defaults
    compare_methods.stypy_call_varargs = varargs
    compare_methods.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_methods', ['methodname', 'args', 'vars', 'nloop', 'test', 'xs', 'nmxs', 'xl', 'nmxl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_methods', localization, ['methodname', 'args', 'vars', 'nloop', 'test', 'xs', 'nmxs', 'xl', 'nmxl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_methods(...)' code ##################

    
    # Call to print(...): (line 72)
    # Processing the call arguments (line 72)
    str_138870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 10), 'str', '-')
    int_138871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'int')
    # Applying the binary operator '*' (line 72)
    result_mul_138872 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 10), '*', str_138870, int_138871)
    
    # Processing the call keyword arguments (line 72)
    kwargs_138873 = {}
    # Getting the type of 'print' (line 72)
    print_138869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'print', False)
    # Calling print(args, kwargs) (line 72)
    print_call_result_138874 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), print_138869, *[result_mul_138872], **kwargs_138873)
    
    
    # Call to print(...): (line 73)
    # Processing the call arguments (line 73)
    str_138876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 10), 'str', '%s on small arrays')
    # Getting the type of 'methodname' (line 73)
    methodname_138877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'methodname', False)
    # Applying the binary operator '%' (line 73)
    result_mod_138878 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 10), '%', str_138876, methodname_138877)
    
    # Processing the call keyword arguments (line 73)
    kwargs_138879 = {}
    # Getting the type of 'print' (line 73)
    print_138875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'print', False)
    # Calling print(args, kwargs) (line 73)
    print_call_result_138880 = invoke(stypy.reporting.localization.Localization(__file__, 73, 4), print_138875, *[result_mod_138878], **kwargs_138879)
    
    
    # Assigning a Tuple to a Tuple (line 74):
    
    # Assigning a BinOp to a Name (line 74):
    str_138881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'str', 'nm%ss')
    # Getting the type of 'vars' (line 74)
    vars_138882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'vars')
    # Applying the binary operator '%' (line 74)
    result_mod_138883 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 16), '%', str_138881, vars_138882)
    
    # Assigning a type to the variable 'tuple_assignment_138548' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_assignment_138548', result_mod_138883)
    
    # Assigning a Str to a Name (line 74):
    str_138884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 32), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138549' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_assignment_138549', str_138884)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_assignment_138548' (line 74)
    tuple_assignment_138548_138885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_assignment_138548')
    # Assigning a type to the variable 'data' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'data', tuple_assignment_138548_138885)
    
    # Assigning a Name to a Name (line 74):
    # Getting the type of 'tuple_assignment_138549' (line 74)
    tuple_assignment_138549_138886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'tuple_assignment_138549')
    # Assigning a type to the variable 'ver' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 10), 'ver', tuple_assignment_138549_138886)
    
    # Call to timer(...): (line 75)
    # Processing the call arguments (line 75)
    str_138888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'str', '%(data)s.%(methodname)s(%(args)s)')
    
    # Call to locals(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_138890 = {}
    # Getting the type of 'locals' (line 75)
    locals_138889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 75)
    locals_call_result_138891 = invoke(stypy.reporting.localization.Localization(__file__, 75, 48), locals_138889, *[], **kwargs_138890)
    
    # Applying the binary operator '%' (line 75)
    result_mod_138892 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 10), '%', str_138888, locals_call_result_138891)
    
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'ver' (line 75)
    ver_138893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 60), 'ver', False)
    keyword_138894 = ver_138893
    # Getting the type of 'nloop' (line 75)
    nloop_138895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 71), 'nloop', False)
    keyword_138896 = nloop_138895
    kwargs_138897 = {'nloop': keyword_138896, 'v': keyword_138894}
    # Getting the type of 'timer' (line 75)
    timer_138887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 75)
    timer_call_result_138898 = invoke(stypy.reporting.localization.Localization(__file__, 75, 4), timer_138887, *[result_mod_138892], **kwargs_138897)
    
    
    # Call to print(...): (line 77)
    # Processing the call arguments (line 77)
    str_138900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 10), 'str', '%s on large arrays')
    # Getting the type of 'methodname' (line 77)
    methodname_138901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'methodname', False)
    # Applying the binary operator '%' (line 77)
    result_mod_138902 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 10), '%', str_138900, methodname_138901)
    
    # Processing the call keyword arguments (line 77)
    kwargs_138903 = {}
    # Getting the type of 'print' (line 77)
    print_138899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'print', False)
    # Calling print(args, kwargs) (line 77)
    print_call_result_138904 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), print_138899, *[result_mod_138902], **kwargs_138903)
    
    
    # Assigning a Tuple to a Tuple (line 78):
    
    # Assigning a BinOp to a Name (line 78):
    str_138905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'str', 'nm%sl')
    # Getting the type of 'vars' (line 78)
    vars_138906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'vars')
    # Applying the binary operator '%' (line 78)
    result_mod_138907 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '%', str_138905, vars_138906)
    
    # Assigning a type to the variable 'tuple_assignment_138550' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_138550', result_mod_138907)
    
    # Assigning a Str to a Name (line 78):
    str_138908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138551' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_138551', str_138908)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_assignment_138550' (line 78)
    tuple_assignment_138550_138909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_138550')
    # Assigning a type to the variable 'data' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'data', tuple_assignment_138550_138909)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_assignment_138551' (line 78)
    tuple_assignment_138551_138910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_138551')
    # Assigning a type to the variable 'ver' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 10), 'ver', tuple_assignment_138551_138910)
    
    # Call to timer(...): (line 79)
    # Processing the call arguments (line 79)
    str_138912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 10), 'str', '%(data)s.%(methodname)s(%(args)s)')
    
    # Call to locals(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_138914 = {}
    # Getting the type of 'locals' (line 79)
    locals_138913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 79)
    locals_call_result_138915 = invoke(stypy.reporting.localization.Localization(__file__, 79, 48), locals_138913, *[], **kwargs_138914)
    
    # Applying the binary operator '%' (line 79)
    result_mod_138916 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 10), '%', str_138912, locals_call_result_138915)
    
    # Processing the call keyword arguments (line 79)
    # Getting the type of 'ver' (line 79)
    ver_138917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 60), 'ver', False)
    keyword_138918 = ver_138917
    # Getting the type of 'nloop' (line 79)
    nloop_138919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 71), 'nloop', False)
    keyword_138920 = nloop_138919
    kwargs_138921 = {'nloop': keyword_138920, 'v': keyword_138918}
    # Getting the type of 'timer' (line 79)
    timer_138911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 79)
    timer_call_result_138922 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), timer_138911, *[result_mod_138916], **kwargs_138921)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'compare_methods(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_methods' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_138923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_methods'
    return stypy_return_type_138923

# Assigning a type to the variable 'compare_methods' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'compare_methods', compare_methods)

@norecursion
def compare_functions_2v(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_138924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 37), 'int')
    # Getting the type of 'True' (line 82)
    True_138925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'True')
    # Getting the type of 'xs' (line 83)
    xs_138926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'xs')
    # Getting the type of 'nmxs' (line 83)
    nmxs_138927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 35), 'nmxs')
    # Getting the type of 'ys' (line 84)
    ys_138928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'ys')
    # Getting the type of 'nmys' (line 84)
    nmys_138929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'nmys')
    # Getting the type of 'xl' (line 85)
    xl_138930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'xl')
    # Getting the type of 'nmxl' (line 85)
    nmxl_138931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 35), 'nmxl')
    # Getting the type of 'yl' (line 86)
    yl_138932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'yl')
    # Getting the type of 'nmyl' (line 86)
    nmyl_138933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'nmyl')
    defaults = [int_138924, True_138925, xs_138926, nmxs_138927, ys_138928, nmys_138929, xl_138930, nmxl_138931, yl_138932, nmyl_138933]
    # Create a new context for function 'compare_functions_2v'
    module_type_store = module_type_store.open_function_context('compare_functions_2v', 82, 0, False)
    
    # Passed parameters checking function
    compare_functions_2v.stypy_localization = localization
    compare_functions_2v.stypy_type_of_self = None
    compare_functions_2v.stypy_type_store = module_type_store
    compare_functions_2v.stypy_function_name = 'compare_functions_2v'
    compare_functions_2v.stypy_param_names_list = ['func', 'nloop', 'test', 'xs', 'nmxs', 'ys', 'nmys', 'xl', 'nmxl', 'yl', 'nmyl']
    compare_functions_2v.stypy_varargs_param_name = None
    compare_functions_2v.stypy_kwargs_param_name = None
    compare_functions_2v.stypy_call_defaults = defaults
    compare_functions_2v.stypy_call_varargs = varargs
    compare_functions_2v.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compare_functions_2v', ['func', 'nloop', 'test', 'xs', 'nmxs', 'ys', 'nmys', 'xl', 'nmxl', 'yl', 'nmyl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compare_functions_2v', localization, ['func', 'nloop', 'test', 'xs', 'nmxs', 'ys', 'nmys', 'xl', 'nmxl', 'yl', 'nmyl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compare_functions_2v(...)' code ##################

    
    # Assigning a Attribute to a Name (line 87):
    
    # Assigning a Attribute to a Name (line 87):
    # Getting the type of 'func' (line 87)
    func_138934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'func')
    # Obtaining the member '__name__' of a type (line 87)
    name___138935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), func_138934, '__name__')
    # Assigning a type to the variable 'funcname' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'funcname', name___138935)
    
    # Call to print(...): (line 88)
    # Processing the call arguments (line 88)
    str_138937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 10), 'str', '-')
    int_138938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 14), 'int')
    # Applying the binary operator '*' (line 88)
    result_mul_138939 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 10), '*', str_138937, int_138938)
    
    # Processing the call keyword arguments (line 88)
    kwargs_138940 = {}
    # Getting the type of 'print' (line 88)
    print_138936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'print', False)
    # Calling print(args, kwargs) (line 88)
    print_call_result_138941 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), print_138936, *[result_mul_138939], **kwargs_138940)
    
    
    # Call to print(...): (line 89)
    # Processing the call arguments (line 89)
    str_138943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 10), 'str', '%s on small arrays')
    # Getting the type of 'funcname' (line 89)
    funcname_138944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'funcname', False)
    # Applying the binary operator '%' (line 89)
    result_mod_138945 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 10), '%', str_138943, funcname_138944)
    
    # Processing the call keyword arguments (line 89)
    kwargs_138946 = {}
    # Getting the type of 'print' (line 89)
    print_138942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'print', False)
    # Calling print(args, kwargs) (line 89)
    print_call_result_138947 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), print_138942, *[result_mod_138945], **kwargs_138946)
    
    
    # Assigning a Tuple to a Tuple (line 90):
    
    # Assigning a Str to a Name (line 90):
    str_138948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138552' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_138552', str_138948)
    
    # Assigning a Str to a Name (line 90):
    str_138949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 31), 'str', 'nmxs,nmys')
    # Assigning a type to the variable 'tuple_assignment_138553' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_138553', str_138949)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_assignment_138552' (line 90)
    tuple_assignment_138552_138950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_138552')
    # Assigning a type to the variable 'module' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'module', tuple_assignment_138552_138950)
    
    # Assigning a Name to a Name (line 90):
    # Getting the type of 'tuple_assignment_138553' (line 90)
    tuple_assignment_138553_138951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tuple_assignment_138553')
    # Assigning a type to the variable 'data' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'data', tuple_assignment_138553_138951)
    
    # Call to timer(...): (line 91)
    # Processing the call arguments (line 91)
    str_138953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 10), 'str', '%(module)s.%(funcname)s(%(data)s)')
    
    # Call to locals(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_138955 = {}
    # Getting the type of 'locals' (line 91)
    locals_138954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 91)
    locals_call_result_138956 = invoke(stypy.reporting.localization.Localization(__file__, 91, 48), locals_138954, *[], **kwargs_138955)
    
    # Applying the binary operator '%' (line 91)
    result_mod_138957 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 10), '%', str_138953, locals_call_result_138956)
    
    # Processing the call keyword arguments (line 91)
    str_138958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 60), 'str', '%11s')
    # Getting the type of 'module' (line 91)
    module_138959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 69), 'module', False)
    # Applying the binary operator '%' (line 91)
    result_mod_138960 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 60), '%', str_138958, module_138959)
    
    keyword_138961 = result_mod_138960
    # Getting the type of 'nloop' (line 91)
    nloop_138962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 83), 'nloop', False)
    keyword_138963 = nloop_138962
    kwargs_138964 = {'nloop': keyword_138963, 'v': keyword_138961}
    # Getting the type of 'timer' (line 91)
    timer_138952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 91)
    timer_call_result_138965 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), timer_138952, *[result_mod_138957], **kwargs_138964)
    
    
    # Call to print(...): (line 93)
    # Processing the call arguments (line 93)
    str_138967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 10), 'str', '%s on large arrays')
    # Getting the type of 'funcname' (line 93)
    funcname_138968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'funcname', False)
    # Applying the binary operator '%' (line 93)
    result_mod_138969 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 10), '%', str_138967, funcname_138968)
    
    # Processing the call keyword arguments (line 93)
    kwargs_138970 = {}
    # Getting the type of 'print' (line 93)
    print_138966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'print', False)
    # Calling print(args, kwargs) (line 93)
    print_call_result_138971 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), print_138966, *[result_mod_138969], **kwargs_138970)
    
    
    # Assigning a Tuple to a Tuple (line 94):
    
    # Assigning a Str to a Name (line 94):
    str_138972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'str', 'numpy.ma')
    # Assigning a type to the variable 'tuple_assignment_138554' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_assignment_138554', str_138972)
    
    # Assigning a Str to a Name (line 94):
    str_138973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'str', 'nmxl,nmyl')
    # Assigning a type to the variable 'tuple_assignment_138555' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_assignment_138555', str_138973)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_assignment_138554' (line 94)
    tuple_assignment_138554_138974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_assignment_138554')
    # Assigning a type to the variable 'module' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'module', tuple_assignment_138554_138974)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'tuple_assignment_138555' (line 94)
    tuple_assignment_138555_138975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tuple_assignment_138555')
    # Assigning a type to the variable 'data' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'data', tuple_assignment_138555_138975)
    
    # Call to timer(...): (line 95)
    # Processing the call arguments (line 95)
    str_138977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 10), 'str', '%(module)s.%(funcname)s(%(data)s)')
    
    # Call to locals(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_138979 = {}
    # Getting the type of 'locals' (line 95)
    locals_138978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 48), 'locals', False)
    # Calling locals(args, kwargs) (line 95)
    locals_call_result_138980 = invoke(stypy.reporting.localization.Localization(__file__, 95, 48), locals_138978, *[], **kwargs_138979)
    
    # Applying the binary operator '%' (line 95)
    result_mod_138981 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 10), '%', str_138977, locals_call_result_138980)
    
    # Processing the call keyword arguments (line 95)
    str_138982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 60), 'str', '%11s')
    # Getting the type of 'module' (line 95)
    module_138983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 69), 'module', False)
    # Applying the binary operator '%' (line 95)
    result_mod_138984 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 60), '%', str_138982, module_138983)
    
    keyword_138985 = result_mod_138984
    # Getting the type of 'nloop' (line 95)
    nloop_138986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 83), 'nloop', False)
    keyword_138987 = nloop_138986
    kwargs_138988 = {'nloop': keyword_138987, 'v': keyword_138985}
    # Getting the type of 'timer' (line 95)
    timer_138976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 95)
    timer_call_result_138989 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), timer_138976, *[result_mod_138981], **kwargs_138988)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'compare_functions_2v(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compare_functions_2v' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_138990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_138990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compare_functions_2v'
    return stypy_return_type_138990

# Assigning a type to the variable 'compare_functions_2v' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'compare_functions_2v', compare_functions_2v)

if (__name__ == '__main__'):
    
    # Call to compare_functions_1v(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'numpy' (line 100)
    numpy_138992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'numpy', False)
    # Obtaining the member 'sin' of a type (line 100)
    sin_138993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), numpy_138992, 'sin')
    # Processing the call keyword arguments (line 100)
    kwargs_138994 = {}
    # Getting the type of 'compare_functions_1v' (line 100)
    compare_functions_1v_138991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'compare_functions_1v', False)
    # Calling compare_functions_1v(args, kwargs) (line 100)
    compare_functions_1v_call_result_138995 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), compare_functions_1v_138991, *[sin_138993], **kwargs_138994)
    
    
    # Call to compare_functions_1v(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'numpy' (line 101)
    numpy_138997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 'numpy', False)
    # Obtaining the member 'log' of a type (line 101)
    log_138998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 25), numpy_138997, 'log')
    # Processing the call keyword arguments (line 101)
    kwargs_138999 = {}
    # Getting the type of 'compare_functions_1v' (line 101)
    compare_functions_1v_138996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'compare_functions_1v', False)
    # Calling compare_functions_1v(args, kwargs) (line 101)
    compare_functions_1v_call_result_139000 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), compare_functions_1v_138996, *[log_138998], **kwargs_138999)
    
    
    # Call to compare_functions_1v(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'numpy' (line 102)
    numpy_139002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 25), 'numpy', False)
    # Obtaining the member 'sqrt' of a type (line 102)
    sqrt_139003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 25), numpy_139002, 'sqrt')
    # Processing the call keyword arguments (line 102)
    kwargs_139004 = {}
    # Getting the type of 'compare_functions_1v' (line 102)
    compare_functions_1v_139001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'compare_functions_1v', False)
    # Calling compare_functions_1v(args, kwargs) (line 102)
    compare_functions_1v_call_result_139005 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), compare_functions_1v_139001, *[sqrt_139003], **kwargs_139004)
    
    
    # Call to compare_functions_2v(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'numpy' (line 104)
    numpy_139007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'numpy', False)
    # Obtaining the member 'multiply' of a type (line 104)
    multiply_139008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 25), numpy_139007, 'multiply')
    # Processing the call keyword arguments (line 104)
    kwargs_139009 = {}
    # Getting the type of 'compare_functions_2v' (line 104)
    compare_functions_2v_139006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'compare_functions_2v', False)
    # Calling compare_functions_2v(args, kwargs) (line 104)
    compare_functions_2v_call_result_139010 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), compare_functions_2v_139006, *[multiply_139008], **kwargs_139009)
    
    
    # Call to compare_functions_2v(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'numpy' (line 105)
    numpy_139012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'numpy', False)
    # Obtaining the member 'divide' of a type (line 105)
    divide_139013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), numpy_139012, 'divide')
    # Processing the call keyword arguments (line 105)
    kwargs_139014 = {}
    # Getting the type of 'compare_functions_2v' (line 105)
    compare_functions_2v_139011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'compare_functions_2v', False)
    # Calling compare_functions_2v(args, kwargs) (line 105)
    compare_functions_2v_call_result_139015 = invoke(stypy.reporting.localization.Localization(__file__, 105, 4), compare_functions_2v_139011, *[divide_139013], **kwargs_139014)
    
    
    # Call to compare_functions_2v(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'numpy' (line 106)
    numpy_139017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'numpy', False)
    # Obtaining the member 'power' of a type (line 106)
    power_139018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), numpy_139017, 'power')
    # Processing the call keyword arguments (line 106)
    kwargs_139019 = {}
    # Getting the type of 'compare_functions_2v' (line 106)
    compare_functions_2v_139016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'compare_functions_2v', False)
    # Calling compare_functions_2v(args, kwargs) (line 106)
    compare_functions_2v_call_result_139020 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), compare_functions_2v_139016, *[power_139018], **kwargs_139019)
    
    
    # Call to compare_methods(...): (line 108)
    # Processing the call arguments (line 108)
    str_139022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'str', 'ravel')
    str_139023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'str', '')
    # Processing the call keyword arguments (line 108)
    int_139024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'int')
    keyword_139025 = int_139024
    kwargs_139026 = {'nloop': keyword_139025}
    # Getting the type of 'compare_methods' (line 108)
    compare_methods_139021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 108)
    compare_methods_call_result_139027 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), compare_methods_139021, *[str_139022, str_139023], **kwargs_139026)
    
    
    # Call to compare_methods(...): (line 109)
    # Processing the call arguments (line 109)
    str_139029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'str', 'conjugate')
    str_139030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'str', '')
    str_139031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 37), 'str', 'z')
    # Processing the call keyword arguments (line 109)
    int_139032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 48), 'int')
    keyword_139033 = int_139032
    kwargs_139034 = {'nloop': keyword_139033}
    # Getting the type of 'compare_methods' (line 109)
    compare_methods_139028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 109)
    compare_methods_call_result_139035 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), compare_methods_139028, *[str_139029, str_139030, str_139031], **kwargs_139034)
    
    
    # Call to compare_methods(...): (line 110)
    # Processing the call arguments (line 110)
    str_139037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 20), 'str', 'transpose')
    str_139038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 33), 'str', '')
    # Processing the call keyword arguments (line 110)
    int_139039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 43), 'int')
    keyword_139040 = int_139039
    kwargs_139041 = {'nloop': keyword_139040}
    # Getting the type of 'compare_methods' (line 110)
    compare_methods_139036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 110)
    compare_methods_call_result_139042 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), compare_methods_139036, *[str_139037, str_139038], **kwargs_139041)
    
    
    # Call to compare_methods(...): (line 111)
    # Processing the call arguments (line 111)
    str_139044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 20), 'str', 'compressed')
    str_139045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 34), 'str', '')
    # Processing the call keyword arguments (line 111)
    int_139046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 44), 'int')
    keyword_139047 = int_139046
    kwargs_139048 = {'nloop': keyword_139047}
    # Getting the type of 'compare_methods' (line 111)
    compare_methods_139043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 111)
    compare_methods_call_result_139049 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), compare_methods_139043, *[str_139044, str_139045], **kwargs_139048)
    
    
    # Call to compare_methods(...): (line 112)
    # Processing the call arguments (line 112)
    str_139051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'str', '__getitem__')
    str_139052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'str', '0')
    # Processing the call keyword arguments (line 112)
    int_139053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'int')
    keyword_139054 = int_139053
    kwargs_139055 = {'nloop': keyword_139054}
    # Getting the type of 'compare_methods' (line 112)
    compare_methods_139050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 112)
    compare_methods_call_result_139056 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), compare_methods_139050, *[str_139051, str_139052], **kwargs_139055)
    
    
    # Call to compare_methods(...): (line 113)
    # Processing the call arguments (line 113)
    str_139058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 20), 'str', '__getitem__')
    str_139059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 35), 'str', '(0,0)')
    # Processing the call keyword arguments (line 113)
    int_139060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'int')
    keyword_139061 = int_139060
    kwargs_139062 = {'nloop': keyword_139061}
    # Getting the type of 'compare_methods' (line 113)
    compare_methods_139057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 113)
    compare_methods_call_result_139063 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), compare_methods_139057, *[str_139058, str_139059], **kwargs_139062)
    
    
    # Call to compare_methods(...): (line 114)
    # Processing the call arguments (line 114)
    str_139065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'str', '__getitem__')
    str_139066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'str', '[0,-1]')
    # Processing the call keyword arguments (line 114)
    int_139067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 51), 'int')
    keyword_139068 = int_139067
    kwargs_139069 = {'nloop': keyword_139068}
    # Getting the type of 'compare_methods' (line 114)
    compare_methods_139064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 114)
    compare_methods_call_result_139070 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), compare_methods_139064, *[str_139065, str_139066], **kwargs_139069)
    
    
    # Call to compare_methods(...): (line 115)
    # Processing the call arguments (line 115)
    str_139072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 20), 'str', '__setitem__')
    str_139073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'str', '0, 17')
    # Processing the call keyword arguments (line 115)
    int_139074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 50), 'int')
    keyword_139075 = int_139074
    # Getting the type of 'False' (line 115)
    False_139076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 61), 'False', False)
    keyword_139077 = False_139076
    kwargs_139078 = {'test': keyword_139077, 'nloop': keyword_139075}
    # Getting the type of 'compare_methods' (line 115)
    compare_methods_139071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 115)
    compare_methods_call_result_139079 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), compare_methods_139071, *[str_139072, str_139073], **kwargs_139078)
    
    
    # Call to compare_methods(...): (line 116)
    # Processing the call arguments (line 116)
    str_139081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'str', '__setitem__')
    str_139082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 35), 'str', '(0,0), 17')
    # Processing the call keyword arguments (line 116)
    int_139083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 54), 'int')
    keyword_139084 = int_139083
    # Getting the type of 'False' (line 116)
    False_139085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'False', False)
    keyword_139086 = False_139085
    kwargs_139087 = {'test': keyword_139086, 'nloop': keyword_139084}
    # Getting the type of 'compare_methods' (line 116)
    compare_methods_139080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'compare_methods', False)
    # Calling compare_methods(args, kwargs) (line 116)
    compare_methods_call_result_139088 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), compare_methods_139080, *[str_139081, str_139082], **kwargs_139087)
    
    
    # Call to print(...): (line 118)
    # Processing the call arguments (line 118)
    str_139090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'str', '-')
    int_139091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'int')
    # Applying the binary operator '*' (line 118)
    result_mul_139092 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 10), '*', str_139090, int_139091)
    
    # Processing the call keyword arguments (line 118)
    kwargs_139093 = {}
    # Getting the type of 'print' (line 118)
    print_139089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'print', False)
    # Calling print(args, kwargs) (line 118)
    print_call_result_139094 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), print_139089, *[result_mul_139092], **kwargs_139093)
    
    
    # Call to print(...): (line 119)
    # Processing the call arguments (line 119)
    str_139096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 10), 'str', '__setitem__ on small arrays')
    # Processing the call keyword arguments (line 119)
    kwargs_139097 = {}
    # Getting the type of 'print' (line 119)
    print_139095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'print', False)
    # Calling print(args, kwargs) (line 119)
    print_call_result_139098 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), print_139095, *[str_139096], **kwargs_139097)
    
    
    # Call to timer(...): (line 120)
    # Processing the call arguments (line 120)
    str_139100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'str', 'nmxs.__setitem__((-1,0),numpy.ma.masked)')
    str_139101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 54), 'str', 'numpy.ma   ')
    # Processing the call keyword arguments (line 120)
    int_139102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 75), 'int')
    keyword_139103 = int_139102
    kwargs_139104 = {'nloop': keyword_139103}
    # Getting the type of 'timer' (line 120)
    timer_139099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 120)
    timer_call_result_139105 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), timer_139099, *[str_139100, str_139101], **kwargs_139104)
    
    
    # Call to print(...): (line 122)
    # Processing the call arguments (line 122)
    str_139107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 10), 'str', '-')
    int_139108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 14), 'int')
    # Applying the binary operator '*' (line 122)
    result_mul_139109 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 10), '*', str_139107, int_139108)
    
    # Processing the call keyword arguments (line 122)
    kwargs_139110 = {}
    # Getting the type of 'print' (line 122)
    print_139106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'print', False)
    # Calling print(args, kwargs) (line 122)
    print_call_result_139111 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), print_139106, *[result_mul_139109], **kwargs_139110)
    
    
    # Call to print(...): (line 123)
    # Processing the call arguments (line 123)
    str_139113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'str', '__setitem__ on large arrays')
    # Processing the call keyword arguments (line 123)
    kwargs_139114 = {}
    # Getting the type of 'print' (line 123)
    print_139112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'print', False)
    # Calling print(args, kwargs) (line 123)
    print_call_result_139115 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), print_139112, *[str_139113], **kwargs_139114)
    
    
    # Call to timer(...): (line 124)
    # Processing the call arguments (line 124)
    str_139117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 10), 'str', 'nmxl.__setitem__((-1,0),numpy.ma.masked)')
    str_139118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 54), 'str', 'numpy.ma   ')
    # Processing the call keyword arguments (line 124)
    int_139119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 75), 'int')
    keyword_139120 = int_139119
    kwargs_139121 = {'nloop': keyword_139120}
    # Getting the type of 'timer' (line 124)
    timer_139116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 124)
    timer_call_result_139122 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), timer_139116, *[str_139117, str_139118], **kwargs_139121)
    
    
    # Call to print(...): (line 126)
    # Processing the call arguments (line 126)
    str_139124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 10), 'str', '-')
    int_139125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 14), 'int')
    # Applying the binary operator '*' (line 126)
    result_mul_139126 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 10), '*', str_139124, int_139125)
    
    # Processing the call keyword arguments (line 126)
    kwargs_139127 = {}
    # Getting the type of 'print' (line 126)
    print_139123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'print', False)
    # Calling print(args, kwargs) (line 126)
    print_call_result_139128 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), print_139123, *[result_mul_139126], **kwargs_139127)
    
    
    # Call to print(...): (line 127)
    # Processing the call arguments (line 127)
    str_139130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 10), 'str', 'where on small arrays')
    # Processing the call keyword arguments (line 127)
    kwargs_139131 = {}
    # Getting the type of 'print' (line 127)
    print_139129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'print', False)
    # Calling print(args, kwargs) (line 127)
    print_call_result_139132 = invoke(stypy.reporting.localization.Localization(__file__, 127, 4), print_139129, *[str_139130], **kwargs_139131)
    
    
    # Call to timer(...): (line 128)
    # Processing the call arguments (line 128)
    str_139134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 10), 'str', 'numpy.ma.where(nmxs>2,nmxs,nmys)')
    str_139135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 46), 'str', 'numpy.ma   ')
    # Processing the call keyword arguments (line 128)
    int_139136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 67), 'int')
    keyword_139137 = int_139136
    kwargs_139138 = {'nloop': keyword_139137}
    # Getting the type of 'timer' (line 128)
    timer_139133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 128)
    timer_call_result_139139 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), timer_139133, *[str_139134, str_139135], **kwargs_139138)
    
    
    # Call to print(...): (line 129)
    # Processing the call arguments (line 129)
    str_139141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 10), 'str', '-')
    int_139142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 14), 'int')
    # Applying the binary operator '*' (line 129)
    result_mul_139143 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), '*', str_139141, int_139142)
    
    # Processing the call keyword arguments (line 129)
    kwargs_139144 = {}
    # Getting the type of 'print' (line 129)
    print_139140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'print', False)
    # Calling print(args, kwargs) (line 129)
    print_call_result_139145 = invoke(stypy.reporting.localization.Localization(__file__, 129, 4), print_139140, *[result_mul_139143], **kwargs_139144)
    
    
    # Call to print(...): (line 130)
    # Processing the call arguments (line 130)
    str_139147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 10), 'str', 'where on large arrays')
    # Processing the call keyword arguments (line 130)
    kwargs_139148 = {}
    # Getting the type of 'print' (line 130)
    print_139146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'print', False)
    # Calling print(args, kwargs) (line 130)
    print_call_result_139149 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), print_139146, *[str_139147], **kwargs_139148)
    
    
    # Call to timer(...): (line 131)
    # Processing the call arguments (line 131)
    str_139151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 10), 'str', 'numpy.ma.where(nmxl>2,nmxl,nmyl)')
    str_139152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 46), 'str', 'numpy.ma   ')
    # Processing the call keyword arguments (line 131)
    int_139153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 67), 'int')
    keyword_139154 = int_139153
    kwargs_139155 = {'nloop': keyword_139154}
    # Getting the type of 'timer' (line 131)
    timer_139150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'timer', False)
    # Calling timer(args, kwargs) (line 131)
    timer_call_result_139156 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), timer_139150, *[str_139151, str_139152], **kwargs_139155)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
