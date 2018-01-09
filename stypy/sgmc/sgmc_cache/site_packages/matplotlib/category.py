
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 OA-*-za
2: '''
3: catch all for categorical functions
4: '''
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: import six
8: 
9: import numpy as np
10: 
11: import matplotlib.units as units
12: import matplotlib.ticker as ticker
13: 
14: # np 1.6/1.7 support
15: from distutils.version import LooseVersion
16: import collections
17: 
18: 
19: if LooseVersion(np.__version__) >= LooseVersion('1.8.0'):
20:     def shim_array(data):
21:         return np.array(data, dtype=np.unicode)
22: else:
23:     def shim_array(data):
24:         if (isinstance(data, six.string_types) or
25:                 not isinstance(data, collections.Iterable)):
26:             data = [data]
27:         try:
28:             data = [str(d) for d in data]
29:         except UnicodeEncodeError:
30:             # this yields gibberish but unicode text doesn't
31:             # render under numpy1.6 anyway
32:             data = [d.encode('utf-8', 'ignore').decode('utf-8')
33:                     for d in data]
34:         return np.array(data, dtype=np.unicode)
35: 
36: 
37: class StrCategoryConverter(units.ConversionInterface):
38:     @staticmethod
39:     def convert(value, unit, axis):
40:         '''Uses axis.unit_data map to encode
41:         data as floats
42:         '''
43:         vmap = dict(zip(axis.unit_data.seq, axis.unit_data.locs))
44: 
45:         if isinstance(value, six.string_types):
46:             return vmap[value]
47: 
48:         vals = shim_array(value)
49: 
50:         for lab, loc in vmap.items():
51:             vals[vals == lab] = loc
52: 
53:         return vals.astype('float')
54: 
55:     @staticmethod
56:     def axisinfo(unit, axis):
57:         majloc = StrCategoryLocator(axis.unit_data.locs)
58:         majfmt = StrCategoryFormatter(axis.unit_data.seq)
59:         return units.AxisInfo(majloc=majloc, majfmt=majfmt)
60: 
61:     @staticmethod
62:     def default_units(data, axis):
63:         # the conversion call stack is:
64:         # default_units->axis_info->convert
65:         if axis.unit_data is None:
66:             axis.unit_data = UnitData(data)
67:         else:
68:             axis.unit_data.update(data)
69:         return None
70: 
71: 
72: class StrCategoryLocator(ticker.FixedLocator):
73:     def __init__(self, locs):
74:         self.locs = locs
75:         self.nbins = None
76: 
77: 
78: class StrCategoryFormatter(ticker.FixedFormatter):
79:     def __init__(self, seq):
80:         self.seq = seq
81:         self.offset_string = ''
82: 
83: 
84: class UnitData(object):
85:     # debatable makes sense to special code missing values
86:     spdict = {'nan': -1.0, 'inf': -2.0, '-inf': -3.0}
87: 
88:     def __init__(self, data):
89:         '''Create mapping between unique categorical values
90:         and numerical identifier
91: 
92:         Parameters
93:         ----------
94:         data: iterable
95:             sequence of values
96:         '''
97:         self.seq, self.locs = [], []
98:         self._set_seq_locs(data, 0)
99: 
100:     def update(self, new_data):
101:         # so as not to conflict with spdict
102:         value = max(max(self.locs) + 1, 0)
103:         self._set_seq_locs(new_data, value)
104: 
105:     def _set_seq_locs(self, data, value):
106:         strdata = shim_array(data)
107:         new_s = [d for d in np.unique(strdata) if d not in self.seq]
108:         for ns in new_s:
109:             self.seq.append(ns)
110:             if ns in UnitData.spdict:
111:                 self.locs.append(UnitData.spdict[ns])
112:             else:
113:                 self.locs.append(value)
114:                 value += 1
115: 
116: 
117: # Connects the convertor to matplotlib
118: units.registry[str] = StrCategoryConverter()
119: units.registry[np.str_] = StrCategoryConverter()
120: units.registry[six.text_type] = StrCategoryConverter()
121: units.registry[bytes] = StrCategoryConverter()
122: units.registry[np.bytes_] = StrCategoryConverter()
123: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_25215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'unicode', u'\ncatch all for categorical functions\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import six' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six')

if (type(import_25216) is not StypyTypeError):

    if (import_25216 != 'pyd_module'):
        __import__(import_25216)
        sys_modules_25217 = sys.modules[import_25216]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', sys_modules_25217.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'six', import_25216)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25218 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_25218) is not StypyTypeError):

    if (import_25218 != 'pyd_module'):
        __import__(import_25218)
        sys_modules_25219 = sys.modules[import_25218]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_25219.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_25218)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import matplotlib.units' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25220 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.units')

if (type(import_25220) is not StypyTypeError):

    if (import_25220 != 'pyd_module'):
        __import__(import_25220)
        sys_modules_25221 = sys.modules[import_25220]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'units', sys_modules_25221.module_type_store, module_type_store)
    else:
        import matplotlib.units as units

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'units', matplotlib.units, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.units' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'matplotlib.units', import_25220)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import matplotlib.ticker' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25222 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker')

if (type(import_25222) is not StypyTypeError):

    if (import_25222 != 'pyd_module'):
        __import__(import_25222)
        sys_modules_25223 = sys.modules[import_25222]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ticker', sys_modules_25223.module_type_store, module_type_store)
    else:
        import matplotlib.ticker as ticker

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ticker', matplotlib.ticker, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.ticker', import_25222)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.version import LooseVersion' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_25224 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.version')

if (type(import_25224) is not StypyTypeError):

    if (import_25224 != 'pyd_module'):
        __import__(import_25224)
        sys_modules_25225 = sys.modules[import_25224]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.version', sys_modules_25225.module_type_store, module_type_store, ['LooseVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_25225, sys_modules_25225.module_type_store, module_type_store)
    else:
        from distutils.version import LooseVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.version', import_25224)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import collections' statement (line 16)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'collections', collections, module_type_store)




# Call to LooseVersion(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'np' (line 19)
np_25227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 19)
version___25228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), np_25227, '__version__')
# Processing the call keyword arguments (line 19)
kwargs_25229 = {}
# Getting the type of 'LooseVersion' (line 19)
LooseVersion_25226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 3), 'LooseVersion', False)
# Calling LooseVersion(args, kwargs) (line 19)
LooseVersion_call_result_25230 = invoke(stypy.reporting.localization.Localization(__file__, 19, 3), LooseVersion_25226, *[version___25228], **kwargs_25229)


# Call to LooseVersion(...): (line 19)
# Processing the call arguments (line 19)
unicode_25232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 48), 'unicode', u'1.8.0')
# Processing the call keyword arguments (line 19)
kwargs_25233 = {}
# Getting the type of 'LooseVersion' (line 19)
LooseVersion_25231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'LooseVersion', False)
# Calling LooseVersion(args, kwargs) (line 19)
LooseVersion_call_result_25234 = invoke(stypy.reporting.localization.Localization(__file__, 19, 35), LooseVersion_25231, *[unicode_25232], **kwargs_25233)

# Applying the binary operator '>=' (line 19)
result_ge_25235 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 3), '>=', LooseVersion_call_result_25230, LooseVersion_call_result_25234)

# Testing the type of an if condition (line 19)
if_condition_25236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 0), result_ge_25235)
# Assigning a type to the variable 'if_condition_25236' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'if_condition_25236', if_condition_25236)
# SSA begins for if statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def shim_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shim_array'
    module_type_store = module_type_store.open_function_context('shim_array', 20, 4, False)
    
    # Passed parameters checking function
    shim_array.stypy_localization = localization
    shim_array.stypy_type_of_self = None
    shim_array.stypy_type_store = module_type_store
    shim_array.stypy_function_name = 'shim_array'
    shim_array.stypy_param_names_list = ['data']
    shim_array.stypy_varargs_param_name = None
    shim_array.stypy_kwargs_param_name = None
    shim_array.stypy_call_defaults = defaults
    shim_array.stypy_call_varargs = varargs
    shim_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shim_array', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shim_array', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shim_array(...)' code ##################

    
    # Call to array(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'data' (line 21)
    data_25239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'data', False)
    # Processing the call keyword arguments (line 21)
    # Getting the type of 'np' (line 21)
    np_25240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'np', False)
    # Obtaining the member 'unicode' of a type (line 21)
    unicode_25241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 36), np_25240, 'unicode')
    keyword_25242 = unicode_25241
    kwargs_25243 = {'dtype': keyword_25242}
    # Getting the type of 'np' (line 21)
    np_25237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 21)
    array_25238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), np_25237, 'array')
    # Calling array(args, kwargs) (line 21)
    array_call_result_25244 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), array_25238, *[data_25239], **kwargs_25243)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', array_call_result_25244)
    
    # ################# End of 'shim_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shim_array' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_25245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shim_array'
    return stypy_return_type_25245

# Assigning a type to the variable 'shim_array' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'shim_array', shim_array)
# SSA branch for the else part of an if statement (line 19)
module_type_store.open_ssa_branch('else')

@norecursion
def shim_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shim_array'
    module_type_store = module_type_store.open_function_context('shim_array', 23, 4, False)
    
    # Passed parameters checking function
    shim_array.stypy_localization = localization
    shim_array.stypy_type_of_self = None
    shim_array.stypy_type_store = module_type_store
    shim_array.stypy_function_name = 'shim_array'
    shim_array.stypy_param_names_list = ['data']
    shim_array.stypy_varargs_param_name = None
    shim_array.stypy_kwargs_param_name = None
    shim_array.stypy_call_defaults = defaults
    shim_array.stypy_call_varargs = varargs
    shim_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shim_array', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shim_array', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shim_array(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'data' (line 24)
    data_25247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'data', False)
    # Getting the type of 'six' (line 24)
    six_25248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'six', False)
    # Obtaining the member 'string_types' of a type (line 24)
    string_types_25249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), six_25248, 'string_types')
    # Processing the call keyword arguments (line 24)
    kwargs_25250 = {}
    # Getting the type of 'isinstance' (line 24)
    isinstance_25246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 24)
    isinstance_call_result_25251 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), isinstance_25246, *[data_25247, string_types_25249], **kwargs_25250)
    
    
    
    # Call to isinstance(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'data' (line 25)
    data_25253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'data', False)
    # Getting the type of 'collections' (line 25)
    collections_25254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'collections', False)
    # Obtaining the member 'Iterable' of a type (line 25)
    Iterable_25255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 37), collections_25254, 'Iterable')
    # Processing the call keyword arguments (line 25)
    kwargs_25256 = {}
    # Getting the type of 'isinstance' (line 25)
    isinstance_25252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 25)
    isinstance_call_result_25257 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), isinstance_25252, *[data_25253, Iterable_25255], **kwargs_25256)
    
    # Applying the 'not' unary operator (line 25)
    result_not__25258 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), 'not', isinstance_call_result_25257)
    
    # Applying the binary operator 'or' (line 24)
    result_or_keyword_25259 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), 'or', isinstance_call_result_25251, result_not__25258)
    
    # Testing the type of an if condition (line 24)
    if_condition_25260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_or_keyword_25259)
    # Assigning a type to the variable 'if_condition_25260' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_25260', if_condition_25260)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 26):
    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_25261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'data' (line 26)
    data_25262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'data')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_25261, data_25262)
    
    # Assigning a type to the variable 'data' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'data', list_25261)
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a ListComp to a Name (line 28):
    
    # Assigning a ListComp to a Name (line 28):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'data' (line 28)
    data_25267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'data')
    comprehension_25268 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), data_25267)
    # Assigning a type to the variable 'd' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'd', comprehension_25268)
    
    # Call to str(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'd' (line 28)
    d_25264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'd', False)
    # Processing the call keyword arguments (line 28)
    kwargs_25265 = {}
    # Getting the type of 'str' (line 28)
    str_25263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'str', False)
    # Calling str(args, kwargs) (line 28)
    str_call_result_25266 = invoke(stypy.reporting.localization.Localization(__file__, 28, 20), str_25263, *[d_25264], **kwargs_25265)
    
    list_25269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 20), list_25269, str_call_result_25266)
    # Assigning a type to the variable 'data' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'data', list_25269)
    # SSA branch for the except part of a try statement (line 27)
    # SSA branch for the except 'UnicodeEncodeError' branch of a try statement (line 27)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a ListComp to a Name (line 32):
    
    # Assigning a ListComp to a Name (line 32):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'data' (line 33)
    data_25280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'data')
    comprehension_25281 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), data_25280)
    # Assigning a type to the variable 'd' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'd', comprehension_25281)
    
    # Call to decode(...): (line 32)
    # Processing the call arguments (line 32)
    unicode_25277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 55), 'unicode', u'utf-8')
    # Processing the call keyword arguments (line 32)
    kwargs_25278 = {}
    
    # Call to encode(...): (line 32)
    # Processing the call arguments (line 32)
    unicode_25272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'unicode', u'utf-8')
    unicode_25273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 38), 'unicode', u'ignore')
    # Processing the call keyword arguments (line 32)
    kwargs_25274 = {}
    # Getting the type of 'd' (line 32)
    d_25270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'd', False)
    # Obtaining the member 'encode' of a type (line 32)
    encode_25271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), d_25270, 'encode')
    # Calling encode(args, kwargs) (line 32)
    encode_call_result_25275 = invoke(stypy.reporting.localization.Localization(__file__, 32, 20), encode_25271, *[unicode_25272, unicode_25273], **kwargs_25274)
    
    # Obtaining the member 'decode' of a type (line 32)
    decode_25276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 20), encode_call_result_25275, 'decode')
    # Calling decode(args, kwargs) (line 32)
    decode_call_result_25279 = invoke(stypy.reporting.localization.Localization(__file__, 32, 20), decode_25276, *[unicode_25277], **kwargs_25278)
    
    list_25282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 20), list_25282, decode_call_result_25279)
    # Assigning a type to the variable 'data' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'data', list_25282)
    # SSA join for try-except statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to array(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'data' (line 34)
    data_25285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'data', False)
    # Processing the call keyword arguments (line 34)
    # Getting the type of 'np' (line 34)
    np_25286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'np', False)
    # Obtaining the member 'unicode' of a type (line 34)
    unicode_25287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 36), np_25286, 'unicode')
    keyword_25288 = unicode_25287
    kwargs_25289 = {'dtype': keyword_25288}
    # Getting the type of 'np' (line 34)
    np_25283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 34)
    array_25284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), np_25283, 'array')
    # Calling array(args, kwargs) (line 34)
    array_call_result_25290 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), array_25284, *[data_25285], **kwargs_25289)
    
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', array_call_result_25290)
    
    # ################# End of 'shim_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shim_array' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_25291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25291)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shim_array'
    return stypy_return_type_25291

# Assigning a type to the variable 'shim_array' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'shim_array', shim_array)
# SSA join for if statement (line 19)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'StrCategoryConverter' class
# Getting the type of 'units' (line 37)
units_25292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'units')
# Obtaining the member 'ConversionInterface' of a type (line 37)
ConversionInterface_25293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), units_25292, 'ConversionInterface')

class StrCategoryConverter(ConversionInterface_25293, ):

    @staticmethod
    @norecursion
    def convert(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'convert'
        module_type_store = module_type_store.open_function_context('convert', 38, 4, False)
        
        # Passed parameters checking function
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_localization', localization)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_type_of_self', None)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_function_name', 'convert')
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_param_names_list', ['value', 'unit', 'axis'])
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrCategoryConverter.convert.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'convert', ['value', 'unit', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'convert', localization, ['unit', 'axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'convert(...)' code ##################

        unicode_25294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'unicode', u'Uses axis.unit_data map to encode\n        data as floats\n        ')
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to dict(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to zip(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'axis' (line 43)
        axis_25297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'axis', False)
        # Obtaining the member 'unit_data' of a type (line 43)
        unit_data_25298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), axis_25297, 'unit_data')
        # Obtaining the member 'seq' of a type (line 43)
        seq_25299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), unit_data_25298, 'seq')
        # Getting the type of 'axis' (line 43)
        axis_25300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 44), 'axis', False)
        # Obtaining the member 'unit_data' of a type (line 43)
        unit_data_25301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 44), axis_25300, 'unit_data')
        # Obtaining the member 'locs' of a type (line 43)
        locs_25302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 44), unit_data_25301, 'locs')
        # Processing the call keyword arguments (line 43)
        kwargs_25303 = {}
        # Getting the type of 'zip' (line 43)
        zip_25296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 43)
        zip_call_result_25304 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), zip_25296, *[seq_25299, locs_25302], **kwargs_25303)
        
        # Processing the call keyword arguments (line 43)
        kwargs_25305 = {}
        # Getting the type of 'dict' (line 43)
        dict_25295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 43)
        dict_call_result_25306 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), dict_25295, *[zip_call_result_25304], **kwargs_25305)
        
        # Assigning a type to the variable 'vmap' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'vmap', dict_call_result_25306)
        
        
        # Call to isinstance(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'value' (line 45)
        value_25308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'value', False)
        # Getting the type of 'six' (line 45)
        six_25309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'six', False)
        # Obtaining the member 'string_types' of a type (line 45)
        string_types_25310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 29), six_25309, 'string_types')
        # Processing the call keyword arguments (line 45)
        kwargs_25311 = {}
        # Getting the type of 'isinstance' (line 45)
        isinstance_25307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 45)
        isinstance_call_result_25312 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), isinstance_25307, *[value_25308, string_types_25310], **kwargs_25311)
        
        # Testing the type of an if condition (line 45)
        if_condition_25313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), isinstance_call_result_25312)
        # Assigning a type to the variable 'if_condition_25313' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_25313', if_condition_25313)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        # Getting the type of 'value' (line 46)
        value_25314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'value')
        # Getting the type of 'vmap' (line 46)
        vmap_25315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'vmap')
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___25316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), vmap_25315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_25317 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), getitem___25316, value_25314)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', subscript_call_result_25317)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to shim_array(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'value' (line 48)
        value_25319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'value', False)
        # Processing the call keyword arguments (line 48)
        kwargs_25320 = {}
        # Getting the type of 'shim_array' (line 48)
        shim_array_25318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'shim_array', False)
        # Calling shim_array(args, kwargs) (line 48)
        shim_array_call_result_25321 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), shim_array_25318, *[value_25319], **kwargs_25320)
        
        # Assigning a type to the variable 'vals' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'vals', shim_array_call_result_25321)
        
        
        # Call to items(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_25324 = {}
        # Getting the type of 'vmap' (line 50)
        vmap_25322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'vmap', False)
        # Obtaining the member 'items' of a type (line 50)
        items_25323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), vmap_25322, 'items')
        # Calling items(args, kwargs) (line 50)
        items_call_result_25325 = invoke(stypy.reporting.localization.Localization(__file__, 50, 24), items_25323, *[], **kwargs_25324)
        
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 8), items_call_result_25325)
        # Getting the type of the for loop variable (line 50)
        for_loop_var_25326 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 8), items_call_result_25325)
        # Assigning a type to the variable 'lab' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'lab', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), for_loop_var_25326))
        # Assigning a type to the variable 'loc' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'loc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), for_loop_var_25326))
        # SSA begins for a for statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 51):
        
        # Assigning a Name to a Subscript (line 51):
        # Getting the type of 'loc' (line 51)
        loc_25327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'loc')
        # Getting the type of 'vals' (line 51)
        vals_25328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'vals')
        
        # Getting the type of 'vals' (line 51)
        vals_25329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'vals')
        # Getting the type of 'lab' (line 51)
        lab_25330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'lab')
        # Applying the binary operator '==' (line 51)
        result_eq_25331 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 17), '==', vals_25329, lab_25330)
        
        # Storing an element on a container (line 51)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 12), vals_25328, (result_eq_25331, loc_25327))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to astype(...): (line 53)
        # Processing the call arguments (line 53)
        unicode_25334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'unicode', u'float')
        # Processing the call keyword arguments (line 53)
        kwargs_25335 = {}
        # Getting the type of 'vals' (line 53)
        vals_25332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'vals', False)
        # Obtaining the member 'astype' of a type (line 53)
        astype_25333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), vals_25332, 'astype')
        # Calling astype(args, kwargs) (line 53)
        astype_call_result_25336 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), astype_25333, *[unicode_25334], **kwargs_25335)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', astype_call_result_25336)
        
        # ################# End of 'convert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'convert' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_25337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'convert'
        return stypy_return_type_25337


    @staticmethod
    @norecursion
    def axisinfo(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'axisinfo'
        module_type_store = module_type_store.open_function_context('axisinfo', 55, 4, False)
        
        # Passed parameters checking function
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_localization', localization)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_type_of_self', None)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_function_name', 'axisinfo')
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_param_names_list', ['unit', 'axis'])
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrCategoryConverter.axisinfo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'axisinfo', ['unit', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'axisinfo', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'axisinfo(...)' code ##################

        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to StrCategoryLocator(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'axis' (line 57)
        axis_25339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 36), 'axis', False)
        # Obtaining the member 'unit_data' of a type (line 57)
        unit_data_25340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 36), axis_25339, 'unit_data')
        # Obtaining the member 'locs' of a type (line 57)
        locs_25341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 36), unit_data_25340, 'locs')
        # Processing the call keyword arguments (line 57)
        kwargs_25342 = {}
        # Getting the type of 'StrCategoryLocator' (line 57)
        StrCategoryLocator_25338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'StrCategoryLocator', False)
        # Calling StrCategoryLocator(args, kwargs) (line 57)
        StrCategoryLocator_call_result_25343 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), StrCategoryLocator_25338, *[locs_25341], **kwargs_25342)
        
        # Assigning a type to the variable 'majloc' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'majloc', StrCategoryLocator_call_result_25343)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to StrCategoryFormatter(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'axis' (line 58)
        axis_25345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'axis', False)
        # Obtaining the member 'unit_data' of a type (line 58)
        unit_data_25346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 38), axis_25345, 'unit_data')
        # Obtaining the member 'seq' of a type (line 58)
        seq_25347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 38), unit_data_25346, 'seq')
        # Processing the call keyword arguments (line 58)
        kwargs_25348 = {}
        # Getting the type of 'StrCategoryFormatter' (line 58)
        StrCategoryFormatter_25344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'StrCategoryFormatter', False)
        # Calling StrCategoryFormatter(args, kwargs) (line 58)
        StrCategoryFormatter_call_result_25349 = invoke(stypy.reporting.localization.Localization(__file__, 58, 17), StrCategoryFormatter_25344, *[seq_25347], **kwargs_25348)
        
        # Assigning a type to the variable 'majfmt' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'majfmt', StrCategoryFormatter_call_result_25349)
        
        # Call to AxisInfo(...): (line 59)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'majloc' (line 59)
        majloc_25352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'majloc', False)
        keyword_25353 = majloc_25352
        # Getting the type of 'majfmt' (line 59)
        majfmt_25354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'majfmt', False)
        keyword_25355 = majfmt_25354
        kwargs_25356 = {'majloc': keyword_25353, 'majfmt': keyword_25355}
        # Getting the type of 'units' (line 59)
        units_25350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'units', False)
        # Obtaining the member 'AxisInfo' of a type (line 59)
        AxisInfo_25351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), units_25350, 'AxisInfo')
        # Calling AxisInfo(args, kwargs) (line 59)
        AxisInfo_call_result_25357 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), AxisInfo_25351, *[], **kwargs_25356)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', AxisInfo_call_result_25357)
        
        # ################# End of 'axisinfo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'axisinfo' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_25358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25358)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'axisinfo'
        return stypy_return_type_25358


    @staticmethod
    @norecursion
    def default_units(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default_units'
        module_type_store = module_type_store.open_function_context('default_units', 61, 4, False)
        
        # Passed parameters checking function
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_localization', localization)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_type_of_self', None)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_type_store', module_type_store)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_function_name', 'default_units')
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_param_names_list', ['data', 'axis'])
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_varargs_param_name', None)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_call_defaults', defaults)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_call_varargs', varargs)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StrCategoryConverter.default_units.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'default_units', ['data', 'axis'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'default_units', localization, ['axis'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'default_units(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 65)
        # Getting the type of 'axis' (line 65)
        axis_25359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'axis')
        # Obtaining the member 'unit_data' of a type (line 65)
        unit_data_25360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), axis_25359, 'unit_data')
        # Getting the type of 'None' (line 65)
        None_25361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'None')
        
        (may_be_25362, more_types_in_union_25363) = may_be_none(unit_data_25360, None_25361)

        if may_be_25362:

            if more_types_in_union_25363:
                # Runtime conditional SSA (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 66):
            
            # Assigning a Call to a Attribute (line 66):
            
            # Call to UnitData(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'data' (line 66)
            data_25365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'data', False)
            # Processing the call keyword arguments (line 66)
            kwargs_25366 = {}
            # Getting the type of 'UnitData' (line 66)
            UnitData_25364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'UnitData', False)
            # Calling UnitData(args, kwargs) (line 66)
            UnitData_call_result_25367 = invoke(stypy.reporting.localization.Localization(__file__, 66, 29), UnitData_25364, *[data_25365], **kwargs_25366)
            
            # Getting the type of 'axis' (line 66)
            axis_25368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'axis')
            # Setting the type of the member 'unit_data' of a type (line 66)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), axis_25368, 'unit_data', UnitData_call_result_25367)

            if more_types_in_union_25363:
                # Runtime conditional SSA for else branch (line 65)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_25362) or more_types_in_union_25363):
            
            # Call to update(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'data' (line 68)
            data_25372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'data', False)
            # Processing the call keyword arguments (line 68)
            kwargs_25373 = {}
            # Getting the type of 'axis' (line 68)
            axis_25369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'axis', False)
            # Obtaining the member 'unit_data' of a type (line 68)
            unit_data_25370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), axis_25369, 'unit_data')
            # Obtaining the member 'update' of a type (line 68)
            update_25371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), unit_data_25370, 'update')
            # Calling update(args, kwargs) (line 68)
            update_call_result_25374 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), update_25371, *[data_25372], **kwargs_25373)
            

            if (may_be_25362 and more_types_in_union_25363):
                # SSA join for if statement (line 65)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'None' (line 69)
        None_25375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', None_25375)
        
        # ################# End of 'default_units(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default_units' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_25376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default_units'
        return stypy_return_type_25376


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 37, 0, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrCategoryConverter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'StrCategoryConverter' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'StrCategoryConverter', StrCategoryConverter)
# Declaration of the 'StrCategoryLocator' class
# Getting the type of 'ticker' (line 72)
ticker_25377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'ticker')
# Obtaining the member 'FixedLocator' of a type (line 72)
FixedLocator_25378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), ticker_25377, 'FixedLocator')

class StrCategoryLocator(FixedLocator_25378, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrCategoryLocator.__init__', ['locs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['locs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 74):
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'locs' (line 74)
        locs_25379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'locs')
        # Getting the type of 'self' (line 74)
        self_25380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'locs' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_25380, 'locs', locs_25379)
        
        # Assigning a Name to a Attribute (line 75):
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'None' (line 75)
        None_25381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'None')
        # Getting the type of 'self' (line 75)
        self_25382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'nbins' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_25382, 'nbins', None_25381)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'StrCategoryLocator' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'StrCategoryLocator', StrCategoryLocator)
# Declaration of the 'StrCategoryFormatter' class
# Getting the type of 'ticker' (line 78)
ticker_25383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'ticker')
# Obtaining the member 'FixedFormatter' of a type (line 78)
FixedFormatter_25384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), ticker_25383, 'FixedFormatter')

class StrCategoryFormatter(FixedFormatter_25384, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StrCategoryFormatter.__init__', ['seq'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['seq'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 80):
        
        # Assigning a Name to a Attribute (line 80):
        # Getting the type of 'seq' (line 80)
        seq_25385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'seq')
        # Getting the type of 'self' (line 80)
        self_25386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member 'seq' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_25386, 'seq', seq_25385)
        
        # Assigning a Str to a Attribute (line 81):
        
        # Assigning a Str to a Attribute (line 81):
        unicode_25387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'unicode', u'')
        # Getting the type of 'self' (line 81)
        self_25388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'offset_string' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_25388, 'offset_string', unicode_25387)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'StrCategoryFormatter' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'StrCategoryFormatter', StrCategoryFormatter)
# Declaration of the 'UnitData' class

class UnitData(object, ):
    
    # Assigning a Dict to a Name (line 86):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitData.__init__', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        unicode_25389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'unicode', u'Create mapping between unique categorical values\n        and numerical identifier\n\n        Parameters\n        ----------\n        data: iterable\n            sequence of values\n        ')
        
        # Assigning a Tuple to a Tuple (line 97):
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_25390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Assigning a type to the variable 'tuple_assignment_25213' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_assignment_25213', list_25390)
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_25391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Assigning a type to the variable 'tuple_assignment_25214' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_assignment_25214', list_25391)
        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'tuple_assignment_25213' (line 97)
        tuple_assignment_25213_25392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_assignment_25213')
        # Getting the type of 'self' (line 97)
        self_25393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'seq' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_25393, 'seq', tuple_assignment_25213_25392)
        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'tuple_assignment_25214' (line 97)
        tuple_assignment_25214_25394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'tuple_assignment_25214')
        # Getting the type of 'self' (line 97)
        self_25395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'self')
        # Setting the type of the member 'locs' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 18), self_25395, 'locs', tuple_assignment_25214_25394)
        
        # Call to _set_seq_locs(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'data' (line 98)
        data_25398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'data', False)
        int_25399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_25400 = {}
        # Getting the type of 'self' (line 98)
        self_25396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self', False)
        # Obtaining the member '_set_seq_locs' of a type (line 98)
        _set_seq_locs_25397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_25396, '_set_seq_locs')
        # Calling _set_seq_locs(args, kwargs) (line 98)
        _set_seq_locs_call_result_25401 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), _set_seq_locs_25397, *[data_25398, int_25399], **kwargs_25400)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitData.update.__dict__.__setitem__('stypy_localization', localization)
        UnitData.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitData.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitData.update.__dict__.__setitem__('stypy_function_name', 'UnitData.update')
        UnitData.update.__dict__.__setitem__('stypy_param_names_list', ['new_data'])
        UnitData.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitData.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitData.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitData.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitData.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitData.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitData.update', ['new_data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['new_data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to max(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to max(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'self' (line 102)
        self_25404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'self', False)
        # Obtaining the member 'locs' of a type (line 102)
        locs_25405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), self_25404, 'locs')
        # Processing the call keyword arguments (line 102)
        kwargs_25406 = {}
        # Getting the type of 'max' (line 102)
        max_25403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'max', False)
        # Calling max(args, kwargs) (line 102)
        max_call_result_25407 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), max_25403, *[locs_25405], **kwargs_25406)
        
        int_25408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
        # Applying the binary operator '+' (line 102)
        result_add_25409 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 20), '+', max_call_result_25407, int_25408)
        
        int_25410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 40), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_25411 = {}
        # Getting the type of 'max' (line 102)
        max_25402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'max', False)
        # Calling max(args, kwargs) (line 102)
        max_call_result_25412 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), max_25402, *[result_add_25409, int_25410], **kwargs_25411)
        
        # Assigning a type to the variable 'value' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'value', max_call_result_25412)
        
        # Call to _set_seq_locs(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'new_data' (line 103)
        new_data_25415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'new_data', False)
        # Getting the type of 'value' (line 103)
        value_25416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'value', False)
        # Processing the call keyword arguments (line 103)
        kwargs_25417 = {}
        # Getting the type of 'self' (line 103)
        self_25413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self', False)
        # Obtaining the member '_set_seq_locs' of a type (line 103)
        _set_seq_locs_25414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_25413, '_set_seq_locs')
        # Calling _set_seq_locs(args, kwargs) (line 103)
        _set_seq_locs_call_result_25418 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), _set_seq_locs_25414, *[new_data_25415, value_25416], **kwargs_25417)
        
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_25419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_25419


    @norecursion
    def _set_seq_locs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_seq_locs'
        module_type_store = module_type_store.open_function_context('_set_seq_locs', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_localization', localization)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_function_name', 'UnitData._set_seq_locs')
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_param_names_list', ['data', 'value'])
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitData._set_seq_locs.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitData._set_seq_locs', ['data', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_seq_locs', localization, ['data', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_seq_locs(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to shim_array(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'data' (line 106)
        data_25421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'data', False)
        # Processing the call keyword arguments (line 106)
        kwargs_25422 = {}
        # Getting the type of 'shim_array' (line 106)
        shim_array_25420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'shim_array', False)
        # Calling shim_array(args, kwargs) (line 106)
        shim_array_call_result_25423 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), shim_array_25420, *[data_25421], **kwargs_25422)
        
        # Assigning a type to the variable 'strdata' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'strdata', shim_array_call_result_25423)
        
        # Assigning a ListComp to a Name (line 107):
        
        # Assigning a ListComp to a Name (line 107):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to unique(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'strdata' (line 107)
        strdata_25431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'strdata', False)
        # Processing the call keyword arguments (line 107)
        kwargs_25432 = {}
        # Getting the type of 'np' (line 107)
        np_25429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 28), 'np', False)
        # Obtaining the member 'unique' of a type (line 107)
        unique_25430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 28), np_25429, 'unique')
        # Calling unique(args, kwargs) (line 107)
        unique_call_result_25433 = invoke(stypy.reporting.localization.Localization(__file__, 107, 28), unique_25430, *[strdata_25431], **kwargs_25432)
        
        comprehension_25434 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), unique_call_result_25433)
        # Assigning a type to the variable 'd' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'd', comprehension_25434)
        
        # Getting the type of 'd' (line 107)
        d_25425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 50), 'd')
        # Getting the type of 'self' (line 107)
        self_25426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 59), 'self')
        # Obtaining the member 'seq' of a type (line 107)
        seq_25427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 59), self_25426, 'seq')
        # Applying the binary operator 'notin' (line 107)
        result_contains_25428 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 50), 'notin', d_25425, seq_25427)
        
        # Getting the type of 'd' (line 107)
        d_25424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'd')
        list_25435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), list_25435, d_25424)
        # Assigning a type to the variable 'new_s' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'new_s', list_25435)
        
        # Getting the type of 'new_s' (line 108)
        new_s_25436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'new_s')
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), new_s_25436)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_25437 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), new_s_25436)
        # Assigning a type to the variable 'ns' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'ns', for_loop_var_25437)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'ns' (line 109)
        ns_25441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'ns', False)
        # Processing the call keyword arguments (line 109)
        kwargs_25442 = {}
        # Getting the type of 'self' (line 109)
        self_25438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'self', False)
        # Obtaining the member 'seq' of a type (line 109)
        seq_25439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), self_25438, 'seq')
        # Obtaining the member 'append' of a type (line 109)
        append_25440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), seq_25439, 'append')
        # Calling append(args, kwargs) (line 109)
        append_call_result_25443 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), append_25440, *[ns_25441], **kwargs_25442)
        
        
        
        # Getting the type of 'ns' (line 110)
        ns_25444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'ns')
        # Getting the type of 'UnitData' (line 110)
        UnitData_25445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'UnitData')
        # Obtaining the member 'spdict' of a type (line 110)
        spdict_25446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 21), UnitData_25445, 'spdict')
        # Applying the binary operator 'in' (line 110)
        result_contains_25447 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), 'in', ns_25444, spdict_25446)
        
        # Testing the type of an if condition (line 110)
        if_condition_25448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), result_contains_25447)
        # Assigning a type to the variable 'if_condition_25448' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_25448', if_condition_25448)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ns' (line 111)
        ns_25452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'ns', False)
        # Getting the type of 'UnitData' (line 111)
        UnitData_25453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'UnitData', False)
        # Obtaining the member 'spdict' of a type (line 111)
        spdict_25454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), UnitData_25453, 'spdict')
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___25455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 33), spdict_25454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_25456 = invoke(stypy.reporting.localization.Localization(__file__, 111, 33), getitem___25455, ns_25452)
        
        # Processing the call keyword arguments (line 111)
        kwargs_25457 = {}
        # Getting the type of 'self' (line 111)
        self_25449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'locs' of a type (line 111)
        locs_25450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_25449, 'locs')
        # Obtaining the member 'append' of a type (line 111)
        append_25451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), locs_25450, 'append')
        # Calling append(args, kwargs) (line 111)
        append_call_result_25458 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), append_25451, *[subscript_call_result_25456], **kwargs_25457)
        
        # SSA branch for the else part of an if statement (line 110)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'value' (line 113)
        value_25462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'value', False)
        # Processing the call keyword arguments (line 113)
        kwargs_25463 = {}
        # Getting the type of 'self' (line 113)
        self_25459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'self', False)
        # Obtaining the member 'locs' of a type (line 113)
        locs_25460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), self_25459, 'locs')
        # Obtaining the member 'append' of a type (line 113)
        append_25461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), locs_25460, 'append')
        # Calling append(args, kwargs) (line 113)
        append_call_result_25464 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), append_25461, *[value_25462], **kwargs_25463)
        
        
        # Getting the type of 'value' (line 114)
        value_25465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'value')
        int_25466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'int')
        # Applying the binary operator '+=' (line 114)
        result_iadd_25467 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 16), '+=', value_25465, int_25466)
        # Assigning a type to the variable 'value' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'value', result_iadd_25467)
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_set_seq_locs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_seq_locs' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_25468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_seq_locs'
        return stypy_return_type_25468


# Assigning a type to the variable 'UnitData' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'UnitData', UnitData)

# Assigning a Dict to a Name (line 86):

# Obtaining an instance of the builtin type 'dict' (line 86)
dict_25469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 86)
# Adding element type (key, value) (line 86)
unicode_25470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 14), 'unicode', u'nan')
float_25471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), dict_25469, (unicode_25470, float_25471))
# Adding element type (key, value) (line 86)
unicode_25472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'unicode', u'inf')
float_25473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), dict_25469, (unicode_25472, float_25473))
# Adding element type (key, value) (line 86)
unicode_25474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 40), 'unicode', u'-inf')
float_25475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 48), 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 13), dict_25469, (unicode_25474, float_25475))

# Getting the type of 'UnitData'
UnitData_25476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnitData')
# Setting the type of the member 'spdict' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnitData_25476, 'spdict', dict_25469)

# Assigning a Call to a Subscript (line 118):

# Assigning a Call to a Subscript (line 118):

# Call to StrCategoryConverter(...): (line 118)
# Processing the call keyword arguments (line 118)
kwargs_25478 = {}
# Getting the type of 'StrCategoryConverter' (line 118)
StrCategoryConverter_25477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'StrCategoryConverter', False)
# Calling StrCategoryConverter(args, kwargs) (line 118)
StrCategoryConverter_call_result_25479 = invoke(stypy.reporting.localization.Localization(__file__, 118, 22), StrCategoryConverter_25477, *[], **kwargs_25478)

# Getting the type of 'units' (line 118)
units_25480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'units')
# Obtaining the member 'registry' of a type (line 118)
registry_25481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 0), units_25480, 'registry')
# Getting the type of 'str' (line 118)
str_25482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'str')
# Storing an element on a container (line 118)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 0), registry_25481, (str_25482, StrCategoryConverter_call_result_25479))

# Assigning a Call to a Subscript (line 119):

# Assigning a Call to a Subscript (line 119):

# Call to StrCategoryConverter(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_25484 = {}
# Getting the type of 'StrCategoryConverter' (line 119)
StrCategoryConverter_25483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'StrCategoryConverter', False)
# Calling StrCategoryConverter(args, kwargs) (line 119)
StrCategoryConverter_call_result_25485 = invoke(stypy.reporting.localization.Localization(__file__, 119, 26), StrCategoryConverter_25483, *[], **kwargs_25484)

# Getting the type of 'units' (line 119)
units_25486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'units')
# Obtaining the member 'registry' of a type (line 119)
registry_25487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 0), units_25486, 'registry')
# Getting the type of 'np' (line 119)
np_25488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'np')
# Obtaining the member 'str_' of a type (line 119)
str__25489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), np_25488, 'str_')
# Storing an element on a container (line 119)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 0), registry_25487, (str__25489, StrCategoryConverter_call_result_25485))

# Assigning a Call to a Subscript (line 120):

# Assigning a Call to a Subscript (line 120):

# Call to StrCategoryConverter(...): (line 120)
# Processing the call keyword arguments (line 120)
kwargs_25491 = {}
# Getting the type of 'StrCategoryConverter' (line 120)
StrCategoryConverter_25490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'StrCategoryConverter', False)
# Calling StrCategoryConverter(args, kwargs) (line 120)
StrCategoryConverter_call_result_25492 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), StrCategoryConverter_25490, *[], **kwargs_25491)

# Getting the type of 'units' (line 120)
units_25493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'units')
# Obtaining the member 'registry' of a type (line 120)
registry_25494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 0), units_25493, 'registry')
# Getting the type of 'six' (line 120)
six_25495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'six')
# Obtaining the member 'text_type' of a type (line 120)
text_type_25496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 15), six_25495, 'text_type')
# Storing an element on a container (line 120)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 0), registry_25494, (text_type_25496, StrCategoryConverter_call_result_25492))

# Assigning a Call to a Subscript (line 121):

# Assigning a Call to a Subscript (line 121):

# Call to StrCategoryConverter(...): (line 121)
# Processing the call keyword arguments (line 121)
kwargs_25498 = {}
# Getting the type of 'StrCategoryConverter' (line 121)
StrCategoryConverter_25497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'StrCategoryConverter', False)
# Calling StrCategoryConverter(args, kwargs) (line 121)
StrCategoryConverter_call_result_25499 = invoke(stypy.reporting.localization.Localization(__file__, 121, 24), StrCategoryConverter_25497, *[], **kwargs_25498)

# Getting the type of 'units' (line 121)
units_25500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'units')
# Obtaining the member 'registry' of a type (line 121)
registry_25501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 0), units_25500, 'registry')
# Getting the type of 'bytes' (line 121)
bytes_25502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'bytes')
# Storing an element on a container (line 121)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 0), registry_25501, (bytes_25502, StrCategoryConverter_call_result_25499))

# Assigning a Call to a Subscript (line 122):

# Assigning a Call to a Subscript (line 122):

# Call to StrCategoryConverter(...): (line 122)
# Processing the call keyword arguments (line 122)
kwargs_25504 = {}
# Getting the type of 'StrCategoryConverter' (line 122)
StrCategoryConverter_25503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'StrCategoryConverter', False)
# Calling StrCategoryConverter(args, kwargs) (line 122)
StrCategoryConverter_call_result_25505 = invoke(stypy.reporting.localization.Localization(__file__, 122, 28), StrCategoryConverter_25503, *[], **kwargs_25504)

# Getting the type of 'units' (line 122)
units_25506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'units')
# Obtaining the member 'registry' of a type (line 122)
registry_25507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 0), units_25506, 'registry')
# Getting the type of 'np' (line 122)
np_25508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'np')
# Obtaining the member 'bytes_' of a type (line 122)
bytes__25509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 15), np_25508, 'bytes_')
# Storing an element on a container (line 122)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 0), registry_25507, (bytes__25509, StrCategoryConverter_call_result_25505))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
