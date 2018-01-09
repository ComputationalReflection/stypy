
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Jottings to work out format for __function_workspace__ matrix at end
2: of mat file.
3: 
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: import os.path
8: import sys
9: import io
10: 
11: from numpy.compat import asstr
12: 
13: from scipy.io.matlab.mio5 import (MatlabObject, MatFile5Writer,
14:                                   MatFile5Reader, MatlabFunction)
15: 
16: test_data_path = os.path.join(os.path.dirname(__file__), 'data')
17: 
18: 
19: def read_minimat_vars(rdr):
20:     rdr.initialize_read()
21:     mdict = {'__globals__': []}
22:     i = 0
23:     while not rdr.end_of_stream():
24:         hdr, next_position = rdr.read_var_header()
25:         name = asstr(hdr.name)
26:         if name == '':
27:             name = 'var_%d' % i
28:             i += 1
29:         res = rdr.read_var_array(hdr, process=False)
30:         rdr.mat_stream.seek(next_position)
31:         mdict[name] = res
32:         if hdr.is_global:
33:             mdict['__globals__'].append(name)
34:     return mdict
35: 
36: 
37: def read_workspace_vars(fname):
38:     fp = open(fname, 'rb')
39:     rdr = MatFile5Reader(fp, struct_as_record=True)
40:     vars = rdr.get_variables()
41:     fws = vars['__function_workspace__']
42:     ws_bs = io.BytesIO(fws.tostring())
43:     ws_bs.seek(2)
44:     rdr.mat_stream = ws_bs
45:     # Guess byte order.
46:     mi = rdr.mat_stream.read(2)
47:     rdr.byte_order = mi == b'IM' and '<' or '>'
48:     rdr.mat_stream.read(4)  # presumably byte padding
49:     mdict = read_minimat_vars(rdr)
50:     fp.close()
51:     return mdict
52: 
53: 
54: def test_jottings():
55:     # example
56:     fname = os.path.join(test_data_path, 'parabola.mat')
57:     ws_vars = read_workspace_vars(fname)
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_144214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', ' Jottings to work out format for __function_workspace__ matrix at end\nof mat file.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os.path' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144215 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path')

if (type(import_144215) is not StypyTypeError):

    if (import_144215 != 'pyd_module'):
        __import__(import_144215)
        sys_modules_144216 = sys.modules[import_144215]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', sys_modules_144216.module_type_store, module_type_store)
    else:
        import os.path

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', os.path, module_type_store)

else:
    # Assigning a type to the variable 'os.path' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', import_144215)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import io' statement (line 9)
import io

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'io', io, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.compat import asstr' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat')

if (type(import_144217) is not StypyTypeError):

    if (import_144217 != 'pyd_module'):
        __import__(import_144217)
        sys_modules_144218 = sys.modules[import_144217]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', sys_modules_144218.module_type_store, module_type_store, ['asstr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_144218, sys_modules_144218.module_type_store, module_type_store)
    else:
        from numpy.compat import asstr

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', None, module_type_store, ['asstr'], [asstr])

else:
    # Assigning a type to the variable 'numpy.compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.compat', import_144217)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.io.matlab.mio5 import MatlabObject, MatFile5Writer, MatFile5Reader, MatlabFunction' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144219 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio5')

if (type(import_144219) is not StypyTypeError):

    if (import_144219 != 'pyd_module'):
        __import__(import_144219)
        sys_modules_144220 = sys.modules[import_144219]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio5', sys_modules_144220.module_type_store, module_type_store, ['MatlabObject', 'MatFile5Writer', 'MatFile5Reader', 'MatlabFunction'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_144220, sys_modules_144220.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio5 import MatlabObject, MatFile5Writer, MatFile5Reader, MatlabFunction

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio5', None, module_type_store, ['MatlabObject', 'MatFile5Writer', 'MatFile5Reader', 'MatlabFunction'], [MatlabObject, MatFile5Writer, MatFile5Reader, MatlabFunction])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio5' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.matlab.mio5', import_144219)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


# Assigning a Call to a Name (line 16):

# Assigning a Call to a Name (line 16):

# Call to join(...): (line 16)
# Processing the call arguments (line 16)

# Call to dirname(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of '__file__' (line 16)
file___144227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 46), '__file__', False)
# Processing the call keyword arguments (line 16)
kwargs_144228 = {}
# Getting the type of 'os' (line 16)
os_144224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 30), 'os', False)
# Obtaining the member 'path' of a type (line 16)
path_144225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 30), os_144224, 'path')
# Obtaining the member 'dirname' of a type (line 16)
dirname_144226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 30), path_144225, 'dirname')
# Calling dirname(args, kwargs) (line 16)
dirname_call_result_144229 = invoke(stypy.reporting.localization.Localization(__file__, 16, 30), dirname_144226, *[file___144227], **kwargs_144228)

str_144230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 57), 'str', 'data')
# Processing the call keyword arguments (line 16)
kwargs_144231 = {}
# Getting the type of 'os' (line 16)
os_144221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'os', False)
# Obtaining the member 'path' of a type (line 16)
path_144222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), os_144221, 'path')
# Obtaining the member 'join' of a type (line 16)
join_144223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 17), path_144222, 'join')
# Calling join(args, kwargs) (line 16)
join_call_result_144232 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), join_144223, *[dirname_call_result_144229, str_144230], **kwargs_144231)

# Assigning a type to the variable 'test_data_path' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'test_data_path', join_call_result_144232)

@norecursion
def read_minimat_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_minimat_vars'
    module_type_store = module_type_store.open_function_context('read_minimat_vars', 19, 0, False)
    
    # Passed parameters checking function
    read_minimat_vars.stypy_localization = localization
    read_minimat_vars.stypy_type_of_self = None
    read_minimat_vars.stypy_type_store = module_type_store
    read_minimat_vars.stypy_function_name = 'read_minimat_vars'
    read_minimat_vars.stypy_param_names_list = ['rdr']
    read_minimat_vars.stypy_varargs_param_name = None
    read_minimat_vars.stypy_kwargs_param_name = None
    read_minimat_vars.stypy_call_defaults = defaults
    read_minimat_vars.stypy_call_varargs = varargs
    read_minimat_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_minimat_vars', ['rdr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_minimat_vars', localization, ['rdr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_minimat_vars(...)' code ##################

    
    # Call to initialize_read(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_144235 = {}
    # Getting the type of 'rdr' (line 20)
    rdr_144233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'rdr', False)
    # Obtaining the member 'initialize_read' of a type (line 20)
    initialize_read_144234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), rdr_144233, 'initialize_read')
    # Calling initialize_read(args, kwargs) (line 20)
    initialize_read_call_result_144236 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), initialize_read_144234, *[], **kwargs_144235)
    
    
    # Assigning a Dict to a Name (line 21):
    
    # Assigning a Dict to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'dict' (line 21)
    dict_144237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 21)
    # Adding element type (key, value) (line 21)
    str_144238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 13), 'str', '__globals__')
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_144239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), dict_144237, (str_144238, list_144239))
    
    # Assigning a type to the variable 'mdict' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'mdict', dict_144237)
    
    # Assigning a Num to a Name (line 22):
    
    # Assigning a Num to a Name (line 22):
    int_144240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'int')
    # Assigning a type to the variable 'i' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'i', int_144240)
    
    
    
    # Call to end_of_stream(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_144243 = {}
    # Getting the type of 'rdr' (line 23)
    rdr_144241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'rdr', False)
    # Obtaining the member 'end_of_stream' of a type (line 23)
    end_of_stream_144242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 14), rdr_144241, 'end_of_stream')
    # Calling end_of_stream(args, kwargs) (line 23)
    end_of_stream_call_result_144244 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), end_of_stream_144242, *[], **kwargs_144243)
    
    # Applying the 'not' unary operator (line 23)
    result_not__144245 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 10), 'not', end_of_stream_call_result_144244)
    
    # Testing the type of an if condition (line 23)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_not__144245)
    # SSA begins for while statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 24):
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_144246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'int')
    
    # Call to read_var_header(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_144249 = {}
    # Getting the type of 'rdr' (line 24)
    rdr_144247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'rdr', False)
    # Obtaining the member 'read_var_header' of a type (line 24)
    read_var_header_144248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), rdr_144247, 'read_var_header')
    # Calling read_var_header(args, kwargs) (line 24)
    read_var_header_call_result_144250 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), read_var_header_144248, *[], **kwargs_144249)
    
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___144251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), read_var_header_call_result_144250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_144252 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), getitem___144251, int_144246)
    
    # Assigning a type to the variable 'tuple_var_assignment_144212' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_var_assignment_144212', subscript_call_result_144252)
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_144253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'int')
    
    # Call to read_var_header(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_144256 = {}
    # Getting the type of 'rdr' (line 24)
    rdr_144254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'rdr', False)
    # Obtaining the member 'read_var_header' of a type (line 24)
    read_var_header_144255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), rdr_144254, 'read_var_header')
    # Calling read_var_header(args, kwargs) (line 24)
    read_var_header_call_result_144257 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), read_var_header_144255, *[], **kwargs_144256)
    
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___144258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), read_var_header_call_result_144257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_144259 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), getitem___144258, int_144253)
    
    # Assigning a type to the variable 'tuple_var_assignment_144213' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_var_assignment_144213', subscript_call_result_144259)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_144212' (line 24)
    tuple_var_assignment_144212_144260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_var_assignment_144212')
    # Assigning a type to the variable 'hdr' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'hdr', tuple_var_assignment_144212_144260)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_144213' (line 24)
    tuple_var_assignment_144213_144261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'tuple_var_assignment_144213')
    # Assigning a type to the variable 'next_position' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'next_position', tuple_var_assignment_144213_144261)
    
    # Assigning a Call to a Name (line 25):
    
    # Assigning a Call to a Name (line 25):
    
    # Call to asstr(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'hdr' (line 25)
    hdr_144263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'hdr', False)
    # Obtaining the member 'name' of a type (line 25)
    name_144264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 21), hdr_144263, 'name')
    # Processing the call keyword arguments (line 25)
    kwargs_144265 = {}
    # Getting the type of 'asstr' (line 25)
    asstr_144262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'asstr', False)
    # Calling asstr(args, kwargs) (line 25)
    asstr_call_result_144266 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), asstr_144262, *[name_144264], **kwargs_144265)
    
    # Assigning a type to the variable 'name' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'name', asstr_call_result_144266)
    
    
    # Getting the type of 'name' (line 26)
    name_144267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'name')
    str_144268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', '')
    # Applying the binary operator '==' (line 26)
    result_eq_144269 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), '==', name_144267, str_144268)
    
    # Testing the type of an if condition (line 26)
    if_condition_144270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), result_eq_144269)
    # Assigning a type to the variable 'if_condition_144270' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_144270', if_condition_144270)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 27):
    
    # Assigning a BinOp to a Name (line 27):
    str_144271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'var_%d')
    # Getting the type of 'i' (line 27)
    i_144272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'i')
    # Applying the binary operator '%' (line 27)
    result_mod_144273 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 19), '%', str_144271, i_144272)
    
    # Assigning a type to the variable 'name' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'name', result_mod_144273)
    
    # Getting the type of 'i' (line 28)
    i_144274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'i')
    int_144275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'int')
    # Applying the binary operator '+=' (line 28)
    result_iadd_144276 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '+=', i_144274, int_144275)
    # Assigning a type to the variable 'i' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'i', result_iadd_144276)
    
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to read_var_array(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'hdr' (line 29)
    hdr_144279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'hdr', False)
    # Processing the call keyword arguments (line 29)
    # Getting the type of 'False' (line 29)
    False_144280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 46), 'False', False)
    keyword_144281 = False_144280
    kwargs_144282 = {'process': keyword_144281}
    # Getting the type of 'rdr' (line 29)
    rdr_144277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'rdr', False)
    # Obtaining the member 'read_var_array' of a type (line 29)
    read_var_array_144278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 14), rdr_144277, 'read_var_array')
    # Calling read_var_array(args, kwargs) (line 29)
    read_var_array_call_result_144283 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), read_var_array_144278, *[hdr_144279], **kwargs_144282)
    
    # Assigning a type to the variable 'res' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'res', read_var_array_call_result_144283)
    
    # Call to seek(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'next_position' (line 30)
    next_position_144287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'next_position', False)
    # Processing the call keyword arguments (line 30)
    kwargs_144288 = {}
    # Getting the type of 'rdr' (line 30)
    rdr_144284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'rdr', False)
    # Obtaining the member 'mat_stream' of a type (line 30)
    mat_stream_144285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), rdr_144284, 'mat_stream')
    # Obtaining the member 'seek' of a type (line 30)
    seek_144286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), mat_stream_144285, 'seek')
    # Calling seek(args, kwargs) (line 30)
    seek_call_result_144289 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), seek_144286, *[next_position_144287], **kwargs_144288)
    
    
    # Assigning a Name to a Subscript (line 31):
    
    # Assigning a Name to a Subscript (line 31):
    # Getting the type of 'res' (line 31)
    res_144290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'res')
    # Getting the type of 'mdict' (line 31)
    mdict_144291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'mdict')
    # Getting the type of 'name' (line 31)
    name_144292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'name')
    # Storing an element on a container (line 31)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 8), mdict_144291, (name_144292, res_144290))
    
    # Getting the type of 'hdr' (line 32)
    hdr_144293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'hdr')
    # Obtaining the member 'is_global' of a type (line 32)
    is_global_144294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 11), hdr_144293, 'is_global')
    # Testing the type of an if condition (line 32)
    if_condition_144295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), is_global_144294)
    # Assigning a type to the variable 'if_condition_144295' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_144295', if_condition_144295)
    # SSA begins for if statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'name' (line 33)
    name_144301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'name', False)
    # Processing the call keyword arguments (line 33)
    kwargs_144302 = {}
    
    # Obtaining the type of the subscript
    str_144296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', '__globals__')
    # Getting the type of 'mdict' (line 33)
    mdict_144297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'mdict', False)
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___144298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), mdict_144297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_144299 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), getitem___144298, str_144296)
    
    # Obtaining the member 'append' of a type (line 33)
    append_144300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), subscript_call_result_144299, 'append')
    # Calling append(args, kwargs) (line 33)
    append_call_result_144303 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), append_144300, *[name_144301], **kwargs_144302)
    
    # SSA join for if statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'mdict' (line 34)
    mdict_144304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'mdict')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', mdict_144304)
    
    # ################# End of 'read_minimat_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_minimat_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_144305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_minimat_vars'
    return stypy_return_type_144305

# Assigning a type to the variable 'read_minimat_vars' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'read_minimat_vars', read_minimat_vars)

@norecursion
def read_workspace_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_workspace_vars'
    module_type_store = module_type_store.open_function_context('read_workspace_vars', 37, 0, False)
    
    # Passed parameters checking function
    read_workspace_vars.stypy_localization = localization
    read_workspace_vars.stypy_type_of_self = None
    read_workspace_vars.stypy_type_store = module_type_store
    read_workspace_vars.stypy_function_name = 'read_workspace_vars'
    read_workspace_vars.stypy_param_names_list = ['fname']
    read_workspace_vars.stypy_varargs_param_name = None
    read_workspace_vars.stypy_kwargs_param_name = None
    read_workspace_vars.stypy_call_defaults = defaults
    read_workspace_vars.stypy_call_varargs = varargs
    read_workspace_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_workspace_vars', ['fname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_workspace_vars', localization, ['fname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_workspace_vars(...)' code ##################

    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to open(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'fname' (line 38)
    fname_144307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'fname', False)
    str_144308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'rb')
    # Processing the call keyword arguments (line 38)
    kwargs_144309 = {}
    # Getting the type of 'open' (line 38)
    open_144306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'open', False)
    # Calling open(args, kwargs) (line 38)
    open_call_result_144310 = invoke(stypy.reporting.localization.Localization(__file__, 38, 9), open_144306, *[fname_144307, str_144308], **kwargs_144309)
    
    # Assigning a type to the variable 'fp' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'fp', open_call_result_144310)
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to MatFile5Reader(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'fp' (line 39)
    fp_144312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'fp', False)
    # Processing the call keyword arguments (line 39)
    # Getting the type of 'True' (line 39)
    True_144313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'True', False)
    keyword_144314 = True_144313
    kwargs_144315 = {'struct_as_record': keyword_144314}
    # Getting the type of 'MatFile5Reader' (line 39)
    MatFile5Reader_144311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'MatFile5Reader', False)
    # Calling MatFile5Reader(args, kwargs) (line 39)
    MatFile5Reader_call_result_144316 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), MatFile5Reader_144311, *[fp_144312], **kwargs_144315)
    
    # Assigning a type to the variable 'rdr' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'rdr', MatFile5Reader_call_result_144316)
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to get_variables(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_144319 = {}
    # Getting the type of 'rdr' (line 40)
    rdr_144317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'rdr', False)
    # Obtaining the member 'get_variables' of a type (line 40)
    get_variables_144318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 11), rdr_144317, 'get_variables')
    # Calling get_variables(args, kwargs) (line 40)
    get_variables_call_result_144320 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), get_variables_144318, *[], **kwargs_144319)
    
    # Assigning a type to the variable 'vars' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'vars', get_variables_call_result_144320)
    
    # Assigning a Subscript to a Name (line 41):
    
    # Assigning a Subscript to a Name (line 41):
    
    # Obtaining the type of the subscript
    str_144321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', '__function_workspace__')
    # Getting the type of 'vars' (line 41)
    vars_144322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 10), 'vars')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___144323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 10), vars_144322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_144324 = invoke(stypy.reporting.localization.Localization(__file__, 41, 10), getitem___144323, str_144321)
    
    # Assigning a type to the variable 'fws' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'fws', subscript_call_result_144324)
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to BytesIO(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to tostring(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_144329 = {}
    # Getting the type of 'fws' (line 42)
    fws_144327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'fws', False)
    # Obtaining the member 'tostring' of a type (line 42)
    tostring_144328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), fws_144327, 'tostring')
    # Calling tostring(args, kwargs) (line 42)
    tostring_call_result_144330 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), tostring_144328, *[], **kwargs_144329)
    
    # Processing the call keyword arguments (line 42)
    kwargs_144331 = {}
    # Getting the type of 'io' (line 42)
    io_144325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'io', False)
    # Obtaining the member 'BytesIO' of a type (line 42)
    BytesIO_144326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), io_144325, 'BytesIO')
    # Calling BytesIO(args, kwargs) (line 42)
    BytesIO_call_result_144332 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), BytesIO_144326, *[tostring_call_result_144330], **kwargs_144331)
    
    # Assigning a type to the variable 'ws_bs' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'ws_bs', BytesIO_call_result_144332)
    
    # Call to seek(...): (line 43)
    # Processing the call arguments (line 43)
    int_144335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'int')
    # Processing the call keyword arguments (line 43)
    kwargs_144336 = {}
    # Getting the type of 'ws_bs' (line 43)
    ws_bs_144333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'ws_bs', False)
    # Obtaining the member 'seek' of a type (line 43)
    seek_144334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), ws_bs_144333, 'seek')
    # Calling seek(args, kwargs) (line 43)
    seek_call_result_144337 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), seek_144334, *[int_144335], **kwargs_144336)
    
    
    # Assigning a Name to a Attribute (line 44):
    
    # Assigning a Name to a Attribute (line 44):
    # Getting the type of 'ws_bs' (line 44)
    ws_bs_144338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'ws_bs')
    # Getting the type of 'rdr' (line 44)
    rdr_144339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'rdr')
    # Setting the type of the member 'mat_stream' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), rdr_144339, 'mat_stream', ws_bs_144338)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to read(...): (line 46)
    # Processing the call arguments (line 46)
    int_144343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'int')
    # Processing the call keyword arguments (line 46)
    kwargs_144344 = {}
    # Getting the type of 'rdr' (line 46)
    rdr_144340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'rdr', False)
    # Obtaining the member 'mat_stream' of a type (line 46)
    mat_stream_144341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 9), rdr_144340, 'mat_stream')
    # Obtaining the member 'read' of a type (line 46)
    read_144342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 9), mat_stream_144341, 'read')
    # Calling read(args, kwargs) (line 46)
    read_call_result_144345 = invoke(stypy.reporting.localization.Localization(__file__, 46, 9), read_144342, *[int_144343], **kwargs_144344)
    
    # Assigning a type to the variable 'mi' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'mi', read_call_result_144345)
    
    # Assigning a BoolOp to a Attribute (line 47):
    
    # Assigning a BoolOp to a Attribute (line 47):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mi' (line 47)
    mi_144346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'mi')
    str_144347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'str', 'IM')
    # Applying the binary operator '==' (line 47)
    result_eq_144348 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 21), '==', mi_144346, str_144347)
    
    str_144349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'str', '<')
    # Applying the binary operator 'and' (line 47)
    result_and_keyword_144350 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 21), 'and', result_eq_144348, str_144349)
    
    str_144351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 44), 'str', '>')
    # Applying the binary operator 'or' (line 47)
    result_or_keyword_144352 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 21), 'or', result_and_keyword_144350, str_144351)
    
    # Getting the type of 'rdr' (line 47)
    rdr_144353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'rdr')
    # Setting the type of the member 'byte_order' of a type (line 47)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), rdr_144353, 'byte_order', result_or_keyword_144352)
    
    # Call to read(...): (line 48)
    # Processing the call arguments (line 48)
    int_144357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 24), 'int')
    # Processing the call keyword arguments (line 48)
    kwargs_144358 = {}
    # Getting the type of 'rdr' (line 48)
    rdr_144354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'rdr', False)
    # Obtaining the member 'mat_stream' of a type (line 48)
    mat_stream_144355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), rdr_144354, 'mat_stream')
    # Obtaining the member 'read' of a type (line 48)
    read_144356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), mat_stream_144355, 'read')
    # Calling read(args, kwargs) (line 48)
    read_call_result_144359 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), read_144356, *[int_144357], **kwargs_144358)
    
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to read_minimat_vars(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'rdr' (line 49)
    rdr_144361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'rdr', False)
    # Processing the call keyword arguments (line 49)
    kwargs_144362 = {}
    # Getting the type of 'read_minimat_vars' (line 49)
    read_minimat_vars_144360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'read_minimat_vars', False)
    # Calling read_minimat_vars(args, kwargs) (line 49)
    read_minimat_vars_call_result_144363 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), read_minimat_vars_144360, *[rdr_144361], **kwargs_144362)
    
    # Assigning a type to the variable 'mdict' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'mdict', read_minimat_vars_call_result_144363)
    
    # Call to close(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_144366 = {}
    # Getting the type of 'fp' (line 50)
    fp_144364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'fp', False)
    # Obtaining the member 'close' of a type (line 50)
    close_144365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 4), fp_144364, 'close')
    # Calling close(args, kwargs) (line 50)
    close_call_result_144367 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), close_144365, *[], **kwargs_144366)
    
    # Getting the type of 'mdict' (line 51)
    mdict_144368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'mdict')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', mdict_144368)
    
    # ################# End of 'read_workspace_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_workspace_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_144369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_workspace_vars'
    return stypy_return_type_144369

# Assigning a type to the variable 'read_workspace_vars' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'read_workspace_vars', read_workspace_vars)

@norecursion
def test_jottings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_jottings'
    module_type_store = module_type_store.open_function_context('test_jottings', 54, 0, False)
    
    # Passed parameters checking function
    test_jottings.stypy_localization = localization
    test_jottings.stypy_type_of_self = None
    test_jottings.stypy_type_store = module_type_store
    test_jottings.stypy_function_name = 'test_jottings'
    test_jottings.stypy_param_names_list = []
    test_jottings.stypy_varargs_param_name = None
    test_jottings.stypy_kwargs_param_name = None
    test_jottings.stypy_call_defaults = defaults
    test_jottings.stypy_call_varargs = varargs
    test_jottings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_jottings', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_jottings', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_jottings(...)' code ##################

    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to join(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'test_data_path' (line 56)
    test_data_path_144373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'test_data_path', False)
    str_144374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 41), 'str', 'parabola.mat')
    # Processing the call keyword arguments (line 56)
    kwargs_144375 = {}
    # Getting the type of 'os' (line 56)
    os_144370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'os', False)
    # Obtaining the member 'path' of a type (line 56)
    path_144371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), os_144370, 'path')
    # Obtaining the member 'join' of a type (line 56)
    join_144372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), path_144371, 'join')
    # Calling join(args, kwargs) (line 56)
    join_call_result_144376 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), join_144372, *[test_data_path_144373, str_144374], **kwargs_144375)
    
    # Assigning a type to the variable 'fname' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'fname', join_call_result_144376)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to read_workspace_vars(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'fname' (line 57)
    fname_144378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'fname', False)
    # Processing the call keyword arguments (line 57)
    kwargs_144379 = {}
    # Getting the type of 'read_workspace_vars' (line 57)
    read_workspace_vars_144377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'read_workspace_vars', False)
    # Calling read_workspace_vars(args, kwargs) (line 57)
    read_workspace_vars_call_result_144380 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), read_workspace_vars_144377, *[fname_144378], **kwargs_144379)
    
    # Assigning a type to the variable 'ws_vars' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'ws_vars', read_workspace_vars_call_result_144380)
    
    # ################# End of 'test_jottings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_jottings' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_144381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144381)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_jottings'
    return stypy_return_type_144381

# Assigning a type to the variable 'test_jottings' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test_jottings', test_jottings)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
