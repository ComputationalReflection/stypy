
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import scipy.sparse.linalg as la
4: import scipy.sparse as sp
5: import scipy.io as io
6: import numpy as np
7: import sys
8: 
9: #problem = "SPARSKIT/drivcav/e05r0100"
10: problem = "SPARSKIT/drivcav/e05r0200"
11: #problem = "Harwell-Boeing/sherman/sherman1"
12: #problem = "misc/hamm/add32"
13: 
14: mm = np.lib._datasource.Repository('ftp://math.nist.gov/pub/MatrixMarket2/')
15: f = mm.open('%s.mtx.gz' % problem)
16: Am = io.mmread(f).tocsr()
17: f.close()
18: 
19: f = mm.open('%s_rhs1.mtx.gz' % problem)
20: b = np.array(io.mmread(f)).ravel()
21: f.close()
22: 
23: count = [0]
24: 
25: 
26: def matvec(v):
27:     count[0] += 1
28:     sys.stderr.write('%d\r' % count[0])
29:     return Am*v
30: A = la.LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)
31: 
32: M = 100
33: 
34: print("MatrixMarket problem %s" % problem)
35: print("Invert %d x %d matrix; nnz = %d" % (Am.shape[0], Am.shape[1], Am.nnz))
36: 
37: count[0] = 0
38: x0, info = la.gmres(A, b, restrt=M, tol=1e-14)
39: count_0 = count[0]
40: err0 = np.linalg.norm(Am*x0 - b) / np.linalg.norm(b)
41: print("GMRES(%d):" % M, count_0, "matvecs, residual", err0)
42: if info != 0:
43:     print("Didn't converge")
44: 
45: count[0] = 0
46: x1, info = la.lgmres(A, b, inner_m=M-6*2, outer_k=6, tol=1e-14)
47: count_1 = count[0]
48: err1 = np.linalg.norm(Am*x1 - b) / np.linalg.norm(b)
49: print("LGMRES(%d,6) [same memory req.]:" % (M-2*6), count_1,
50:       "matvecs, residual:", err1)
51: if info != 0:
52:     print("Didn't converge")
53: 
54: count[0] = 0
55: x2, info = la.lgmres(A, b, inner_m=M-6, outer_k=6, tol=1e-14)
56: count_2 = count[0]
57: err2 = np.linalg.norm(Am*x2 - b) / np.linalg.norm(b)
58: print("LGMRES(%d,6) [same subspace size]:" % (M-6), count_2,
59:       "matvecs, residual:", err2)
60: if info != 0:
61:     print("Didn't converge")
62: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import scipy.sparse.linalg' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.sparse.linalg')

if (type(import_416387) is not StypyTypeError):

    if (import_416387 != 'pyd_module'):
        __import__(import_416387)
        sys_modules_416388 = sys.modules[import_416387]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'la', sys_modules_416388.module_type_store, module_type_store)
    else:
        import scipy.sparse.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'la', scipy.sparse.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse.linalg' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.sparse.linalg', import_416387)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import scipy.sparse' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416389 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse')

if (type(import_416389) is not StypyTypeError):

    if (import_416389 != 'pyd_module'):
        __import__(import_416389)
        sys_modules_416390 = sys.modules[import_416389]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sp', sys_modules_416390.module_type_store, module_type_store)
    else:
        import scipy.sparse as sp

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sp', scipy.sparse, module_type_store)

else:
    # Assigning a type to the variable 'scipy.sparse' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.sparse', import_416389)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.io' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416391 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.io')

if (type(import_416391) is not StypyTypeError):

    if (import_416391 != 'pyd_module'):
        __import__(import_416391)
        sys_modules_416392 = sys.modules[import_416391]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'io', sys_modules_416392.module_type_store, module_type_store)
    else:
        import scipy.io as io

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'io', scipy.io, module_type_store)

else:
    # Assigning a type to the variable 'scipy.io' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.io', import_416391)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')
import_416393 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_416393) is not StypyTypeError):

    if (import_416393 != 'pyd_module'):
        __import__(import_416393)
        sys_modules_416394 = sys.modules[import_416393]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_416394.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_416393)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)


# Assigning a Str to a Name (line 10):

# Assigning a Str to a Name (line 10):
str_416395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'str', 'SPARSKIT/drivcav/e05r0200')
# Assigning a type to the variable 'problem' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'problem', str_416395)

# Assigning a Call to a Name (line 14):

# Assigning a Call to a Name (line 14):

# Call to Repository(...): (line 14)
# Processing the call arguments (line 14)
str_416400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'str', 'ftp://math.nist.gov/pub/MatrixMarket2/')
# Processing the call keyword arguments (line 14)
kwargs_416401 = {}
# Getting the type of 'np' (line 14)
np_416396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'lib' of a type (line 14)
lib_416397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_416396, 'lib')
# Obtaining the member '_datasource' of a type (line 14)
_datasource_416398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), lib_416397, '_datasource')
# Obtaining the member 'Repository' of a type (line 14)
Repository_416399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), _datasource_416398, 'Repository')
# Calling Repository(args, kwargs) (line 14)
Repository_call_result_416402 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), Repository_416399, *[str_416400], **kwargs_416401)

# Assigning a type to the variable 'mm' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'mm', Repository_call_result_416402)

# Assigning a Call to a Name (line 15):

# Assigning a Call to a Name (line 15):

# Call to open(...): (line 15)
# Processing the call arguments (line 15)
str_416405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'str', '%s.mtx.gz')
# Getting the type of 'problem' (line 15)
problem_416406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'problem', False)
# Applying the binary operator '%' (line 15)
result_mod_416407 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 12), '%', str_416405, problem_416406)

# Processing the call keyword arguments (line 15)
kwargs_416408 = {}
# Getting the type of 'mm' (line 15)
mm_416403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'mm', False)
# Obtaining the member 'open' of a type (line 15)
open_416404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), mm_416403, 'open')
# Calling open(args, kwargs) (line 15)
open_call_result_416409 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), open_416404, *[result_mod_416407], **kwargs_416408)

# Assigning a type to the variable 'f' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'f', open_call_result_416409)

# Assigning a Call to a Name (line 16):

# Assigning a Call to a Name (line 16):

# Call to tocsr(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_416416 = {}

# Call to mmread(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'f' (line 16)
f_416412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'f', False)
# Processing the call keyword arguments (line 16)
kwargs_416413 = {}
# Getting the type of 'io' (line 16)
io_416410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'io', False)
# Obtaining the member 'mmread' of a type (line 16)
mmread_416411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), io_416410, 'mmread')
# Calling mmread(args, kwargs) (line 16)
mmread_call_result_416414 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), mmread_416411, *[f_416412], **kwargs_416413)

# Obtaining the member 'tocsr' of a type (line 16)
tocsr_416415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), mmread_call_result_416414, 'tocsr')
# Calling tocsr(args, kwargs) (line 16)
tocsr_call_result_416417 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), tocsr_416415, *[], **kwargs_416416)

# Assigning a type to the variable 'Am' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'Am', tocsr_call_result_416417)

# Call to close(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_416420 = {}
# Getting the type of 'f' (line 17)
f_416418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'f', False)
# Obtaining the member 'close' of a type (line 17)
close_416419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 0), f_416418, 'close')
# Calling close(args, kwargs) (line 17)
close_call_result_416421 = invoke(stypy.reporting.localization.Localization(__file__, 17, 0), close_416419, *[], **kwargs_416420)


# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to open(...): (line 19)
# Processing the call arguments (line 19)
str_416424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'str', '%s_rhs1.mtx.gz')
# Getting the type of 'problem' (line 19)
problem_416425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'problem', False)
# Applying the binary operator '%' (line 19)
result_mod_416426 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 12), '%', str_416424, problem_416425)

# Processing the call keyword arguments (line 19)
kwargs_416427 = {}
# Getting the type of 'mm' (line 19)
mm_416422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'mm', False)
# Obtaining the member 'open' of a type (line 19)
open_416423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), mm_416422, 'open')
# Calling open(args, kwargs) (line 19)
open_call_result_416428 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), open_416423, *[result_mod_416426], **kwargs_416427)

# Assigning a type to the variable 'f' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'f', open_call_result_416428)

# Assigning a Call to a Name (line 20):

# Assigning a Call to a Name (line 20):

# Call to ravel(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_416439 = {}

# Call to array(...): (line 20)
# Processing the call arguments (line 20)

# Call to mmread(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'f' (line 20)
f_416433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'f', False)
# Processing the call keyword arguments (line 20)
kwargs_416434 = {}
# Getting the type of 'io' (line 20)
io_416431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'io', False)
# Obtaining the member 'mmread' of a type (line 20)
mmread_416432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 13), io_416431, 'mmread')
# Calling mmread(args, kwargs) (line 20)
mmread_call_result_416435 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), mmread_416432, *[f_416433], **kwargs_416434)

# Processing the call keyword arguments (line 20)
kwargs_416436 = {}
# Getting the type of 'np' (line 20)
np_416429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'np', False)
# Obtaining the member 'array' of a type (line 20)
array_416430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), np_416429, 'array')
# Calling array(args, kwargs) (line 20)
array_call_result_416437 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), array_416430, *[mmread_call_result_416435], **kwargs_416436)

# Obtaining the member 'ravel' of a type (line 20)
ravel_416438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), array_call_result_416437, 'ravel')
# Calling ravel(args, kwargs) (line 20)
ravel_call_result_416440 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), ravel_416438, *[], **kwargs_416439)

# Assigning a type to the variable 'b' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'b', ravel_call_result_416440)

# Call to close(...): (line 21)
# Processing the call keyword arguments (line 21)
kwargs_416443 = {}
# Getting the type of 'f' (line 21)
f_416441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'f', False)
# Obtaining the member 'close' of a type (line 21)
close_416442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 0), f_416441, 'close')
# Calling close(args, kwargs) (line 21)
close_call_result_416444 = invoke(stypy.reporting.localization.Localization(__file__, 21, 0), close_416442, *[], **kwargs_416443)


# Assigning a List to a Name (line 23):

# Assigning a List to a Name (line 23):

# Obtaining an instance of the builtin type 'list' (line 23)
list_416445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_416446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), list_416445, int_416446)

# Assigning a type to the variable 'count' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'count', list_416445)

@norecursion
def matvec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'matvec'
    module_type_store = module_type_store.open_function_context('matvec', 26, 0, False)
    
    # Passed parameters checking function
    matvec.stypy_localization = localization
    matvec.stypy_type_of_self = None
    matvec.stypy_type_store = module_type_store
    matvec.stypy_function_name = 'matvec'
    matvec.stypy_param_names_list = ['v']
    matvec.stypy_varargs_param_name = None
    matvec.stypy_kwargs_param_name = None
    matvec.stypy_call_defaults = defaults
    matvec.stypy_call_varargs = varargs
    matvec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matvec', ['v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matvec', localization, ['v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matvec(...)' code ##################

    
    # Getting the type of 'count' (line 27)
    count_416447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'count')
    
    # Obtaining the type of the subscript
    int_416448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'int')
    # Getting the type of 'count' (line 27)
    count_416449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'count')
    # Obtaining the member '__getitem__' of a type (line 27)
    getitem___416450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), count_416449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 27)
    subscript_call_result_416451 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), getitem___416450, int_416448)
    
    int_416452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'int')
    # Applying the binary operator '+=' (line 27)
    result_iadd_416453 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 4), '+=', subscript_call_result_416451, int_416452)
    # Getting the type of 'count' (line 27)
    count_416454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'count')
    int_416455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'int')
    # Storing an element on a container (line 27)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), count_416454, (int_416455, result_iadd_416453))
    
    
    # Call to write(...): (line 28)
    # Processing the call arguments (line 28)
    str_416459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'str', '%d\r')
    
    # Obtaining the type of the subscript
    int_416460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 36), 'int')
    # Getting the type of 'count' (line 28)
    count_416461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'count', False)
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___416462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 30), count_416461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_416463 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), getitem___416462, int_416460)
    
    # Applying the binary operator '%' (line 28)
    result_mod_416464 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 21), '%', str_416459, subscript_call_result_416463)
    
    # Processing the call keyword arguments (line 28)
    kwargs_416465 = {}
    # Getting the type of 'sys' (line 28)
    sys_416456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 28)
    stderr_416457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), sys_416456, 'stderr')
    # Obtaining the member 'write' of a type (line 28)
    write_416458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), stderr_416457, 'write')
    # Calling write(args, kwargs) (line 28)
    write_call_result_416466 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), write_416458, *[result_mod_416464], **kwargs_416465)
    
    # Getting the type of 'Am' (line 29)
    Am_416467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'Am')
    # Getting the type of 'v' (line 29)
    v_416468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'v')
    # Applying the binary operator '*' (line 29)
    result_mul_416469 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), '*', Am_416467, v_416468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', result_mul_416469)
    
    # ################# End of 'matvec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matvec' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_416470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matvec'
    return stypy_return_type_416470

# Assigning a type to the variable 'matvec' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'matvec', matvec)

# Assigning a Call to a Name (line 30):

# Assigning a Call to a Name (line 30):

# Call to LinearOperator(...): (line 30)
# Processing the call keyword arguments (line 30)
# Getting the type of 'matvec' (line 30)
matvec_416473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'matvec', False)
keyword_416474 = matvec_416473
# Getting the type of 'Am' (line 30)
Am_416475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'Am', False)
# Obtaining the member 'shape' of a type (line 30)
shape_416476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 43), Am_416475, 'shape')
keyword_416477 = shape_416476
# Getting the type of 'Am' (line 30)
Am_416478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 59), 'Am', False)
# Obtaining the member 'dtype' of a type (line 30)
dtype_416479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 59), Am_416478, 'dtype')
keyword_416480 = dtype_416479
kwargs_416481 = {'dtype': keyword_416480, 'shape': keyword_416477, 'matvec': keyword_416474}
# Getting the type of 'la' (line 30)
la_416471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'la', False)
# Obtaining the member 'LinearOperator' of a type (line 30)
LinearOperator_416472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), la_416471, 'LinearOperator')
# Calling LinearOperator(args, kwargs) (line 30)
LinearOperator_call_result_416482 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), LinearOperator_416472, *[], **kwargs_416481)

# Assigning a type to the variable 'A' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'A', LinearOperator_call_result_416482)

# Assigning a Num to a Name (line 32):

# Assigning a Num to a Name (line 32):
int_416483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'int')
# Assigning a type to the variable 'M' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'M', int_416483)

# Call to print(...): (line 34)
# Processing the call arguments (line 34)
str_416485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 6), 'str', 'MatrixMarket problem %s')
# Getting the type of 'problem' (line 34)
problem_416486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'problem', False)
# Applying the binary operator '%' (line 34)
result_mod_416487 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 6), '%', str_416485, problem_416486)

# Processing the call keyword arguments (line 34)
kwargs_416488 = {}
# Getting the type of 'print' (line 34)
print_416484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'print', False)
# Calling print(args, kwargs) (line 34)
print_call_result_416489 = invoke(stypy.reporting.localization.Localization(__file__, 34, 0), print_416484, *[result_mod_416487], **kwargs_416488)


# Call to print(...): (line 35)
# Processing the call arguments (line 35)
str_416491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 6), 'str', 'Invert %d x %d matrix; nnz = %d')

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_416492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)

# Obtaining the type of the subscript
int_416493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 52), 'int')
# Getting the type of 'Am' (line 35)
Am_416494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'Am', False)
# Obtaining the member 'shape' of a type (line 35)
shape_416495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 43), Am_416494, 'shape')
# Obtaining the member '__getitem__' of a type (line 35)
getitem___416496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 43), shape_416495, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 35)
subscript_call_result_416497 = invoke(stypy.reporting.localization.Localization(__file__, 35, 43), getitem___416496, int_416493)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_416492, subscript_call_result_416497)
# Adding element type (line 35)

# Obtaining the type of the subscript
int_416498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 65), 'int')
# Getting the type of 'Am' (line 35)
Am_416499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 56), 'Am', False)
# Obtaining the member 'shape' of a type (line 35)
shape_416500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 56), Am_416499, 'shape')
# Obtaining the member '__getitem__' of a type (line 35)
getitem___416501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 56), shape_416500, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 35)
subscript_call_result_416502 = invoke(stypy.reporting.localization.Localization(__file__, 35, 56), getitem___416501, int_416498)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_416492, subscript_call_result_416502)
# Adding element type (line 35)
# Getting the type of 'Am' (line 35)
Am_416503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 69), 'Am', False)
# Obtaining the member 'nnz' of a type (line 35)
nnz_416504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 69), Am_416503, 'nnz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 43), tuple_416492, nnz_416504)

# Applying the binary operator '%' (line 35)
result_mod_416505 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 6), '%', str_416491, tuple_416492)

# Processing the call keyword arguments (line 35)
kwargs_416506 = {}
# Getting the type of 'print' (line 35)
print_416490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'print', False)
# Calling print(args, kwargs) (line 35)
print_call_result_416507 = invoke(stypy.reporting.localization.Localization(__file__, 35, 0), print_416490, *[result_mod_416505], **kwargs_416506)


# Assigning a Num to a Subscript (line 37):

# Assigning a Num to a Subscript (line 37):
int_416508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'int')
# Getting the type of 'count' (line 37)
count_416509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'count')
int_416510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 6), 'int')
# Storing an element on a container (line 37)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 0), count_416509, (int_416510, int_416508))

# Assigning a Call to a Tuple (line 38):

# Assigning a Subscript to a Name (line 38):

# Obtaining the type of the subscript
int_416511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 0), 'int')

# Call to gmres(...): (line 38)
# Processing the call arguments (line 38)
# Getting the type of 'A' (line 38)
A_416514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'A', False)
# Getting the type of 'b' (line 38)
b_416515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'b', False)
# Processing the call keyword arguments (line 38)
# Getting the type of 'M' (line 38)
M_416516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'M', False)
keyword_416517 = M_416516
float_416518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'float')
keyword_416519 = float_416518
kwargs_416520 = {'tol': keyword_416519, 'restrt': keyword_416517}
# Getting the type of 'la' (line 38)
la_416512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'la', False)
# Obtaining the member 'gmres' of a type (line 38)
gmres_416513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), la_416512, 'gmres')
# Calling gmres(args, kwargs) (line 38)
gmres_call_result_416521 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), gmres_416513, *[A_416514, b_416515], **kwargs_416520)

# Obtaining the member '__getitem__' of a type (line 38)
getitem___416522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 0), gmres_call_result_416521, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 38)
subscript_call_result_416523 = invoke(stypy.reporting.localization.Localization(__file__, 38, 0), getitem___416522, int_416511)

# Assigning a type to the variable 'tuple_var_assignment_416381' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'tuple_var_assignment_416381', subscript_call_result_416523)

# Assigning a Subscript to a Name (line 38):

# Obtaining the type of the subscript
int_416524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 0), 'int')

# Call to gmres(...): (line 38)
# Processing the call arguments (line 38)
# Getting the type of 'A' (line 38)
A_416527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'A', False)
# Getting the type of 'b' (line 38)
b_416528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'b', False)
# Processing the call keyword arguments (line 38)
# Getting the type of 'M' (line 38)
M_416529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'M', False)
keyword_416530 = M_416529
float_416531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'float')
keyword_416532 = float_416531
kwargs_416533 = {'tol': keyword_416532, 'restrt': keyword_416530}
# Getting the type of 'la' (line 38)
la_416525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'la', False)
# Obtaining the member 'gmres' of a type (line 38)
gmres_416526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), la_416525, 'gmres')
# Calling gmres(args, kwargs) (line 38)
gmres_call_result_416534 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), gmres_416526, *[A_416527, b_416528], **kwargs_416533)

# Obtaining the member '__getitem__' of a type (line 38)
getitem___416535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 0), gmres_call_result_416534, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 38)
subscript_call_result_416536 = invoke(stypy.reporting.localization.Localization(__file__, 38, 0), getitem___416535, int_416524)

# Assigning a type to the variable 'tuple_var_assignment_416382' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'tuple_var_assignment_416382', subscript_call_result_416536)

# Assigning a Name to a Name (line 38):
# Getting the type of 'tuple_var_assignment_416381' (line 38)
tuple_var_assignment_416381_416537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'tuple_var_assignment_416381')
# Assigning a type to the variable 'x0' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'x0', tuple_var_assignment_416381_416537)

# Assigning a Name to a Name (line 38):
# Getting the type of 'tuple_var_assignment_416382' (line 38)
tuple_var_assignment_416382_416538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'tuple_var_assignment_416382')
# Assigning a type to the variable 'info' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'info', tuple_var_assignment_416382_416538)

# Assigning a Subscript to a Name (line 39):

# Assigning a Subscript to a Name (line 39):

# Obtaining the type of the subscript
int_416539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 16), 'int')
# Getting the type of 'count' (line 39)
count_416540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'count')
# Obtaining the member '__getitem__' of a type (line 39)
getitem___416541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 10), count_416540, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 39)
subscript_call_result_416542 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), getitem___416541, int_416539)

# Assigning a type to the variable 'count_0' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'count_0', subscript_call_result_416542)

# Assigning a BinOp to a Name (line 40):

# Assigning a BinOp to a Name (line 40):

# Call to norm(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'Am' (line 40)
Am_416546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'Am', False)
# Getting the type of 'x0' (line 40)
x0_416547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'x0', False)
# Applying the binary operator '*' (line 40)
result_mul_416548 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 22), '*', Am_416546, x0_416547)

# Getting the type of 'b' (line 40)
b_416549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'b', False)
# Applying the binary operator '-' (line 40)
result_sub_416550 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 22), '-', result_mul_416548, b_416549)

# Processing the call keyword arguments (line 40)
kwargs_416551 = {}
# Getting the type of 'np' (line 40)
np_416543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'np', False)
# Obtaining the member 'linalg' of a type (line 40)
linalg_416544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 7), np_416543, 'linalg')
# Obtaining the member 'norm' of a type (line 40)
norm_416545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 7), linalg_416544, 'norm')
# Calling norm(args, kwargs) (line 40)
norm_call_result_416552 = invoke(stypy.reporting.localization.Localization(__file__, 40, 7), norm_416545, *[result_sub_416550], **kwargs_416551)


# Call to norm(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'b' (line 40)
b_416556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 50), 'b', False)
# Processing the call keyword arguments (line 40)
kwargs_416557 = {}
# Getting the type of 'np' (line 40)
np_416553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'np', False)
# Obtaining the member 'linalg' of a type (line 40)
linalg_416554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), np_416553, 'linalg')
# Obtaining the member 'norm' of a type (line 40)
norm_416555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), linalg_416554, 'norm')
# Calling norm(args, kwargs) (line 40)
norm_call_result_416558 = invoke(stypy.reporting.localization.Localization(__file__, 40, 35), norm_416555, *[b_416556], **kwargs_416557)

# Applying the binary operator 'div' (line 40)
result_div_416559 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 7), 'div', norm_call_result_416552, norm_call_result_416558)

# Assigning a type to the variable 'err0' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'err0', result_div_416559)

# Call to print(...): (line 41)
# Processing the call arguments (line 41)
str_416561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 6), 'str', 'GMRES(%d):')
# Getting the type of 'M' (line 41)
M_416562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'M', False)
# Applying the binary operator '%' (line 41)
result_mod_416563 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 6), '%', str_416561, M_416562)

# Getting the type of 'count_0' (line 41)
count_0_416564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'count_0', False)
str_416565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'str', 'matvecs, residual')
# Getting the type of 'err0' (line 41)
err0_416566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'err0', False)
# Processing the call keyword arguments (line 41)
kwargs_416567 = {}
# Getting the type of 'print' (line 41)
print_416560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'print', False)
# Calling print(args, kwargs) (line 41)
print_call_result_416568 = invoke(stypy.reporting.localization.Localization(__file__, 41, 0), print_416560, *[result_mod_416563, count_0_416564, str_416565, err0_416566], **kwargs_416567)



# Getting the type of 'info' (line 42)
info_416569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 3), 'info')
int_416570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'int')
# Applying the binary operator '!=' (line 42)
result_ne_416571 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 3), '!=', info_416569, int_416570)

# Testing the type of an if condition (line 42)
if_condition_416572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 0), result_ne_416571)
# Assigning a type to the variable 'if_condition_416572' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'if_condition_416572', if_condition_416572)
# SSA begins for if statement (line 42)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 43)
# Processing the call arguments (line 43)
str_416574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 10), 'str', "Didn't converge")
# Processing the call keyword arguments (line 43)
kwargs_416575 = {}
# Getting the type of 'print' (line 43)
print_416573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'print', False)
# Calling print(args, kwargs) (line 43)
print_call_result_416576 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), print_416573, *[str_416574], **kwargs_416575)

# SSA join for if statement (line 42)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Num to a Subscript (line 45):

# Assigning a Num to a Subscript (line 45):
int_416577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'int')
# Getting the type of 'count' (line 45)
count_416578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'count')
int_416579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 6), 'int')
# Storing an element on a container (line 45)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 0), count_416578, (int_416579, int_416577))

# Assigning a Call to a Tuple (line 46):

# Assigning a Subscript to a Name (line 46):

# Obtaining the type of the subscript
int_416580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 0), 'int')

# Call to lgmres(...): (line 46)
# Processing the call arguments (line 46)
# Getting the type of 'A' (line 46)
A_416583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'A', False)
# Getting the type of 'b' (line 46)
b_416584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'b', False)
# Processing the call keyword arguments (line 46)
# Getting the type of 'M' (line 46)
M_416585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'M', False)
int_416586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'int')
int_416587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'int')
# Applying the binary operator '*' (line 46)
result_mul_416588 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 37), '*', int_416586, int_416587)

# Applying the binary operator '-' (line 46)
result_sub_416589 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 35), '-', M_416585, result_mul_416588)

keyword_416590 = result_sub_416589
int_416591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 50), 'int')
keyword_416592 = int_416591
float_416593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 57), 'float')
keyword_416594 = float_416593
kwargs_416595 = {'outer_k': keyword_416592, 'tol': keyword_416594, 'inner_m': keyword_416590}
# Getting the type of 'la' (line 46)
la_416581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'la', False)
# Obtaining the member 'lgmres' of a type (line 46)
lgmres_416582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), la_416581, 'lgmres')
# Calling lgmres(args, kwargs) (line 46)
lgmres_call_result_416596 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), lgmres_416582, *[A_416583, b_416584], **kwargs_416595)

# Obtaining the member '__getitem__' of a type (line 46)
getitem___416597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 0), lgmres_call_result_416596, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 46)
subscript_call_result_416598 = invoke(stypy.reporting.localization.Localization(__file__, 46, 0), getitem___416597, int_416580)

# Assigning a type to the variable 'tuple_var_assignment_416383' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'tuple_var_assignment_416383', subscript_call_result_416598)

# Assigning a Subscript to a Name (line 46):

# Obtaining the type of the subscript
int_416599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 0), 'int')

# Call to lgmres(...): (line 46)
# Processing the call arguments (line 46)
# Getting the type of 'A' (line 46)
A_416602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'A', False)
# Getting the type of 'b' (line 46)
b_416603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'b', False)
# Processing the call keyword arguments (line 46)
# Getting the type of 'M' (line 46)
M_416604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'M', False)
int_416605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'int')
int_416606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'int')
# Applying the binary operator '*' (line 46)
result_mul_416607 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 37), '*', int_416605, int_416606)

# Applying the binary operator '-' (line 46)
result_sub_416608 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 35), '-', M_416604, result_mul_416607)

keyword_416609 = result_sub_416608
int_416610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 50), 'int')
keyword_416611 = int_416610
float_416612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 57), 'float')
keyword_416613 = float_416612
kwargs_416614 = {'outer_k': keyword_416611, 'tol': keyword_416613, 'inner_m': keyword_416609}
# Getting the type of 'la' (line 46)
la_416600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'la', False)
# Obtaining the member 'lgmres' of a type (line 46)
lgmres_416601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), la_416600, 'lgmres')
# Calling lgmres(args, kwargs) (line 46)
lgmres_call_result_416615 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), lgmres_416601, *[A_416602, b_416603], **kwargs_416614)

# Obtaining the member '__getitem__' of a type (line 46)
getitem___416616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 0), lgmres_call_result_416615, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 46)
subscript_call_result_416617 = invoke(stypy.reporting.localization.Localization(__file__, 46, 0), getitem___416616, int_416599)

# Assigning a type to the variable 'tuple_var_assignment_416384' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'tuple_var_assignment_416384', subscript_call_result_416617)

# Assigning a Name to a Name (line 46):
# Getting the type of 'tuple_var_assignment_416383' (line 46)
tuple_var_assignment_416383_416618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'tuple_var_assignment_416383')
# Assigning a type to the variable 'x1' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'x1', tuple_var_assignment_416383_416618)

# Assigning a Name to a Name (line 46):
# Getting the type of 'tuple_var_assignment_416384' (line 46)
tuple_var_assignment_416384_416619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'tuple_var_assignment_416384')
# Assigning a type to the variable 'info' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'info', tuple_var_assignment_416384_416619)

# Assigning a Subscript to a Name (line 47):

# Assigning a Subscript to a Name (line 47):

# Obtaining the type of the subscript
int_416620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'int')
# Getting the type of 'count' (line 47)
count_416621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'count')
# Obtaining the member '__getitem__' of a type (line 47)
getitem___416622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 10), count_416621, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 47)
subscript_call_result_416623 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), getitem___416622, int_416620)

# Assigning a type to the variable 'count_1' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'count_1', subscript_call_result_416623)

# Assigning a BinOp to a Name (line 48):

# Assigning a BinOp to a Name (line 48):

# Call to norm(...): (line 48)
# Processing the call arguments (line 48)
# Getting the type of 'Am' (line 48)
Am_416627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'Am', False)
# Getting the type of 'x1' (line 48)
x1_416628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'x1', False)
# Applying the binary operator '*' (line 48)
result_mul_416629 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '*', Am_416627, x1_416628)

# Getting the type of 'b' (line 48)
b_416630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'b', False)
# Applying the binary operator '-' (line 48)
result_sub_416631 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 22), '-', result_mul_416629, b_416630)

# Processing the call keyword arguments (line 48)
kwargs_416632 = {}
# Getting the type of 'np' (line 48)
np_416624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'np', False)
# Obtaining the member 'linalg' of a type (line 48)
linalg_416625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), np_416624, 'linalg')
# Obtaining the member 'norm' of a type (line 48)
norm_416626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), linalg_416625, 'norm')
# Calling norm(args, kwargs) (line 48)
norm_call_result_416633 = invoke(stypy.reporting.localization.Localization(__file__, 48, 7), norm_416626, *[result_sub_416631], **kwargs_416632)


# Call to norm(...): (line 48)
# Processing the call arguments (line 48)
# Getting the type of 'b' (line 48)
b_416637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'b', False)
# Processing the call keyword arguments (line 48)
kwargs_416638 = {}
# Getting the type of 'np' (line 48)
np_416634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'np', False)
# Obtaining the member 'linalg' of a type (line 48)
linalg_416635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), np_416634, 'linalg')
# Obtaining the member 'norm' of a type (line 48)
norm_416636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), linalg_416635, 'norm')
# Calling norm(args, kwargs) (line 48)
norm_call_result_416639 = invoke(stypy.reporting.localization.Localization(__file__, 48, 35), norm_416636, *[b_416637], **kwargs_416638)

# Applying the binary operator 'div' (line 48)
result_div_416640 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), 'div', norm_call_result_416633, norm_call_result_416639)

# Assigning a type to the variable 'err1' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'err1', result_div_416640)

# Call to print(...): (line 49)
# Processing the call arguments (line 49)
str_416642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 6), 'str', 'LGMRES(%d,6) [same memory req.]:')
# Getting the type of 'M' (line 49)
M_416643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 44), 'M', False)
int_416644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 46), 'int')
int_416645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 48), 'int')
# Applying the binary operator '*' (line 49)
result_mul_416646 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 46), '*', int_416644, int_416645)

# Applying the binary operator '-' (line 49)
result_sub_416647 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 44), '-', M_416643, result_mul_416646)

# Applying the binary operator '%' (line 49)
result_mod_416648 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 6), '%', str_416642, result_sub_416647)

# Getting the type of 'count_1' (line 49)
count_1_416649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 52), 'count_1', False)
str_416650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 6), 'str', 'matvecs, residual:')
# Getting the type of 'err1' (line 50)
err1_416651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'err1', False)
# Processing the call keyword arguments (line 49)
kwargs_416652 = {}
# Getting the type of 'print' (line 49)
print_416641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'print', False)
# Calling print(args, kwargs) (line 49)
print_call_result_416653 = invoke(stypy.reporting.localization.Localization(__file__, 49, 0), print_416641, *[result_mod_416648, count_1_416649, str_416650, err1_416651], **kwargs_416652)



# Getting the type of 'info' (line 51)
info_416654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 3), 'info')
int_416655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'int')
# Applying the binary operator '!=' (line 51)
result_ne_416656 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 3), '!=', info_416654, int_416655)

# Testing the type of an if condition (line 51)
if_condition_416657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 0), result_ne_416656)
# Assigning a type to the variable 'if_condition_416657' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'if_condition_416657', if_condition_416657)
# SSA begins for if statement (line 51)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 52)
# Processing the call arguments (line 52)
str_416659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'str', "Didn't converge")
# Processing the call keyword arguments (line 52)
kwargs_416660 = {}
# Getting the type of 'print' (line 52)
print_416658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'print', False)
# Calling print(args, kwargs) (line 52)
print_call_result_416661 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), print_416658, *[str_416659], **kwargs_416660)

# SSA join for if statement (line 51)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Num to a Subscript (line 54):

# Assigning a Num to a Subscript (line 54):
int_416662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'int')
# Getting the type of 'count' (line 54)
count_416663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'count')
int_416664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 6), 'int')
# Storing an element on a container (line 54)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 0), count_416663, (int_416664, int_416662))

# Assigning a Call to a Tuple (line 55):

# Assigning a Subscript to a Name (line 55):

# Obtaining the type of the subscript
int_416665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 0), 'int')

# Call to lgmres(...): (line 55)
# Processing the call arguments (line 55)
# Getting the type of 'A' (line 55)
A_416668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'A', False)
# Getting the type of 'b' (line 55)
b_416669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'b', False)
# Processing the call keyword arguments (line 55)
# Getting the type of 'M' (line 55)
M_416670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'M', False)
int_416671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'int')
# Applying the binary operator '-' (line 55)
result_sub_416672 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 35), '-', M_416670, int_416671)

keyword_416673 = result_sub_416672
int_416674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
keyword_416675 = int_416674
float_416676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 55), 'float')
keyword_416677 = float_416676
kwargs_416678 = {'outer_k': keyword_416675, 'tol': keyword_416677, 'inner_m': keyword_416673}
# Getting the type of 'la' (line 55)
la_416666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'la', False)
# Obtaining the member 'lgmres' of a type (line 55)
lgmres_416667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), la_416666, 'lgmres')
# Calling lgmres(args, kwargs) (line 55)
lgmres_call_result_416679 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), lgmres_416667, *[A_416668, b_416669], **kwargs_416678)

# Obtaining the member '__getitem__' of a type (line 55)
getitem___416680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 0), lgmres_call_result_416679, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 55)
subscript_call_result_416681 = invoke(stypy.reporting.localization.Localization(__file__, 55, 0), getitem___416680, int_416665)

# Assigning a type to the variable 'tuple_var_assignment_416385' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'tuple_var_assignment_416385', subscript_call_result_416681)

# Assigning a Subscript to a Name (line 55):

# Obtaining the type of the subscript
int_416682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 0), 'int')

# Call to lgmres(...): (line 55)
# Processing the call arguments (line 55)
# Getting the type of 'A' (line 55)
A_416685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'A', False)
# Getting the type of 'b' (line 55)
b_416686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'b', False)
# Processing the call keyword arguments (line 55)
# Getting the type of 'M' (line 55)
M_416687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'M', False)
int_416688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'int')
# Applying the binary operator '-' (line 55)
result_sub_416689 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 35), '-', M_416687, int_416688)

keyword_416690 = result_sub_416689
int_416691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
keyword_416692 = int_416691
float_416693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 55), 'float')
keyword_416694 = float_416693
kwargs_416695 = {'outer_k': keyword_416692, 'tol': keyword_416694, 'inner_m': keyword_416690}
# Getting the type of 'la' (line 55)
la_416683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'la', False)
# Obtaining the member 'lgmres' of a type (line 55)
lgmres_416684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), la_416683, 'lgmres')
# Calling lgmres(args, kwargs) (line 55)
lgmres_call_result_416696 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), lgmres_416684, *[A_416685, b_416686], **kwargs_416695)

# Obtaining the member '__getitem__' of a type (line 55)
getitem___416697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 0), lgmres_call_result_416696, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 55)
subscript_call_result_416698 = invoke(stypy.reporting.localization.Localization(__file__, 55, 0), getitem___416697, int_416682)

# Assigning a type to the variable 'tuple_var_assignment_416386' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'tuple_var_assignment_416386', subscript_call_result_416698)

# Assigning a Name to a Name (line 55):
# Getting the type of 'tuple_var_assignment_416385' (line 55)
tuple_var_assignment_416385_416699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'tuple_var_assignment_416385')
# Assigning a type to the variable 'x2' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'x2', tuple_var_assignment_416385_416699)

# Assigning a Name to a Name (line 55):
# Getting the type of 'tuple_var_assignment_416386' (line 55)
tuple_var_assignment_416386_416700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'tuple_var_assignment_416386')
# Assigning a type to the variable 'info' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'info', tuple_var_assignment_416386_416700)

# Assigning a Subscript to a Name (line 56):

# Assigning a Subscript to a Name (line 56):

# Obtaining the type of the subscript
int_416701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 16), 'int')
# Getting the type of 'count' (line 56)
count_416702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'count')
# Obtaining the member '__getitem__' of a type (line 56)
getitem___416703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), count_416702, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 56)
subscript_call_result_416704 = invoke(stypy.reporting.localization.Localization(__file__, 56, 10), getitem___416703, int_416701)

# Assigning a type to the variable 'count_2' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'count_2', subscript_call_result_416704)

# Assigning a BinOp to a Name (line 57):

# Assigning a BinOp to a Name (line 57):

# Call to norm(...): (line 57)
# Processing the call arguments (line 57)
# Getting the type of 'Am' (line 57)
Am_416708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 22), 'Am', False)
# Getting the type of 'x2' (line 57)
x2_416709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'x2', False)
# Applying the binary operator '*' (line 57)
result_mul_416710 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 22), '*', Am_416708, x2_416709)

# Getting the type of 'b' (line 57)
b_416711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'b', False)
# Applying the binary operator '-' (line 57)
result_sub_416712 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 22), '-', result_mul_416710, b_416711)

# Processing the call keyword arguments (line 57)
kwargs_416713 = {}
# Getting the type of 'np' (line 57)
np_416705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'np', False)
# Obtaining the member 'linalg' of a type (line 57)
linalg_416706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 7), np_416705, 'linalg')
# Obtaining the member 'norm' of a type (line 57)
norm_416707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 7), linalg_416706, 'norm')
# Calling norm(args, kwargs) (line 57)
norm_call_result_416714 = invoke(stypy.reporting.localization.Localization(__file__, 57, 7), norm_416707, *[result_sub_416712], **kwargs_416713)


# Call to norm(...): (line 57)
# Processing the call arguments (line 57)
# Getting the type of 'b' (line 57)
b_416718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 50), 'b', False)
# Processing the call keyword arguments (line 57)
kwargs_416719 = {}
# Getting the type of 'np' (line 57)
np_416715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 35), 'np', False)
# Obtaining the member 'linalg' of a type (line 57)
linalg_416716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 35), np_416715, 'linalg')
# Obtaining the member 'norm' of a type (line 57)
norm_416717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 35), linalg_416716, 'norm')
# Calling norm(args, kwargs) (line 57)
norm_call_result_416720 = invoke(stypy.reporting.localization.Localization(__file__, 57, 35), norm_416717, *[b_416718], **kwargs_416719)

# Applying the binary operator 'div' (line 57)
result_div_416721 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), 'div', norm_call_result_416714, norm_call_result_416720)

# Assigning a type to the variable 'err2' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'err2', result_div_416721)

# Call to print(...): (line 58)
# Processing the call arguments (line 58)
str_416723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 6), 'str', 'LGMRES(%d,6) [same subspace size]:')
# Getting the type of 'M' (line 58)
M_416724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 46), 'M', False)
int_416725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 48), 'int')
# Applying the binary operator '-' (line 58)
result_sub_416726 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 46), '-', M_416724, int_416725)

# Applying the binary operator '%' (line 58)
result_mod_416727 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 6), '%', str_416723, result_sub_416726)

# Getting the type of 'count_2' (line 58)
count_2_416728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'count_2', False)
str_416729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 6), 'str', 'matvecs, residual:')
# Getting the type of 'err2' (line 59)
err2_416730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'err2', False)
# Processing the call keyword arguments (line 58)
kwargs_416731 = {}
# Getting the type of 'print' (line 58)
print_416722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'print', False)
# Calling print(args, kwargs) (line 58)
print_call_result_416732 = invoke(stypy.reporting.localization.Localization(__file__, 58, 0), print_416722, *[result_mod_416727, count_2_416728, str_416729, err2_416730], **kwargs_416731)



# Getting the type of 'info' (line 60)
info_416733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 3), 'info')
int_416734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 11), 'int')
# Applying the binary operator '!=' (line 60)
result_ne_416735 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 3), '!=', info_416733, int_416734)

# Testing the type of an if condition (line 60)
if_condition_416736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 0), result_ne_416735)
# Assigning a type to the variable 'if_condition_416736' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'if_condition_416736', if_condition_416736)
# SSA begins for if statement (line 60)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to print(...): (line 61)
# Processing the call arguments (line 61)
str_416738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 10), 'str', "Didn't converge")
# Processing the call keyword arguments (line 61)
kwargs_416739 = {}
# Getting the type of 'print' (line 61)
print_416737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'print', False)
# Calling print(args, kwargs) (line 61)
print_call_result_416740 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), print_416737, *[str_416738], **kwargs_416739)

# SSA join for if statement (line 60)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
