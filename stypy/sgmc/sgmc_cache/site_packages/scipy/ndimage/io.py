
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: 
5: 
6: _have_pil = True
7: try:
8:     from scipy.misc.pilutil import imread as _imread
9: except ImportError:
10:     _have_pil = False
11: 
12: 
13: __all__ = ['imread']
14: 
15: 
16: # Use the implementation of `imread` in `scipy.misc.pilutil.imread`.
17: # If it weren't for the different names of the first arguments of
18: # ndimage.io.imread and misc.pilutil.imread, we could simplify this file
19: # by writing
20: #     from scipy.misc.pilutil import imread
21: # Unfortunately, because the argument names are different, that
22: # introduces a backwards incompatibility.
23: 
24: @np.deprecate(message="`imread` is deprecated in SciPy 1.0.0.\n"
25:                       "Use ``matplotlib.pyplot.imread`` instead.")
26: def imread(fname, flatten=False, mode=None):
27:     if _have_pil:
28:         return _imread(fname, flatten, mode)
29:     raise ImportError("Could not import the Python Imaging Library (PIL)"
30:                       " required to load image files.  Please refer to"
31:                       " http://pillow.readthedocs.org/en/latest/installation.html"
32:                       " for installation instructions.")
33: 
34: if _have_pil and _imread.__doc__ is not None:
35:     imread.__doc__ = _imread.__doc__.replace('name : str', 'fname : str')
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121864 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_121864) is not StypyTypeError):

    if (import_121864 != 'pyd_module'):
        __import__(import_121864)
        sys_modules_121865 = sys.modules[import_121864]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_121865.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_121864)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_121866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'True')
# Assigning a type to the variable '_have_pil' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '_have_pil', True_121866)


# SSA begins for try-except statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from scipy.misc.pilutil import _imread' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121867 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'scipy.misc.pilutil')

if (type(import_121867) is not StypyTypeError):

    if (import_121867 != 'pyd_module'):
        __import__(import_121867)
        sys_modules_121868 = sys.modules[import_121867]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'scipy.misc.pilutil', sys_modules_121868.module_type_store, module_type_store, ['imread'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_121868, sys_modules_121868.module_type_store, module_type_store)
    else:
        from scipy.misc.pilutil import imread as _imread

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'scipy.misc.pilutil', None, module_type_store, ['imread'], [_imread])

else:
    # Assigning a type to the variable 'scipy.misc.pilutil' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'scipy.misc.pilutil', import_121867)

# Adding an alias
module_type_store.add_alias('_imread', 'imread')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

# SSA branch for the except part of a try statement (line 7)
# SSA branch for the except 'ImportError' branch of a try statement (line 7)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 10):
# Getting the type of 'False' (line 10)
False_121869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'False')
# Assigning a type to the variable '_have_pil' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), '_have_pil', False_121869)
# SSA join for try-except statement (line 7)
module_type_store = module_type_store.join_ssa_context()


# Assigning a List to a Name (line 13):
__all__ = ['imread']
module_type_store.set_exportable_members(['imread'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_121870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_121871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'imread')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_121870, str_121871)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_121870)

@norecursion
def imread(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 26)
    False_121872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'False')
    # Getting the type of 'None' (line 26)
    None_121873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'None')
    defaults = [False_121872, None_121873]
    # Create a new context for function 'imread'
    module_type_store = module_type_store.open_function_context('imread', 24, 0, False)
    
    # Passed parameters checking function
    imread.stypy_localization = localization
    imread.stypy_type_of_self = None
    imread.stypy_type_store = module_type_store
    imread.stypy_function_name = 'imread'
    imread.stypy_param_names_list = ['fname', 'flatten', 'mode']
    imread.stypy_varargs_param_name = None
    imread.stypy_kwargs_param_name = None
    imread.stypy_call_defaults = defaults
    imread.stypy_call_varargs = varargs
    imread.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'imread', ['fname', 'flatten', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'imread', localization, ['fname', 'flatten', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'imread(...)' code ##################

    
    # Getting the type of '_have_pil' (line 27)
    _have_pil_121874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), '_have_pil')
    # Testing the type of an if condition (line 27)
    if_condition_121875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), _have_pil_121874)
    # Assigning a type to the variable 'if_condition_121875' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_121875', if_condition_121875)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _imread(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'fname' (line 28)
    fname_121877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'fname', False)
    # Getting the type of 'flatten' (line 28)
    flatten_121878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'flatten', False)
    # Getting the type of 'mode' (line 28)
    mode_121879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'mode', False)
    # Processing the call keyword arguments (line 28)
    kwargs_121880 = {}
    # Getting the type of '_imread' (line 28)
    _imread_121876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), '_imread', False)
    # Calling _imread(args, kwargs) (line 28)
    _imread_call_result_121881 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), _imread_121876, *[fname_121877, flatten_121878, mode_121879], **kwargs_121880)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', _imread_call_result_121881)
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ImportError(...): (line 29)
    # Processing the call arguments (line 29)
    str_121883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'str', 'Could not import the Python Imaging Library (PIL) required to load image files.  Please refer to http://pillow.readthedocs.org/en/latest/installation.html for installation instructions.')
    # Processing the call keyword arguments (line 29)
    kwargs_121884 = {}
    # Getting the type of 'ImportError' (line 29)
    ImportError_121882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'ImportError', False)
    # Calling ImportError(args, kwargs) (line 29)
    ImportError_call_result_121885 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), ImportError_121882, *[str_121883], **kwargs_121884)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 4), ImportError_call_result_121885, 'raise parameter', BaseException)
    
    # ################# End of 'imread(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'imread' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_121886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'imread'
    return stypy_return_type_121886

# Assigning a type to the variable 'imread' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'imread', imread)


# Evaluating a boolean operation
# Getting the type of '_have_pil' (line 34)
_have_pil_121887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 3), '_have_pil')

# Getting the type of '_imread' (line 34)
_imread_121888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), '_imread')
# Obtaining the member '__doc__' of a type (line 34)
doc___121889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 17), _imread_121888, '__doc__')
# Getting the type of 'None' (line 34)
None_121890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'None')
# Applying the binary operator 'isnot' (line 34)
result_is_not_121891 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 17), 'isnot', doc___121889, None_121890)

# Applying the binary operator 'and' (line 34)
result_and_keyword_121892 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 3), 'and', _have_pil_121887, result_is_not_121891)

# Testing the type of an if condition (line 34)
if_condition_121893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 0), result_and_keyword_121892)
# Assigning a type to the variable 'if_condition_121893' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'if_condition_121893', if_condition_121893)
# SSA begins for if statement (line 34)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Attribute (line 35):

# Call to replace(...): (line 35)
# Processing the call arguments (line 35)
str_121897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 45), 'str', 'name : str')
str_121898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 59), 'str', 'fname : str')
# Processing the call keyword arguments (line 35)
kwargs_121899 = {}
# Getting the type of '_imread' (line 35)
_imread_121894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), '_imread', False)
# Obtaining the member '__doc__' of a type (line 35)
doc___121895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), _imread_121894, '__doc__')
# Obtaining the member 'replace' of a type (line 35)
replace_121896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), doc___121895, 'replace')
# Calling replace(args, kwargs) (line 35)
replace_call_result_121900 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), replace_121896, *[str_121897, str_121898], **kwargs_121899)

# Getting the type of 'imread' (line 35)
imread_121901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'imread')
# Setting the type of the member '__doc__' of a type (line 35)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), imread_121901, '__doc__', replace_call_result_121900)
# SSA join for if statement (line 34)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
