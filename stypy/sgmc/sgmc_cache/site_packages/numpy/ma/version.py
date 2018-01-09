
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Version number
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: version = '1.00'
7: release = False
8: 
9: if not release:
10:     from . import core
11:     from . import extras
12:     revision = [core.__revision__.split(':')[-1][:-1].strip(),
13:                 extras.__revision__.split(':')[-1][:-1].strip(),]
14:     version += '.dev%04i' % max([int(rev) for rev in revision])
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_160406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Version number\n\n')

# Assigning a Str to a Name (line 6):
str_160407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', '1.00')
# Assigning a type to the variable 'version' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'version', str_160407)

# Assigning a Name to a Name (line 7):
# Getting the type of 'False' (line 7)
False_160408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'False')
# Assigning a type to the variable 'release' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'release', False_160408)


# Getting the type of 'release' (line 9)
release_160409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 7), 'release')
# Applying the 'not' unary operator (line 9)
result_not__160410 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 3), 'not', release_160409)

# Testing the type of an if condition (line 9)
if_condition_160411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 0), result_not__160410)
# Assigning a type to the variable 'if_condition_160411' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'if_condition_160411', if_condition_160411)
# SSA begins for if statement (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from numpy.ma import core' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.ma')

if (type(import_160412) is not StypyTypeError):

    if (import_160412 != 'pyd_module'):
        __import__(import_160412)
        sys_modules_160413 = sys.modules[import_160412]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.ma', sys_modules_160413.module_type_store, module_type_store, ['core'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_160413, sys_modules_160413.module_type_store, module_type_store)
    else:
        from numpy.ma import core

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.ma', None, module_type_store, ['core'], [core])

else:
    # Assigning a type to the variable 'numpy.ma' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.ma', import_160412)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))

# 'from numpy.ma import extras' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_160414 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.ma')

if (type(import_160414) is not StypyTypeError):

    if (import_160414 != 'pyd_module'):
        __import__(import_160414)
        sys_modules_160415 = sys.modules[import_160414]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.ma', sys_modules_160415.module_type_store, module_type_store, ['extras'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_160415, sys_modules_160415.module_type_store, module_type_store)
    else:
        from numpy.ma import extras

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.ma', None, module_type_store, ['extras'], [extras])

else:
    # Assigning a type to the variable 'numpy.ma' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.ma', import_160414)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_160416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

# Call to strip(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_160431 = {}

# Obtaining the type of the subscript
int_160417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 50), 'int')
slice_160418 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 12, 16), None, int_160417, None)

# Obtaining the type of the subscript
int_160419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 45), 'int')

# Call to split(...): (line 12)
# Processing the call arguments (line 12)
str_160423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 40), 'str', ':')
# Processing the call keyword arguments (line 12)
kwargs_160424 = {}
# Getting the type of 'core' (line 12)
core_160420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'core', False)
# Obtaining the member '__revision__' of a type (line 12)
revision___160421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), core_160420, '__revision__')
# Obtaining the member 'split' of a type (line 12)
split_160422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), revision___160421, 'split')
# Calling split(args, kwargs) (line 12)
split_call_result_160425 = invoke(stypy.reporting.localization.Localization(__file__, 12, 16), split_160422, *[str_160423], **kwargs_160424)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___160426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), split_call_result_160425, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_160427 = invoke(stypy.reporting.localization.Localization(__file__, 12, 16), getitem___160426, int_160419)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___160428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), subscript_call_result_160427, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_160429 = invoke(stypy.reporting.localization.Localization(__file__, 12, 16), getitem___160428, slice_160418)

# Obtaining the member 'strip' of a type (line 12)
strip_160430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 16), subscript_call_result_160429, 'strip')
# Calling strip(args, kwargs) (line 12)
strip_call_result_160432 = invoke(stypy.reporting.localization.Localization(__file__, 12, 16), strip_160430, *[], **kwargs_160431)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 15), list_160416, strip_call_result_160432)
# Adding element type (line 12)

# Call to strip(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_160447 = {}

# Obtaining the type of the subscript
int_160433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 52), 'int')
slice_160434 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 16), None, int_160433, None)

# Obtaining the type of the subscript
int_160435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 47), 'int')

# Call to split(...): (line 13)
# Processing the call arguments (line 13)
str_160439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 42), 'str', ':')
# Processing the call keyword arguments (line 13)
kwargs_160440 = {}
# Getting the type of 'extras' (line 13)
extras_160436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'extras', False)
# Obtaining the member '__revision__' of a type (line 13)
revision___160437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), extras_160436, '__revision__')
# Obtaining the member 'split' of a type (line 13)
split_160438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), revision___160437, 'split')
# Calling split(args, kwargs) (line 13)
split_call_result_160441 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), split_160438, *[str_160439], **kwargs_160440)

# Obtaining the member '__getitem__' of a type (line 13)
getitem___160442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), split_call_result_160441, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_160443 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), getitem___160442, int_160435)

# Obtaining the member '__getitem__' of a type (line 13)
getitem___160444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), subscript_call_result_160443, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_160445 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), getitem___160444, slice_160434)

# Obtaining the member 'strip' of a type (line 13)
strip_160446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), subscript_call_result_160445, 'strip')
# Calling strip(args, kwargs) (line 13)
strip_call_result_160448 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), strip_160446, *[], **kwargs_160447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 15), list_160416, strip_call_result_160448)

# Assigning a type to the variable 'revision' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'revision', list_160416)

# Getting the type of 'version' (line 14)
version_160449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'version')
str_160450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', '.dev%04i')

# Call to max(...): (line 14)
# Processing the call arguments (line 14)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'revision' (line 14)
revision_160456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 53), 'revision', False)
comprehension_160457 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), revision_160456)
# Assigning a type to the variable 'rev' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 33), 'rev', comprehension_160457)

# Call to int(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'rev' (line 14)
rev_160453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'rev', False)
# Processing the call keyword arguments (line 14)
kwargs_160454 = {}
# Getting the type of 'int' (line 14)
int_160452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 33), 'int', False)
# Calling int(args, kwargs) (line 14)
int_call_result_160455 = invoke(stypy.reporting.localization.Localization(__file__, 14, 33), int_160452, *[rev_160453], **kwargs_160454)

list_160458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_160458, int_call_result_160455)
# Processing the call keyword arguments (line 14)
kwargs_160459 = {}
# Getting the type of 'max' (line 14)
max_160451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'max', False)
# Calling max(args, kwargs) (line 14)
max_call_result_160460 = invoke(stypy.reporting.localization.Localization(__file__, 14, 28), max_160451, *[list_160458], **kwargs_160459)

# Applying the binary operator '%' (line 14)
result_mod_160461 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '%', str_160450, max_call_result_160460)

# Applying the binary operator '+=' (line 14)
result_iadd_160462 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 4), '+=', version_160449, result_mod_160461)
# Assigning a type to the variable 'version' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'version', result_iadd_160462)

# SSA join for if statement (line 9)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
