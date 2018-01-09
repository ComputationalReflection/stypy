
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import six
2: 
3: defaultParams = {0: ('zero', '0'), 1: ('one', '1')}
4: 
5: param1 = six.iteritems(defaultParams)
6: 
7: validate = dict((key, converter) for key, (default, converter) in six.iteritems(defaultParams))
8: 
9: 
10: params2 = {0: '0', 1: '1'}
11: 
12: it = six.iteritems(params2)
13: 
14: 
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import six' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/without_classes/generator_expression/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', sys_modules_2.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'six', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/without_classes/generator_expression/')


# Assigning a Dict to a Name (line 3):

# Obtaining an instance of the builtin type 'dict' (line 3)
dict_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 3)
# Adding element type (key, value) (line 3)
int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 17), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 21), 'str', 'zero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 21), tuple_5, str_6)
# Adding element type (line 3)
str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 29), 'str', '0')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 21), tuple_5, str_7)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 16), dict_3, (int_4, tuple_5))
# Adding element type (key, value) (line 3)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 35), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'str', 'one')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 39), tuple_9, str_10)
# Adding element type (line 3)
str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 46), 'str', '1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 39), tuple_9, str_11)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 16), dict_3, (int_8, tuple_9))

# Assigning a type to the variable 'defaultParams' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'defaultParams', dict_3)

# Assigning a Call to a Name (line 5):

# Call to iteritems(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of 'defaultParams' (line 5)
defaultParams_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 23), 'defaultParams', False)
# Processing the call keyword arguments (line 5)
kwargs_15 = {}
# Getting the type of 'six' (line 5)
six_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 9), 'six', False)
# Obtaining the member 'iteritems' of a type (line 5)
iteritems_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 9), six_12, 'iteritems')
# Calling iteritems(args, kwargs) (line 5)
iteritems_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 5, 9), iteritems_13, *[defaultParams_14], **kwargs_15)

# Assigning a type to the variable 'param1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'param1', iteritems_call_result_16)

# Assigning a Call to a Name (line 7):

# Call to dict(...): (line 7)
# Processing the call arguments (line 7)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 7, 16, True)
# Calculating comprehension expression

# Call to iteritems(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'defaultParams' (line 7)
defaultParams_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 80), 'defaultParams', False)
# Processing the call keyword arguments (line 7)
kwargs_24 = {}
# Getting the type of 'six' (line 7)
six_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 66), 'six', False)
# Obtaining the member 'iteritems' of a type (line 7)
iteritems_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 66), six_21, 'iteritems')
# Calling iteritems(args, kwargs) (line 7)
iteritems_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 7, 66), iteritems_22, *[defaultParams_23], **kwargs_24)

comprehension_26 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), iteritems_call_result_25)
# Assigning a type to the variable 'key' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), comprehension_26))
# Assigning a type to the variable 'default' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'default', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), comprehension_26))
# Assigning a type to the variable 'converter' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'converter', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), comprehension_26))

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'key' (line 7)
key_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 17), 'key', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 17), tuple_18, key_19)
# Adding element type (line 7)
# Getting the type of 'converter' (line 7)
converter_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 22), 'converter', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 17), tuple_18, converter_20)

list_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), list_28, tuple_18)
# Processing the call keyword arguments (line 7)
kwargs_29 = {}
# Getting the type of 'dict' (line 7)
dict_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'dict', False)
# Calling dict(args, kwargs) (line 7)
dict_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), dict_17, *[list_28], **kwargs_29)

# Assigning a type to the variable 'validate' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'validate', dict_call_result_30)

# Assigning a Dict to a Name (line 10):

# Obtaining an instance of the builtin type 'dict' (line 10)
dict_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 10)
# Adding element type (key, value) (line 10)
int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'int')
str_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', '0')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), dict_31, (int_32, str_33))
# Adding element type (key, value) (line 10)
int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
str_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'str', '1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), dict_31, (int_34, str_35))

# Assigning a type to the variable 'params2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'params2', dict_31)

# Assigning a Call to a Name (line 12):

# Call to iteritems(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'params2' (line 12)
params2_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'params2', False)
# Processing the call keyword arguments (line 12)
kwargs_39 = {}
# Getting the type of 'six' (line 12)
six_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'six', False)
# Obtaining the member 'iteritems' of a type (line 12)
iteritems_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), six_36, 'iteritems')
# Calling iteritems(args, kwargs) (line 12)
iteritems_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), iteritems_37, *[params2_38], **kwargs_39)

# Assigning a type to the variable 'it' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'it', iteritems_call_result_40)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
