
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy import stypy_parameters_copy
2: 
3: r1 = stypy_parameters_copy.type_inference_file_directory_name
4: r2 = stypy_parameters_copy.type_inference_file_postfix
5: r3 = stypy_parameters_copy.type_modifier_file_postfix
6: r4 = stypy_parameters_copy.type_data_autogenerator_file_postfix
7: r5 = stypy_parameters_copy.type_data_file_postfix
8: r6 = stypy_parameters_copy.type_annotation_file_postfix
9: r7 = stypy_parameters_copy.type_rule_file_postfix
10: 
11: source_file_path = stypy_parameters_copy.ROOT_PATH + "/stypy.py"
12: r8 = stypy_parameters_copy. go_to_parent_folder(0, source_file_path)
13: r9 = stypy_parameters_copy. get_original_program_from_type_inference_file(source_file_path)
14: r10 = stypy_parameters_copy. get_stypy_type_inference_program_file_path(source_file_path)
15: r11 = stypy_parameters_copy. get_stypy_type_data_autogenerator_program_file_path(source_file_path)
16: r12 = stypy_parameters_copy. get_stypy_type_data_file_path(source_file_path)
17: r13 = stypy_parameters_copy. get_stypy_type_annotation_file_path(source_file_path)
18: 
19: 
20: r14 = stypy_parameters_copy.PYTHON_EXE_PATH
21: r15 = stypy_parameters_copy.PYTHON_EXE
22: r16 = stypy_parameters_copy.ROOT_PATH
23: r17 = stypy_parameters_copy.TYPE_INFERENCE_PATH
24: r18 = stypy_parameters_copy.RULE_FILE_PATH
25: r19 = stypy_parameters_copy.LOG_PATH
26: r20 = stypy_parameters_copy.ERROR_LOG_FILE
27: r21 = stypy_parameters_copy.WARNING_LOG_FILE
28: r22 = stypy_parameters_copy.INFO_LOG_FILE
29: r23 = stypy_parameters_copy.ENABLE_CODING_ADVICES

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/individual_file_tests/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', sys_modules_2.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/individual_file_tests/')


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'stypy_parameters_copy' (line 3)
stypy_parameters_copy_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_inference_file_directory_name' of a type (line 3)
type_inference_file_directory_name_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 5), stypy_parameters_copy_3, 'type_inference_file_directory_name')
# Assigning a type to the variable 'r1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'r1', type_inference_file_directory_name_4)

# Assigning a Attribute to a Name (line 4):
# Getting the type of 'stypy_parameters_copy' (line 4)
stypy_parameters_copy_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_inference_file_postfix' of a type (line 4)
type_inference_file_postfix_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), stypy_parameters_copy_5, 'type_inference_file_postfix')
# Assigning a type to the variable 'r2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'r2', type_inference_file_postfix_6)

# Assigning a Attribute to a Name (line 5):
# Getting the type of 'stypy_parameters_copy' (line 5)
stypy_parameters_copy_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_modifier_file_postfix' of a type (line 5)
type_modifier_file_postfix_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), stypy_parameters_copy_7, 'type_modifier_file_postfix')
# Assigning a type to the variable 'r3' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r3', type_modifier_file_postfix_8)

# Assigning a Attribute to a Name (line 6):
# Getting the type of 'stypy_parameters_copy' (line 6)
stypy_parameters_copy_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_data_autogenerator_file_postfix' of a type (line 6)
type_data_autogenerator_file_postfix_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), stypy_parameters_copy_9, 'type_data_autogenerator_file_postfix')
# Assigning a type to the variable 'r4' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r4', type_data_autogenerator_file_postfix_10)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'stypy_parameters_copy' (line 7)
stypy_parameters_copy_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_data_file_postfix' of a type (line 7)
type_data_file_postfix_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), stypy_parameters_copy_11, 'type_data_file_postfix')
# Assigning a type to the variable 'r5' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r5', type_data_file_postfix_12)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'stypy_parameters_copy' (line 8)
stypy_parameters_copy_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_annotation_file_postfix' of a type (line 8)
type_annotation_file_postfix_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), stypy_parameters_copy_13, 'type_annotation_file_postfix')
# Assigning a type to the variable 'r6' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r6', type_annotation_file_postfix_14)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'stypy_parameters_copy' (line 9)
stypy_parameters_copy_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'stypy_parameters_copy')
# Obtaining the member 'type_rule_file_postfix' of a type (line 9)
type_rule_file_postfix_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), stypy_parameters_copy_15, 'type_rule_file_postfix')
# Assigning a type to the variable 'r7' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r7', type_rule_file_postfix_16)

# Assigning a BinOp to a Name (line 11):
# Getting the type of 'stypy_parameters_copy' (line 11)
stypy_parameters_copy_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 19), 'stypy_parameters_copy')
# Obtaining the member 'ROOT_PATH' of a type (line 11)
ROOT_PATH_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 19), stypy_parameters_copy_17, 'ROOT_PATH')
str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 53), 'str', '/stypy.py')
# Applying the binary operator '+' (line 11)
result_add_20 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 19), '+', ROOT_PATH_18, str_19)

# Assigning a type to the variable 'source_file_path' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'source_file_path', result_add_20)

# Assigning a Call to a Name (line 12):

# Call to go_to_parent_folder(...): (line 12)
# Processing the call arguments (line 12)
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 48), 'int')
# Getting the type of 'source_file_path' (line 12)
source_file_path_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 51), 'source_file_path', False)
# Processing the call keyword arguments (line 12)
kwargs_25 = {}
# Getting the type of 'stypy_parameters_copy' (line 12)
stypy_parameters_copy_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'stypy_parameters_copy', False)
# Obtaining the member 'go_to_parent_folder' of a type (line 12)
go_to_parent_folder_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), stypy_parameters_copy_21, 'go_to_parent_folder')
# Calling go_to_parent_folder(args, kwargs) (line 12)
go_to_parent_folder_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), go_to_parent_folder_22, *[int_23, source_file_path_24], **kwargs_25)

# Assigning a type to the variable 'r8' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r8', go_to_parent_folder_call_result_26)

# Assigning a Call to a Name (line 13):

# Call to get_original_program_from_type_inference_file(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'source_file_path' (line 13)
source_file_path_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 74), 'source_file_path', False)
# Processing the call keyword arguments (line 13)
kwargs_30 = {}
# Getting the type of 'stypy_parameters_copy' (line 13)
stypy_parameters_copy_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'stypy_parameters_copy', False)
# Obtaining the member 'get_original_program_from_type_inference_file' of a type (line 13)
get_original_program_from_type_inference_file_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), stypy_parameters_copy_27, 'get_original_program_from_type_inference_file')
# Calling get_original_program_from_type_inference_file(args, kwargs) (line 13)
get_original_program_from_type_inference_file_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), get_original_program_from_type_inference_file_28, *[source_file_path_29], **kwargs_30)

# Assigning a type to the variable 'r9' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r9', get_original_program_from_type_inference_file_call_result_31)

# Assigning a Call to a Name (line 14):

# Call to get_stypy_type_inference_program_file_path(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'source_file_path' (line 14)
source_file_path_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 72), 'source_file_path', False)
# Processing the call keyword arguments (line 14)
kwargs_35 = {}
# Getting the type of 'stypy_parameters_copy' (line 14)
stypy_parameters_copy_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 6), 'stypy_parameters_copy', False)
# Obtaining the member 'get_stypy_type_inference_program_file_path' of a type (line 14)
get_stypy_type_inference_program_file_path_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 6), stypy_parameters_copy_32, 'get_stypy_type_inference_program_file_path')
# Calling get_stypy_type_inference_program_file_path(args, kwargs) (line 14)
get_stypy_type_inference_program_file_path_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 14, 6), get_stypy_type_inference_program_file_path_33, *[source_file_path_34], **kwargs_35)

# Assigning a type to the variable 'r10' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r10', get_stypy_type_inference_program_file_path_call_result_36)

# Assigning a Call to a Name (line 15):

# Call to get_stypy_type_data_autogenerator_program_file_path(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'source_file_path' (line 15)
source_file_path_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 81), 'source_file_path', False)
# Processing the call keyword arguments (line 15)
kwargs_40 = {}
# Getting the type of 'stypy_parameters_copy' (line 15)
stypy_parameters_copy_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 6), 'stypy_parameters_copy', False)
# Obtaining the member 'get_stypy_type_data_autogenerator_program_file_path' of a type (line 15)
get_stypy_type_data_autogenerator_program_file_path_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 6), stypy_parameters_copy_37, 'get_stypy_type_data_autogenerator_program_file_path')
# Calling get_stypy_type_data_autogenerator_program_file_path(args, kwargs) (line 15)
get_stypy_type_data_autogenerator_program_file_path_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 15, 6), get_stypy_type_data_autogenerator_program_file_path_38, *[source_file_path_39], **kwargs_40)

# Assigning a type to the variable 'r11' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r11', get_stypy_type_data_autogenerator_program_file_path_call_result_41)

# Assigning a Call to a Name (line 16):

# Call to get_stypy_type_data_file_path(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'source_file_path' (line 16)
source_file_path_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 59), 'source_file_path', False)
# Processing the call keyword arguments (line 16)
kwargs_45 = {}
# Getting the type of 'stypy_parameters_copy' (line 16)
stypy_parameters_copy_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'stypy_parameters_copy', False)
# Obtaining the member 'get_stypy_type_data_file_path' of a type (line 16)
get_stypy_type_data_file_path_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 6), stypy_parameters_copy_42, 'get_stypy_type_data_file_path')
# Calling get_stypy_type_data_file_path(args, kwargs) (line 16)
get_stypy_type_data_file_path_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 16, 6), get_stypy_type_data_file_path_43, *[source_file_path_44], **kwargs_45)

# Assigning a type to the variable 'r12' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r12', get_stypy_type_data_file_path_call_result_46)

# Assigning a Call to a Name (line 17):

# Call to get_stypy_type_annotation_file_path(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'source_file_path' (line 17)
source_file_path_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 65), 'source_file_path', False)
# Processing the call keyword arguments (line 17)
kwargs_50 = {}
# Getting the type of 'stypy_parameters_copy' (line 17)
stypy_parameters_copy_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 6), 'stypy_parameters_copy', False)
# Obtaining the member 'get_stypy_type_annotation_file_path' of a type (line 17)
get_stypy_type_annotation_file_path_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 6), stypy_parameters_copy_47, 'get_stypy_type_annotation_file_path')
# Calling get_stypy_type_annotation_file_path(args, kwargs) (line 17)
get_stypy_type_annotation_file_path_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 17, 6), get_stypy_type_annotation_file_path_48, *[source_file_path_49], **kwargs_50)

# Assigning a type to the variable 'r13' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r13', get_stypy_type_annotation_file_path_call_result_51)

# Assigning a Attribute to a Name (line 20):
# Getting the type of 'stypy_parameters_copy' (line 20)
stypy_parameters_copy_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'stypy_parameters_copy')
# Obtaining the member 'PYTHON_EXE_PATH' of a type (line 20)
PYTHON_EXE_PATH_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 6), stypy_parameters_copy_52, 'PYTHON_EXE_PATH')
# Assigning a type to the variable 'r14' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r14', PYTHON_EXE_PATH_53)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'stypy_parameters_copy' (line 21)
stypy_parameters_copy_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'stypy_parameters_copy')
# Obtaining the member 'PYTHON_EXE' of a type (line 21)
PYTHON_EXE_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), stypy_parameters_copy_54, 'PYTHON_EXE')
# Assigning a type to the variable 'r15' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r15', PYTHON_EXE_55)

# Assigning a Attribute to a Name (line 22):
# Getting the type of 'stypy_parameters_copy' (line 22)
stypy_parameters_copy_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'stypy_parameters_copy')
# Obtaining the member 'ROOT_PATH' of a type (line 22)
ROOT_PATH_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), stypy_parameters_copy_56, 'ROOT_PATH')
# Assigning a type to the variable 'r16' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r16', ROOT_PATH_57)

# Assigning a Attribute to a Name (line 23):
# Getting the type of 'stypy_parameters_copy' (line 23)
stypy_parameters_copy_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'stypy_parameters_copy')
# Obtaining the member 'TYPE_INFERENCE_PATH' of a type (line 23)
TYPE_INFERENCE_PATH_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 6), stypy_parameters_copy_58, 'TYPE_INFERENCE_PATH')
# Assigning a type to the variable 'r17' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r17', TYPE_INFERENCE_PATH_59)

# Assigning a Attribute to a Name (line 24):
# Getting the type of 'stypy_parameters_copy' (line 24)
stypy_parameters_copy_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 6), 'stypy_parameters_copy')
# Obtaining the member 'RULE_FILE_PATH' of a type (line 24)
RULE_FILE_PATH_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), stypy_parameters_copy_60, 'RULE_FILE_PATH')
# Assigning a type to the variable 'r18' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r18', RULE_FILE_PATH_61)

# Assigning a Attribute to a Name (line 25):
# Getting the type of 'stypy_parameters_copy' (line 25)
stypy_parameters_copy_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'stypy_parameters_copy')
# Obtaining the member 'LOG_PATH' of a type (line 25)
LOG_PATH_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 6), stypy_parameters_copy_62, 'LOG_PATH')
# Assigning a type to the variable 'r19' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r19', LOG_PATH_63)

# Assigning a Attribute to a Name (line 26):
# Getting the type of 'stypy_parameters_copy' (line 26)
stypy_parameters_copy_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'stypy_parameters_copy')
# Obtaining the member 'ERROR_LOG_FILE' of a type (line 26)
ERROR_LOG_FILE_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), stypy_parameters_copy_64, 'ERROR_LOG_FILE')
# Assigning a type to the variable 'r20' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r20', ERROR_LOG_FILE_65)

# Assigning a Attribute to a Name (line 27):
# Getting the type of 'stypy_parameters_copy' (line 27)
stypy_parameters_copy_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'stypy_parameters_copy')
# Obtaining the member 'WARNING_LOG_FILE' of a type (line 27)
WARNING_LOG_FILE_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), stypy_parameters_copy_66, 'WARNING_LOG_FILE')
# Assigning a type to the variable 'r21' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r21', WARNING_LOG_FILE_67)

# Assigning a Attribute to a Name (line 28):
# Getting the type of 'stypy_parameters_copy' (line 28)
stypy_parameters_copy_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'stypy_parameters_copy')
# Obtaining the member 'INFO_LOG_FILE' of a type (line 28)
INFO_LOG_FILE_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), stypy_parameters_copy_68, 'INFO_LOG_FILE')
# Assigning a type to the variable 'r22' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'r22', INFO_LOG_FILE_69)

# Assigning a Attribute to a Name (line 29):
# Getting the type of 'stypy_parameters_copy' (line 29)
stypy_parameters_copy_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'stypy_parameters_copy')
# Obtaining the member 'ENABLE_CODING_ADVICES' of a type (line 29)
ENABLE_CODING_ADVICES_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), stypy_parameters_copy_70, 'ENABLE_CODING_ADVICES')
# Assigning a type to the variable 'r23' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r23', ENABLE_CODING_ADVICES_71)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
