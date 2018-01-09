
from stypy.code_generation.type_inference_programs.checking.type_data_file_writer import TypeDataFileWriter

type_test = TypeDataFileWriter(__file__)
from stypy_copy import stypy_parameters_copy

r1 = stypy_parameters_copy.type_inference_file_directory_name
r2 = stypy_parameters_copy.type_inference_file_postfix
r3 = stypy_parameters_copy.type_modifier_file_postfix
r4 = stypy_parameters_copy.type_data_autogenerator_file_postfix
r5 = stypy_parameters_copy.type_data_file_postfix
r6 = stypy_parameters_copy.type_annotation_file_postfix
r7 = stypy_parameters_copy.type_rule_file_postfix
source_file_path = (stypy_parameters_copy.ROOT_PATH + '/stypy.py')
r8 = stypy_parameters_copy.go_to_parent_folder(0, source_file_path)
r9 = stypy_parameters_copy.get_original_program_from_type_inference_file(source_file_path)
r10 = stypy_parameters_copy.get_stypy_type_inference_program_file_path(source_file_path)
r11 = stypy_parameters_copy.get_stypy_type_data_autogenerator_program_file_path(source_file_path)
r12 = stypy_parameters_copy.get_stypy_type_data_file_path(source_file_path)
r13 = stypy_parameters_copy.get_stypy_type_annotation_file_path(source_file_path)
r14 = stypy_parameters_copy.PYTHON_EXE_PATH
r15 = stypy_parameters_copy.PYTHON_EXE
r16 = stypy_parameters_copy.ROOT_PATH
r17 = stypy_parameters_copy.TYPE_INFERENCE_PATH
r18 = stypy_parameters_copy.RULE_FILE_PATH
r19 = stypy_parameters_copy.LOG_PATH
r20 = stypy_parameters_copy.ERROR_LOG_FILE
r21 = stypy_parameters_copy.WARNING_LOG_FILE
r22 = stypy_parameters_copy.INFO_LOG_FILE
r23 = stypy_parameters_copy.ENABLE_CODING_ADVICES
type_test.add_type_dict_for_main_context(globals())
type_test.generate_type_data_file()