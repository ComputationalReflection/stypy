import sys

from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from testing.code_generation_testing.codegen_testing_common import TestCommon
from testing.testing_parameters import *


class TestCodeGeneration(TestCommon):
    def test_stypy(self):
        file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy/stypy_copy.py"
        sys.path = [STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy"] + sys.path
        result = self.run_stypy_with_program(file_path, generate_type_data_file=False, output_results=False)

        # self.print_errors()
        # self.print_stypy_modules_cache()
        self.assertEqual(result, 0)

    def print_errors(self):
        print "Found " + str(len(StypyTypeError.errors)) + " errors."
        for error in StypyTypeError.errors:
            print error

        print "Found " + str(len(TypeWarning.warnings)) + " warnings."
        for warning in TypeWarning.warnings:
            print warning

    def test_stypy_parameters(self):
        sys.path = [STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy"] + sys.path
        file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/individual_file_tests/stypy_parameters_test.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        # self.print_errors()
        self.assertEqual(result, 0)
    #
    # def test_source_code_writer(self):
    #     file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/individual_file_tests/code_generation/source_code_writer_test.py"
    #     result = self.run_stypy_with_program(file_path,  output_results=True)
    #
    #     #self.print_errors()
    #     self.assertEqual(result, 0)
    #
    def test_type_warning(self):
        sys.path = [STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy"] + sys.path
        file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/individual_file_tests/errors/type_warning_test.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.print_errors()
        self.assertEqual(result, 0)

    def test_known_python_types(self):
        sys.path = [STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy"] + sys.path
        file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/individual_file_tests/python_lib/python_types/instantiation/known_python_types_test.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.print_errors()
        self.assertEqual(result, 0)

    # def test_misc(self):
    #     sys.path = [STYPY_OVER_STYPY_PROGRAMS_PATH + "/stypy_code_copy"] + sys.path
    #     file_path = STYPY_OVER_STYPY_PROGRAMS_PATH + "/individual_file_tests/misc_test.py"
    #     result = self.run_stypy_with_program(file_path, output_results=True)
    #
    #     self.print_errors()
    #     self.assertEqual(result, 0)

        # def test_import_stypy(self):
        #     file_path = self.file_path + "/without_classes/imports/import_stypy.py"
        #     result = self.run_stypy_with_program(file_path, False)
        #
        #     self.assertEqual(result, 0)