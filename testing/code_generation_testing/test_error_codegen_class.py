from codegen_testing_common import TestCommon


class TestErrorCodeGenerationClass(TestCommon):
    def test_error_class_default_constructor(self):
        file_path = self.file_path + "/with_classes/class_declaration/error_class_default_constructor.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_class_method_invokation(self):
        file_path = self.file_path + "/with_classes/class_declaration/error_class_method_invokation.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_static_methods(self):
        file_path = self.file_path + "/with_classes/class_methods/error_static_methods.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
