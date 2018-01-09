from codegen_testing_common import TestCommon


class TestCodeGenClass(TestCommon):
    def test_class_inherits_union_type(self):
        file_path = self.file_path + "/with_classes/class_declaration/class_inherits_union.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_class_inheritance(self):
        file_path = self.file_path + "/with_classes/class_declaration/error_class_inheritance.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)