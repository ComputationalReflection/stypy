from codegen_testing_common import TestCommon


class TestCodeGenDicts(TestCommon):
    def test_basic_dict_comprehensions(self):
        file_path = self.file_path + "/without_classes/dict/dict_comprehensions.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)