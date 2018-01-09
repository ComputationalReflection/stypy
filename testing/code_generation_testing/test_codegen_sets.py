from codegen_testing_common import TestCommon


class TestCodeGenSets(TestCommon):
    def test_basic_set_comprehensions(self):
        file_path = self.file_path + "/without_classes/set/set_comprehensions.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)