from codegen_testing_common import TestCommon


class TestCodeGenListsTuples(TestCommon):
    def test_basic_list_comprehensions(self):
        file_path = self.file_path + "/without_classes/list/list_comprehensions.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_list_comprehensions(self):
        file_path = self.file_path + "/without_classes/list/list_comprehensions_2.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_tuples(self):
        file_path = self.file_path + "/without_classes/tuple/tuples.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_other_containers(self):
        file_path = self.file_path + "/without_classes/list/other_containers.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)