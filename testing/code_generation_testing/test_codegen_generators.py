from codegen_testing_common import TestCommon


class TestCodeGenGenerators(TestCommon):
    def test_basic_set_comprehensions(self):
        file_path = self.file_path + "/without_classes/generator_expression/simple_generator_expression.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_basic_if_expressions(self):
        file_path = self.file_path + "/without_classes/generator_expression/simple_if_expression.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_dict_generator_expression(self):
        file_path = self.file_path + "/without_classes/generator_expression/dict_generator_expression.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)