from codegen_testing_common import TestCommon


class TestCodeGenDynamic(TestCommon):
    def test_simple_eval(self):
        file_path = self.file_path + "/without_classes/exec_eval/simple_eval.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_simple_exec(self):
        file_path = self.file_path + "/without_classes/exec_eval/simple_exec.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_union_dynamic_type(self):
        file_path = self.file_path + "/without_classes/exec_eval/union_dynamic_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)