from codegen_testing_common import TestCommon


class TestCodeGenerationSlices(TestCommon):
    def test_ellipsis_usage(self):
        file_path = self.file_path + "/without_classes/ellipsis/ellipsis_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)