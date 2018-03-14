from codegen_testing_common import TestCommon


class TestCodeGenerationIdioms(TestCommon):
    def test_idiom_simple_if_type(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_if_else_type(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_if_else_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_nested_if_type(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_nested_if_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_not_type(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_not_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_isinstance(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_isinstance.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_hasattr(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_hasattr.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_hasattr_variants(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_hasattr_variants.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_not_hasattr_variants(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_not_hasattr_variants.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_not_type_variants(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_not_type_variants.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_simple_if_type_variants(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_simple_if_type_variants.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_is_none(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_is_none.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_is_none_variants(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_is_none_variants.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_while_true(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_while_true.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_idiom_if_none_break_or_return(self):
        file_path = self.file_path + "/without_classes/idioms/idiom_if_none_break_or_return.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)
