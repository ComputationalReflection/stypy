

from testing.code_generation_testing.codegen_testing_common import TestCommon
from stypy.module_imports.python_library_modules import is_python_library_module


class TestImport(TestCommon):
    def setUp(self):
        super(TestImport, self).setUp()
        import sys, copy
        self.after_sys_modules = copy.copy(sys.modules.keys())

    def tearDown(self):
        import sys, copy
        self.before_sys_modules = copy.copy(sys.modules.keys())
        for mod in self.before_sys_modules:
            if mod not in self.after_sys_modules:
                if not is_python_library_module(mod):
                    del sys.modules[mod]

    def test_import_var_and_func_non_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_var_and_func_non_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_var_and_func_non_python_module_folder(self):
        file_path = self.file_path + "/without_classes/imports/import_var_and_func_non_python_module_folder.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_all_non_python_module_folder(self):
        file_path = self.file_path + "/without_classes/imports/import_all_non_python_module_folder.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_all_non_python_module_folder2(self):
        file_path = self.file_path + "/without_classes/imports/import_all_non_python_module_folder2.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_all_non_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_all_non_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_var_and_func_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_var_and_func_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_all_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_all_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_non_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_non_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_future(self):
        file_path = self.file_path + "/without_classes/imports/import_future.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_python_module(self):
        file_path = self.file_path + "/without_classes/imports/import_python_module.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_non_python_module_nested(self):
        file_path = self.file_path + "/without_classes/imports/import_non_python_module_nested.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_non_python_module_nested_mixed(self):
        file_path = self.file_path + "/without_classes/imports/import_non_python_module_nested_mixed.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_multiple_modules(self):
        file_path = self.file_path + "/without_classes/imports/import_multiple_modules.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_relative(self):
        file_path = self.file_path + "/without_classes/imports/relative/import_non_python_module_relative_main.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_python_3_or_2(self):
        file_path = self.file_path + "/without_classes/imports/import_python_3_or_2.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_import_alias(self):
        file_path = self.file_path + "/without_classes/imports/import_alias.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_import_in_try(self):
        file_path = self.file_path + "/without_classes/imports/import_in_try.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)