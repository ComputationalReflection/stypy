from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestCodeGenClass(TestCommon):
    def test_simple_class_declaration(self):
        file_path = self.file_path + "/with_classes/class_declaration/simple_class.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_instantation_constructor_parameters(self):
        file_path = self.file_path + "/with_classes/class_declaration/class_instantation_constructor_parameters.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_in_function(self):
        file_path = self.file_path + "/with_classes/class_declaration/class_in_function.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_assigments(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_assigments_from_self(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_assignments_from_self.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_assigments_methods(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_assignments_methods.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_assigments_method_calls(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_assignments_method_calls.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_assigments_method_code(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_assignments_method_code.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_assigments_attributes(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_assignments_attributes.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_nested_assigments(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_nested_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_multiple_assigments(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_multiple_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_tuple_assigments_2(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_tuple_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_simple_class_instantiation_and_usage(self):
        file_path = self.file_path + "/with_classes/class_declaration/simple_class_instantiation_and_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_constructor_assignments(self):
        file_path = self.file_path + "/with_classes/class_declaration/class_constructor_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_method_invokation(self):
        file_path = self.file_path + "/with_classes/class_methods/class_method_invokation.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_method_flow_sensitive(self):
        file_path = self.file_path + "/with_classes/class_methods/class_method_flow_sensitive.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_static_methods(self):
        file_path = self.file_path + "/with_classes/class_methods/static_methods.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_simple_class_multiple_inheritance(self):
        file_path = self.file_path + "/with_classes/class_declaration/simple_class_multiple_inheritance.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_method_attribute(self):
        file_path = self.file_path + "/with_classes/class_declaration/class_method_attribute.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_class_att_multiple_assignments_function(self):
        file_path = self.file_path + "/with_classes/class_assignments/class_att_multiple_assignments_function.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)