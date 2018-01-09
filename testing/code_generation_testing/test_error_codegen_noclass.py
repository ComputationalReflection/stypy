from codegen_testing_common import TestCommon
from stypy.errors import type_error
from stypy.errors import type_warning


class TestErrorCodeGeneration(TestCommon):
    def test_error_basic_arithmetic(self):
        file_path = self.file_path + "/without_classes/operators/error_basic_arithmetic_operators.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_error_operators_none(self):
        file_path = self.file_path + "/without_classes/operators/error_operators_none.py"
        result = self.run_stypy_with_program(file_path, output_results=False)

        self.assertEqual(result, 0)

    def test_error_type_conversion_methods(self):
        file_path = self.file_path + "/without_classes/operators/error_type_conversion_methods.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_classes_instances(self):
        file_path = self.file_path + "/without_classes/operators/error_classes_instances.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_invalid_variable_both_branches(self):
        file_path = self.file_path + "/without_classes/if_statements/error_invalid_variable_both_branches.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_variable_out_and_for(self):
        file_path = self.file_path + "/without_classes/for_statements/error_variable_out_and_for.py"
        result = self.run_stypy_with_program(file_path)

        # for w in type_warning.TypeWarning.get_warning_msgs():
        #     print w

        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)
        self.assertEqual(result, 0)

    def test_error_function_return_type_warning(self):
        file_path = self.file_path + "/without_classes/functions/error_function_return_type_warning.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)
        self.assertEqual(result, 0)

    def test_error_function_return_type(self):
        file_path = self.file_path + "/without_classes/functions/error_function_return_type.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_param_return(self):
        file_path = self.file_path + "/without_classes/functions/error_function_param_return.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_return_type_usage(self):
        file_path = self.file_path + "/without_classes/functions/error_function_return_type_usage.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_call_if(self):
        file_path = self.file_path + "/without_classes/functions/error_function_call_if.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 2)

    def test_error_function_call_if_both_error(self):
        file_path = self.file_path + "/without_classes/functions/error_function_call_if_both_error.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_kwargs(self):
        file_path = self.file_path + "/without_classes/functions/error_function_kwargs.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_loops(self):
        file_path = self.file_path + "/without_classes/functions/error_function_loops.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_args_kwargs(self):
        file_path = self.file_path + "/without_classes/functions/error_function_args_kwargs.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_iterable_param(self):
        file_path = self.file_path + "/without_classes/functions/error_function_iterable_param.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_function_iterable_param_class(self):
        file_path = self.file_path + "/without_classes/functions/error_function_iterable_param_class.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_iterable_class(self):
        file_path = self.file_path + "/without_classes/for_statements/error_iterable_class.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_library_calls(self):
        file_path = self.file_path + "/without_classes/python_library_calls/error_library_calls.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_library_class_conversion_methods(self):
        file_path = self.file_path + "/without_classes/python_library_calls/error_library_calls_conversion_methods.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_builtins_library_calls(self):
        file_path = self.file_path + "/without_classes/python_library_calls/error_builtins_library_calls.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_builtins_library_calls_iterables(self):
        file_path = self.file_path + "/without_classes/python_library_calls/error_builtins_library_calls_iterables.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_generators(self):
        file_path = self.file_path + "/without_classes/generator_expression/error_generators.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        # for w in type_warning.TypeWarning.get_warning_msgs():
        #     print w

        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_error_map(self):
        file_path = self.file_path + "/without_classes/higher_order/error_map.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_error_map_invalid_lists(self):
        file_path = self.file_path + "/without_classes/higher_order/error_map_invalid_lists.py"
        result = self.run_stypy_with_program(file_path)

        # for w in type_warning.TypeWarning.get_warning_msgs():
        #     print w

        self.assertEqual(result, 0)

    def test_error_filter(self):
        file_path = self.file_path + "/without_classes/higher_order/error_filter.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_error_reduce(self):
        file_path = self.file_path + "/without_classes/higher_order/error_reduce.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_slice_bounds(self):
        file_path = self.file_path + "/without_classes/slice/error_slice_bounds.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_get_slice(self):
        file_path = self.file_path + "/without_classes/slice/error_get_slice.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_add_attributes(self):
        file_path = self.file_path + "/without_classes/structural/error_add_attributes.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_add_attributes_method_call(self):
        file_path = self.file_path + "/without_classes/structural/error_add_attributes_method_call.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_del_attributes(self):
        file_path = self.file_path + "/without_classes/structural/error_del_attributes.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEquals(len(type_error.StypyTypeError.get_error_msgs()), 2)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 2)

        self.assertEqual(result, 0)

    def test_error_alias(self):
        file_path = self.file_path + "/without_classes/alias/error_alias.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_assignments(self):
        file_path = self.file_path + "/without_classes/assignments/error_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)

    def test_error_augment_assignments(self):
        file_path = self.file_path + "/without_classes/assignments/error_augment_assignments.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
        # for e in type_error.StypyTypeError.get_error_msgs():
        #      print e
        self.assertEquals(len(type_error.StypyTypeError.get_error_msgs()), 2)

    def test_error_comparations(self):
        file_path = self.file_path + "/without_classes/comparations/error_comparations.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_list(self):
        file_path = self.file_path + "/without_classes/list/error_list.py"
        result = self.run_stypy_with_program(file_path)

        # for w in type_error.TypeError.get_error_msgs():
        #     print w
        #
        # for w in type_warning.TypeWarning.get_warning_msgs():
        #     print w

        self.assertEqual(result, 0)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 2)

    def test_error_dict(self):
        file_path = self.file_path + "/without_classes/dict/error_dict.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)

    def test_error_try_except(self):
        file_path = self.file_path + "/without_classes/exceptions/error_try_except.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEquals(len(type_error.StypyTypeError.get_error_msgs()), 1)
        self.assertEquals(len(type_warning.TypeWarning.get_warning_msgs()), 1)
        self.assertEqual(result, 0)

    def test_error_try_except_else_finally(self):
        file_path = self.file_path + "/without_classes/exceptions/error_try_except_else_finally.py"
        result = self.run_stypy_with_program(file_path)

        self.assertEqual(result, 0)
