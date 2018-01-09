#!/usr/bin/env python
# -*- coding: utf-8 -*-
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestShedSkin(TestCommon):
    def test_adatron(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/adatron.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)
