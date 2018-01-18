#!/usr/bin/env python
# -*- coding: utf-8 -*-
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestShedSkin(TestCommon):
    def test_adatron(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/adatron.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_ac_encode(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/ac_encode.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_amaze(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/amaze.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_ant(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/ant.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_bh(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/bh.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_block(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/block.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)