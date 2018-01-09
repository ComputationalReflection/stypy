#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print path
if not path in sys.path:
    sys.path.append(path)

from stypy.errors import type_error, type_warning
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestBasicScipy(TestCommon):
    def test_scipy_usage(self):
        file_path = self.file_path + "/scipy/scipy_simple_example.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)
