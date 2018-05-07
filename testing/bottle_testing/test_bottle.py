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


class TestBottle(TestCommon):
    def test_bottle_usage(self):
        file_path = self.file_path + "/bottle/bottle.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)
