#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.reporting.localization import Localization
from stypy.ssa.ssa_context import SSAContext
from stypy.types.undefined_type import UndefinedType
from testing.model_testing.model_testing_common import compare_types


class TestSSATypesExceptions(unittest.TestCase):
    """
    As exceptions, loops and ifs are equally handled when dealing with SSA algorithm, we only test here the multi-branch
    SSA functionality. Context and branch names are purely cosmetic (except the finally branch)
    """

    def setUp(self):
        self.context = SSAContext(None)
        self.localization = Localization(__file__, 1, 1)

    def test_try_except(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str])

    def test_try_multiple_except2(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, long])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, long])

    def test_try_multiple_except3(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, long, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, long, complex])

    def test_try_multiple_except4(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, long, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, long, complex, list])

    def test_try_multiple_except5_if_else(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("except")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", set)
        self.context.set_type_of(self.localization, "var2", set)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, long, complex, list, dict, set])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, long, complex, list, dict, set])

    def test_try_multiple_except5_if_only(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("except")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"),
                      [int, str, long, complex, list, dict, UndefinedType])
        compare_types(self.context.get_type_of(self.localization, "var2"),
                      [float, str, long, complex, list, dict, UndefinedType])

    def test_try_except_finally(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("finally")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.set_type_of(self.localization, "var3", list)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), list)
        compare_types(self.context.get_type_of(self.localization, "var2"), list)
        compare_types(self.context.get_type_of(self.localization, "var3"), list)

    def test_try_multiple_except4_finally(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), dict)
        compare_types(self.context.get_type_of(self.localization, "var2"), dict)

    def test_try_multiple_except4_if_only_finally(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, long, complex, list, dict])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, long, complex, list, dict])

    def test_try_multiple_except4_if_else_finally(self):
        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context.open_ssa_branch("else")
        self.context.set_type_of(self.localization, "var1", set)
        self.context.set_type_of(self.localization, "var2", set)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [dict, set])
        compare_types(self.context.get_type_of(self.localization, "var2"), [dict, set])
