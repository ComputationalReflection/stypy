#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.types.undefined_type import UndefinedType
from stypy.ssa.ssa_context import SSAContext
from stypy.contexts.context import Context
from testing.model_testing.model_testing_common import compare_types
from stypy.reporting.localization import Localization


class TestSSATypes(unittest.TestCase):
    def setUp(self):
        parentContext = Context(None, __file__)
        self.context = SSAContext(parentContext, "func")
        self.localization = Localization(__file__, 1, 1)
        Localization.set_current(self.localization)
        
    # SIMPLE STORE - RETRIEVE

    def test_context_store(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.set_type_of(self.localization, "var3", str)

        compare_types(self.context.get_type_of(self.localization, "var1"), int)
        compare_types(self.context.get_type_of(self.localization, "var2"), float)
        compare_types(self.context.get_type_of(self.localization, "var3"), str)

    def test_nullify_ssa_types(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        compare_types(self.context.get_type_of(self.localization, "var1"), str)
        compare_types(self.context.get_type_of(self.localization, "var2"), str)

    def test_same_types(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), str)
        compare_types(self.context.get_type_of(self.localization, "var2"), str)

    # OUT AND IF

    def test_out_and_if(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str])

    def test_out_and_nested_if(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex])

    def test_out_and_nested_nested_if(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex, list])

    def test_out_and_nested_nested_if_some_branches(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex])

    # ONLY IF

    def test_only_if(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str])

    def test_only_nested_if(self):
        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str, complex])

    def test_only_nested_nested_if(self):
        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str, complex, list])

    def test_only_nested_nested_if_some_branches(self):
        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, complex])

    def test_only_maxdepth_nested_if(self):
        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, complex])

    # OUT AND ELSE

    def test_out_and_else(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str])

    def test_out_and_nested_else(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex])

    def test_out_and_nested_nested_else(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex, list])

    def test_out_and_nested_nested_else_some_branches(self):
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex])

    # ONLY ELSE

    def test_only_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str])

    def test_only_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str, complex])

    def test_only_nested_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, str, complex, list])

    def test_only_nested_nested_else_some_branches(self):
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, complex])

    def test_only_maxdepth_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [UndefinedType, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [UndefinedType, complex])

    # BOTH IF AND ELSE

    def test_both_if_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str])

    def test_both_if_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex])

    def test_both_if_nested_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex, list])

    def test_both_if_nested_nested_else_some_branches(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, complex, list, UndefinedType])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex, UndefinedType])

    def test_both_if_maxdepth_nested_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, list, UndefinedType])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex, UndefinedType])

    def test_nested_both_if_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        # Either if or else will be executed. Therefore the type of the variables will be overwritten no matter what
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [str, list, dict])
        compare_types(self.context.get_type_of(self.localization, "var2"), [str, list, dict])

    def test_nested_both_if_both_else(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", set)
        self.context.set_type_of(self.localization, "var2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", bytearray)
        self.context.set_type_of(self.localization, "var2", bytearray)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [list, dict, set, bytearray])
        compare_types(self.context.get_type_of(self.localization, "var2"), [list, dict, set, bytearray])

    def test_nested_both_if_both_else_nested_if(self):
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else  # else

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", set)
        self.context.set_type_of(self.localization, "var2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", bytearray)
        self.context.set_type_of(self.localization, "var2", bytearray)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [list, dict, set, bytearray, long])
        compare_types(self.context.get_type_of(self.localization, "var2"), [list, dict, set, bytearray, long])

    # OUT, IF AND ELSE

    def test_out_both_if_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str])

    def test_out_both_if_nested_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex])

    def test_out_both_if_nested_nested_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, str, complex, list])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, str, complex, list])

    def test_out_both_if_nested_nested_else_some_branches(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", complex)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, complex, list, long])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex, long])

    def test_out_both_if_maxdepth_nested_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [int, list, long])
        compare_types(self.context.get_type_of(self.localization, "var2"), [float, complex, long])

    def test_out_nested_both_if_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        # Either if or else will be executed. Therefore the type of the variables will be overwritten no matter what
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [str, list, dict])
        compare_types(self.context.get_type_of(self.localization, "var2"), [str, list, dict])

    def test_out_nested_both_if_both_else(self):
        self.context.set_type_of(self.localization, "var1", long)
        self.context.set_type_of(self.localization, "var2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", int)
        self.context.set_type_of(self.localization, "var2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", list)
        self.context.set_type_of(self.localization, "var2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", dict)
        self.context.set_type_of(self.localization, "var2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else

        self.context.set_type_of(self.localization, "var1", str)
        self.context.set_type_of(self.localization, "var2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of(self.localization, "var1", set)
        self.context.set_type_of(self.localization, "var2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of(self.localization, "var1", bytearray)
        self.context.set_type_of(self.localization, "var2", bytearray)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), [list, dict, set, bytearray])
        compare_types(self.context.get_type_of(self.localization, "var2"), [list, dict, set, bytearray])
