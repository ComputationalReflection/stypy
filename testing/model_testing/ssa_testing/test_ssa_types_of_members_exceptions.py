#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.reporting.localization import Localization
from stypy.ssa.ssa_context import SSAContext
from stypy.types.undefined_type import UndefinedType
from testing.model_testing.model_testing_common import compare_types


class ClassA(object):
    # class_attribute = int

    def __init__(self, type_):
        self.instance_attribute = type_


class ClassB(object):
    # class_attribute = float

    def __init__(self, type_):
        self.instance_attribute = type_


class TestSSATypesExceptionsTypesOfMember(unittest.TestCase):
    """
    As exceptions, loops and ifs are equally handled when dealing with SSA algorithm, we only test here the multi-branch
    SSA functionality. Context and branch names are purely cosmetic (except the finally branch)
    """

    def setUp(self):
        try:
            delattr(ClassA, "class_attribute1")
        except:
            pass
        try:
            delattr(ClassA, "class_attribute2")
        except:
            pass
        try:
            delattr(ClassA, "class_attribute3")
        except:
            pass

        self.context = SSAContext(None)
        self.localization = Localization(__file__, 1, 1)

        self.context.set_type_of(self.localization, "var1", ClassA)
        self.context.set_type_of(self.localization, "var2", ClassA)

    def test_try_except(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"), [int, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, str])

    def test_try_multiple_except2(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"), [int, str, long])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, str, long])

    def test_try_multiple_except3(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"),
                      [int, str, long, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, long, complex])

    def test_try_multiple_except4(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"),
                      [int, str, long, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, long, complex, list])

    def test_try_multiple_except5_if_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("except")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", set)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", set)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"),
                      [int, str, long, complex, list, dict, set])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, long, complex, list, dict, set])

    def test_try_multiple_except5_if_only(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("except")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"),
                      [int, str, long, complex, list, dict, UndefinedType])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, long, complex, list, dict, UndefinedType])

    def test_try_except_finally(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("finally")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute3", list)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"), list)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), list)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute3"), list)

    def test_try_multiple_except4_finally(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"), dict)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), dict)

    def test_try_multiple_except4_if_only_finally(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"),
                      [int, str, long, complex, list, dict])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, long, complex, list, dict])

    def test_try_multiple_except4_if_else_finally(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("try")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)
        self.context.open_ssa_branch("except")  # except
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("finally")  # except
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context.open_ssa_branch("else")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute1", set)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", set)
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute1"), [dict, set])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [dict, set])
