#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.reporting.localization import Localization
from stypy.ssa.ssa_context import SSAContext
from stypy.types.undefined_type import UndefinedType
from testing.model_testing.model_testing_common import compare_types


class ClassA(object):
    def __init__(self, type_):
        self.instance_attribute = type_


class ClassB(object):
    def __init__(self, type_):
        self.instance_attribute = type_


class TestSSATypesOfMembersClasses(unittest.TestCase):
    def setUp(self):
        try:
            delattr(ClassA, "class_attribute")
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

        self.localization = Localization(__file__, 1, 1)
        self.context = SSAContext(None)
        self.context.set_type_of(self.localization, "var1", ClassA)
        self.context.set_type_of(self.localization, "var2", ClassA)

    # SIMPLE STORE - RETRIEVE

    def test_context_store(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute3", str)

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), int)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), float)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute3"), str)

    def test_nullify_ssa_types(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), str)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), str)

    def test_same_types(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), str)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), str)

    def test_two_classes(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", ClassA)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", ClassA)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", ClassB)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [ClassA, ClassB])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), ClassA)

    # OUT AND IF

    def test_out_and_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", float)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), float)

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of(self.localization, "var1"), ClassA)
        compare_types(self.context.get_type_of(self.localization, "var2"), ClassA)
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, float])

    def test_out_and_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex])

    def test_out_and_nested_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex, list])

    def test_out_and_nested_nested_if_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, complex])

    # ONLY IF

    def test_only_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str])

    def test_only_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str, complex])

    def test_only_nested_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str, complex, list])

    def test_only_nested_nested_if_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, complex])

    def test_only_maxdepth_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context = self.context.open_ssa_context("if")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, complex])

    # OUT AND ELSE

    def test_out_and_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, str])

    def test_out_and_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex])

    def test_out_and_nested_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex, list])

    def test_out_and_nested_nested_else_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, complex])

    # ONLY ELSE

    def test_only_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str])

    def test_only_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str, complex])

    def test_only_nested_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, str, complex, list])

    def test_only_nested_nested_else_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, complex])

    def test_only_maxdepth_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [UndefinedType, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [UndefinedType, complex])

    # BOTH IF AND ELSE

    def test_both_if_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, str])

    def test_both_if_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex])

    def test_both_if_nested_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex, list])

    def test_both_if_nested_nested_else_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, complex, list, UndefinedType])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, complex, UndefinedType])

    def test_both_if_maxdepth_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, list, UndefinedType])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, complex, UndefinedType])

    def test_nested_both_if_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        # Either if or else will be executed. Therefore the type of the variables will be overwritten no matter what
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [str, list, dict])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [str, list, dict])

    def test_nested_both_if_both_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", set)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", bytearray)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", bytearray)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [list, dict, set, bytearray])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [list, dict, set, bytearray])

    def test_nested_both_if_both_else_nested_if(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else  # else

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", set)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", bytearray)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", bytearray)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [list, dict, set, bytearray, long])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [list, dict, set, bytearray, long])

    # OUT, IF AND ELSE

    def test_out_both_if_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [float, str])

    def test_out_both_if_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, str, complex])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex])

    def test_out_both_if_nested_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, str, complex, list])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, str, complex, list])

    def test_out_both_if_nested_nested_else_some_branches(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", complex)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [int, complex, list, long])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, complex, long])

    def test_out_both_if_maxdepth_nested_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context = self.context.open_ssa_context("if")
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", complex)

        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()
        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [int, list, long])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [float, complex, long])

    def test_out_nested_both_if_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        # Either if or else will be executed. Therefore the type of the variables will be overwritten no matter what
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"), [str, list, dict])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"), [str, list, dict])

    def test_out_nested_both_if_both_else(self):
        temp1 = self.context.get_type_of(self.localization, "var1")

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", long)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", long)
        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", int)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", float)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", list)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", list)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", dict)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", dict)
        self.context = self.context.join_ssa_context()

        self.context.open_ssa_branch("else")  # else

        self.context.set_type_of_member(self.localization, temp1, "class_attribute", str)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", str)

        self.context = self.context.open_ssa_context("if")
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", set)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", set)
        self.context.open_ssa_branch("else")  # else
        self.context.set_type_of_member(self.localization, temp1, "class_attribute", bytearray)
        self.context.set_type_of_member(self.localization, temp1, "class_attribute2", bytearray)
        self.context = self.context.join_ssa_context()

        self.context = self.context.join_ssa_context()

        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute"),
                      [list, dict, set, bytearray])
        compare_types(self.context.get_type_of_member(self.localization, temp1, "class_attribute2"),
                      [list, dict, set, bytearray])
