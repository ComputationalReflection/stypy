#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import unittest

from stypy.contexts.context import Context
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.localization import Localization
from stypy.types.undefined_type import UndefinedType
from stypy.types.union_type import UnionType
from testing.model_testing.model_testing_common import compare_types, assert_if_not_error


class TestUnionType(unittest.TestCase):
    def setUp(self):
        self.loc = Localization(__file__)
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()

    def test_create_union_type_with_vars(self):
        union = UnionType.add(int, str)
        compare_types(union, [int, str])

        union2 = UnionType(union, list)
        compare_types(union, [int, str])
        compare_types(union2, [int, str, list])

        self.assertFalse(union == union2)

        clone = UnionType.add(int, str)
        self.assertTrue(union == clone)

    def test_create_union_type_with_funcs(self):
        def foo():
            pass

        union = UnionType.add(foo, range)
        compare_types(union, [foo, range])

        union2 = UnionType(union, getattr)
        compare_types(union, [foo, range])
        compare_types(union2, [foo, range, getattr])

        self.assertFalse(union == union2)

        clone = UnionType.add(foo, range)
        self.assertTrue(union == clone)

    def test_create_union_type_with_classes(self):
        class Foo:
            pass

        union = UnionType.add(Foo, AssertionError)
        compare_types(union, [Foo, AssertionError])

        union2 = UnionType(union, object)
        compare_types(union2, [Foo, AssertionError, object])

        clone = UnionType.add(Foo, AssertionError)
        self.assertFalse(union == union2)
        self.assertTrue(union == clone)

    def test_create_union_type_with_instances(self):
        class Foo:
            pass

        foo_inst = Foo()
        assert_inst = AssertionError()
        object_inst = object()

        union = UnionType.add(foo_inst, assert_inst)
        compare_types(union, [foo_inst, assert_inst])

        union2 = UnionType(union, object_inst)
        compare_types(union2, [foo_inst, assert_inst, object_inst])

        clone = UnionType.add(foo_inst, assert_inst)
        self.assertFalse(union == union2)
        self.assertTrue(union == clone)

        clone2 = UnionType.add(foo_inst, AssertionError())
        self.assertFalse(union == clone2)

    def test_create_union_type_with_modules(self):
        import math
        import sys
        import types

        union = UnionType.add(math, sys)
        compare_types(union, [math, sys])

        clone = UnionType.add(math, sys)
        compare_types(union, clone.types)

        union2 = UnionType.add(clone, types)
        compare_types(union2, [math, sys, types])

        self.assertFalse(union == union2)

    def test_create_union_type_with_mixed_types(self):
        int_var = int

        def fun():
            pass

        class Foo:
            def method(self):
                pass

            def method_2(self):
                pass

        class_var = Foo
        method_var = Foo.method

        instance_var = Foo()
        import math

        module_var = math
        union = UnionType.create_from_type_list([int_var, fun, class_var, method_var, instance_var,
                                                 module_var])

        compare_types(union, [int_var, fun, class_var, method_var, instance_var,
                              module_var])

        clone = UnionType.create_from_type_list([int_var, fun, class_var, method_var, instance_var,
                                                 module_var])

        compare_types(union, clone)

        method2_var = Foo.method_2
        UnionType.add(clone, types)
        UnionType.add(clone, method2_var)

        compare_types(union, [int_var, fun, class_var, method_var, instance_var,
                              module_var])

        compare_types(clone, [int_var, fun, class_var, method_var, instance_var,
                              module_var, method2_var,
                              types
                              ])


    def test_merge_union_types(self):
        int_var = int

        def fun():
            pass

        class Foo:
            def method(self):
                pass

            def method_2(self):
                pass

        class_var = Foo
        method_var = Foo.method

        instance_var = Foo()
        import math

        module_var = math
        union1 = UnionType.create_from_type_list([int_var, fun, class_var])

        union2 = UnionType.create_from_type_list([method_var, instance_var,
                                                  module_var])

        compare_types(union1, [int_var, fun, class_var])

        compare_types(union2, [method_var, instance_var, module_var])

        fused_union = UnionType.add(union1, union2)
        compare_types(fused_union, [int_var, fun, class_var, method_var, instance_var,
                                    module_var])

        clone = UnionType.create_from_type_list([int_var, fun, class_var, method_var, instance_var,
                                                 module_var])
        compare_types(fused_union, clone)

        method2_var = Foo.method_2
        UnionType.add(clone, types)
        UnionType.add(clone, method2_var)

        compare_types(fused_union, [int_var, fun, class_var, method_var, instance_var,
                                    module_var])

        compare_types(clone, [int_var, fun, class_var, method_var, instance_var,
                              module_var, method2_var,
                              types])

        clone2 = UnionType.create_from_type_list([int_var, fun, class_var, method_var, instance_var,
                              module_var, method2_var,
                              types])
        self.assertFalse(fused_union == clone)
        compare_types(clone2, clone)

    # ############################## TYPE-BOUND TESTS ###############################

    def test_get_type_of_member(self):
        context = Context(None, __file__)
        # Access a member that none of the stored types has
        union1 = UnionType.add(int,
                               str)

        ret = union1.foo

        assert_if_not_error(ret)

        class Foo:
            att1 = int

            def method(self):
                pass

            def method_2(self):
                pass

        class Foo2:
            att1 = float

            def method(self):
                pass

            def method_2(self):
                pass

        # Access a member that can be provided only by some of the types in the union
        union2 = UnionType.add(Foo, str)

        ret = union2.method
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        TypeWarning.reset_warning_msgs()
        compare_types(ret, Foo.method)

        instance1 = Foo()
        union3 = UnionType.add(instance1, str)

        instance2 = Foo2()
        union3 = UnionType.add(instance2, union3)

        ret = union3.method
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        compare_types(ret, [instance1.method,
                            instance2.method,
                            ])

        TypeWarning.reset_warning_msgs()

        # Access a member that can be provided by all the types in the union
        union4 = UnionType.add(Foo, Foo2)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        ret = union4.method
        compare_types(ret, [Foo.method,
                            Foo2.method])

        ret = union4.att1
        compare_types(ret, [int, float])

        # Access a member that can be provided by all the types in the union (using only Python types)
        union5 = UnionType.add(int, str)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        context.set_type_of(self.loc, "union5", union5)

        # "__str__" is a special method so we have to use the context to access to it
        ret = context.get_type_of_member(self.loc, union5, "__str__")
        compare_types(ret, [int.__str__,
                            str.__str__])

    def test_set_type_of_member(self):
        # Set a member on non-modifiable types
        union1 = UnionType.add(int, str)

        union1.foo = int

        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 1)

        class Foo:
            def method(self):
                pass

            def method_2(self):
                pass

        class Foo2:
            def method(self):
                pass

            def method_2(self):
                pass

        # Set a member with some of the types of the union able to be modified
        union2 = UnionType.add(Foo, str)

        union2.member = str
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        # TypeWarning.print_warning_msgs()
        compare_types(union2.member, str)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 2)

        TypeWarning.reset_warning_msgs()

        instance1 = Foo()
        union3 = UnionType.add(instance1, str)

        instance2 = Foo2()
        union3 = UnionType.add(instance2, union3)

        union3.member = str

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        compare_types(union3.member, str)

        TypeWarning.reset_warning_msgs()
        StypyTypeError.reset_error_msgs()

        # Set a member using all-modifiable types
        union4 = UnionType.add(Foo, Foo2)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        union4.member = float
        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 0)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        ret = union4.member
        compare_types(ret, float)

        def obj_func(cls):
            pass

        union4.class_method = types.MethodType(obj_func, union4)

        ret = union4.class_method
        compare_types(ret, types.MethodType(obj_func, union4))

    # # ############################## DYNAMIC INHERITANCE ###############################

    def test_change_type(self):
        # Set a member on non-modifiable types
        union1 = UnionType.add(int, str)

        union1.__class__ = int

        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 1)

        class Foo:
            def method(self):
                pass

            def method_2(self):
                pass

        class Foo2:
            def method(self):
                pass

            def method_2(self):
                pass

        # Set a member with some of the types of the union able to be modified
        union2 = UnionType.add(Foo(), str)

        union2.__class__ = Foo2
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        # TypeWarning.print_warning_msgs()
        union_types = map(lambda x: type(x), union2.get_types())
        compare_types(union_types, [type(Foo2()), types.TypeType])

        TypeWarning.reset_warning_msgs()

        instance1 = Foo()
        union3 = UnionType.add(instance1, str)

        instance2 = Foo2()
        union3 = UnionType.add(instance2, union3)

        union3.__class__ = Foo2

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        union_types = map(lambda x: type(x), union2.get_types())
        compare_types(union_types, [type(Foo2()), types.TypeType])

        TypeWarning.reset_warning_msgs()
        StypyTypeError.reset_error_msgs()

        # Set a member using all-modifiable types
        foo1 = Foo()
        foo2 = Foo()
        union4 = UnionType.add(foo1, foo2)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 0)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        ret = union4
        compare_types(ret, foo1)

    def test_change_base_types(self):
        # Set a member on non-modifiable types
        union1 = UnionType.add(int, str)

        union1.__bases__ = (int,)

        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 1)

        class Foo:
            def method(self):
                pass

            def method_2(self):
                pass

        class Foo2:
            def method(self):
                pass

            def method_2(self):
                pass

        # Set a member with some of the types of the union able to be modified
        union2 = UnionType.add(Foo(), str)

        union2.__bases__ = (Foo2,)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        # TypeWarning.print_warning_msgs()
        compare_types(union2.__bases__.types[0].contained_types, Foo2)
        compare_types(union2.__bases__.types[1].contained_types, str.__bases__[0])

        TypeWarning.reset_warning_msgs()

        instance1 = Foo()
        union3 = UnionType.add(instance1, str)

        instance2 = Foo2()
        union3 = UnionType.add(instance2, union3)

        union3.__bases__ = (Foo2,)

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 1)

        compare_types(union3.__bases__.types[0].contained_types, Foo2)
        compare_types(union3.__bases__.types[1].contained_types, str.__bases__[0])

        #compare_types(union3.__bases__, [(Foo2,), str.__bases__])

        TypeWarning.reset_warning_msgs()
        StypyTypeError.reset_error_msgs()

        # Set a member using all-modifiable types
        union4 = UnionType.add(Foo(), Foo())

        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        union4.__bases__ = (Foo2,)
        self.assertTrue(len(StypyTypeError.get_error_msgs()) == 0)
        self.assertTrue(len(TypeWarning.get_warning_msgs()) == 0)

        ret = union4.__bases__
        compare_types(ret, (Foo2,))
