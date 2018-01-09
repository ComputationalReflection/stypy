#!/usr/bin/env python
# -*- coding: utf-8 -*-
import operator
import types
import unittest

from stypy.contexts.context import Context
from stypy.type_inference_programs import stypy_interface
from stypy.types.type_intercession import change_base_types, change_type, supports_intercession
from testing.model_testing.model_testing_common import *


class ExtraTypeDefinitions:
    def __init__(self, localization):
        pass

    """
    Additional (not included) type definitions to those defined in the types Python module. This class is needed
    to have an usable type object to refer to when generating Python code
    """
    SetType = set
    iterator = type(iter(""))

    setiterator = type(iter(set()))
    tupleiterator = type(iter(tuple()))
    rangeiterator = type(iter(xrange(1)))
    listiterator = type(iter(list()))
    callable_iterator = type(iter(type(int), 0.1))
    listreverseiterator = type(iter(reversed(list())))
    methodcaller = type(operator.methodcaller(0))
    itemgetter = type(operator.itemgetter(0))
    attrgetter = type(operator.attrgetter(0))

    dict_items = type(dict({"a": 1, "b": 2}).viewitems())
    dict_keys = type(dict({"a": 1, "b": 2}).viewkeys())
    dict_values = type(dict({"a": 1, "b": 2}).viewvalues())

    dictionary_keyiterator = type(iter(dict({"a": 1, "b": 2})))
    dictionary_itemiterator = type(dict({"a": 1, "b": 2}).iteritems())
    dictionary_valueiterator = type(dict({"a": 1, "b": 2}).itervalues())
    bytearray_iterator = type(iter(bytearray("test")))

    # Extra builtins without instance counterparts
    getset_descriptor = type(ArithmeticError.message)
    member_descriptor = type(IOError.errno)
    formatteriterator = type(u"foo"._formatter_parser())


class TestPythonTypeReflection(unittest.TestCase):
    def setUp(self):
        self.type_store = Context(None, __file__)

        class OldStyleBase:
            base_att = "str"

            def __init__(self, localization):
                pass

            @staticmethod
            def base_staticmethod(localization):
                return "foo"

            def base_method(self, localization):
                return 0

        class OldStyleBase2:
            base2_att = "str"

            def __init__(self, localization):
                pass

            @staticmethod
            def base2_staticmethod(localization):
                return "foo"

            def base2_method(self, localization):
                return 0

        class OldStyleC(OldStyleBase):
            att = 3

            def __init__(self, localization):
                pass

            @staticmethod
            def staticmethod(c):
                return "foo"

            def method(self, new_C_instance):
                return 0

        def inst_method(self, localization):
            return list()

        class NewStyleBase2(object):
            base_att = "str"

            def __init__(self, localization):
                pass

            @staticmethod
            def base2_staticmethod(localization):
                return "foo"

            def base2_method(self, localization):
                return 0

        class NewStyleBase(object):
            base_att = "str"

            def __init__(self, localization):
                pass

            @staticmethod
            def base_staticmethod(localization):
                return "foo"

            def base_method(self, localization):
                return 0

        class NewStyleC(NewStyleBase):
            att = 3

            def __init__(self, localization):
                pass

            @staticmethod
            def staticmethod(localization):
                return "foo"

            def method(self, localization):
                return 0

        self.loc = Localization(__file__)
        self.old_Base_class = OldStyleBase
        self.old_Base2_class = OldStyleBase2

        old_inst = OldStyleC(None)
        old_inst.inst_att = True
        old_inst.inst_met = types.MethodType(inst_method, old_inst)

        self.old_C_class = OldStyleC
        self.old_C_instance = invoke(None, self.old_C_class)
        self.old_C_instance_mod = old_inst

        self.new_Base_class = NewStyleBase
        self.new_Base2_class = NewStyleBase2

        new_inst = NewStyleC(None)
        new_inst.inst_att = True
        new_inst.inst_met = types.MethodType(inst_method, new_inst)

        self.new_C_class = NewStyleC
        self.new_C_instance = invoke(None, self.new_C_class)
        self.new_C_instance_mod = new_inst

        self.builtins_ = stypy_interface.builtins_module

    # ########################################## TESTS ###########################################

    def test_introspection_class_attribute(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "att")
        compare_types(type(temp), types.IntType)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "att")
        compare_types(type(temp), types.IntType)
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "att")
        compare_types(type(temp), types.IntType)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "att")
        compare_types(type(temp), types.IntType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "nonexisting")
        assert_if_not_error(temp)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "nonexisting")
        assert_if_not_error(temp)

    def test_introspection_class_method(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "nonexisting")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "method")
        compare_types(type(temp), types.MethodType)
        res = invoke(None, temp)
        assert_if_not_error(res)

        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "nonexisting")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

    def test_introspection_base_class_attribute(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "base_att")
        compare_types(type(temp), types.StringType)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "base_att")
        compare_types(type(temp), types.StringType)
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "base_att")
        compare_types(type(temp), types.StringType)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "base_att")
        compare_types(type(temp), types.StringType)

    def test_introspection_base_class_method(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "base_staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.new_C_class, "base_staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_class, "base_method")
        compare_types(type(temp), types.MethodType)
        res = invoke(None, temp)
        assert_if_not_error(res)

    def test_introspection_instance_attribute(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "att")
        compare_types(type(temp), types.IntType)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "att")
        compare_types(type(temp), types.IntType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "nonexisting")
        assert_if_not_error(temp)
        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "nonexisting")
        assert_if_not_error(temp)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance_mod, "inst_att")
        compare_types(type(temp), types.BooleanType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "inst_att")
        assert_if_not_error(temp)

    def test_introspection_instance_method(self):
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "nonexisting")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "staticmethod")
        compare_types(type(temp), types.FunctionType)
        res = invoke(None, temp)
        compare_types(type(res), types.StringType)

        temp = self.type_store.get_type_of_member(self.loc, self.new_C_instance, "nonexisting")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance_mod, "inst_met")
        compare_types(type(temp), types.MethodType)
        res = invoke(None, temp)
        compare_types(res, types.ListType)

        temp = self.type_store.get_type_of_member(self.loc, self.old_C_instance, "inst_met")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

    def test_structural_reflection_modules(self):
        res = self.type_store.set_type_of_member(self.loc, self.builtins_, "foo", str())
        self.assertTrue(supports_intercession(self.builtins_))
        compare_types(type(res), types.NoneType)

        res = self.type_store.get_type_of_member(self.loc, self.builtins_, 'foo')
        compare_types(type(res), types.StringType)

    def test_structural_reflection_non_modifiable_classes(self):
        int_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'int')
        res = self.type_store.set_type_of_member(self.loc, int_, "foo", str)
        assert_if_not_error(res)

        list_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'list')
        res = self.type_store.set_type_of_member(self.loc, list_, "foo", str)
        assert_if_not_error(res)

    def test_structural_reflection_modifiable_classes(self):
        u_d = ExtraTypeDefinitions
        setit = self.type_store.get_type_of_member(self.loc, u_d, 'setiterator')
        self.type_store.set_type_of_member(self.loc, setit, "foo", str)

        self.type_store.set_type_of_member(self.loc, u_d, "foo", str)
        foo_class = self.type_store.get_type_of_member(self.loc, u_d, "foo")
        compare_types(foo_class, types.StringType)

        self.type_store.set_type_of_member(self.loc, setit, "foo", str)
        foo_class = self.type_store.get_type_of_member(self.loc, setit, "foo")
        assert_if_not_error(foo_class)

    def test_structural_reflection_modifiable_classes_methods(self):
        class Foo:
            pass

        def func(self, localization):
            return getattr(self, "class_property")

        foo_class = Foo
        self.type_store.set_type_of_member(self.loc, foo_class, "class_property", int)
        self.type_store.set_type_of_member(self.loc, foo_class, "class_method", types.MethodType(func, foo_class))

        compare_types(self.type_store.get_type_of_member(self.loc, foo_class, "class_property"), types.IntType)
        class_method = self.type_store.get_type_of_member(self.loc, foo_class, "class_method")
        compare_types(type(class_method), types.MethodType)
        compare_types(invoke(None, class_method), types.IntType)

    def test_structural_reflection_modifiable_instances(self):
        u_d = ExtraTypeDefinitions
        self.type_store.set_type_of_member(self.loc, u_d, "foo", str)
        setit = self.type_store.get_type_of_member(self.loc, u_d, 'SetType')

        set_instance = invoke(None, setit)
        self.type_store.set_type_of_member(self.loc, set_instance, "foo", str)
        res = self.type_store.get_type_of_member(self.loc, set_instance, "foo")
        assert_if_not_error(res)

        u_d_instance = invoke(None, u_d)
        self.type_store.set_type_of_member(self.loc, u_d_instance, "bar", bool)

        foo_inst = self.type_store.get_type_of_member(self.loc, u_d_instance, "bar")
        compare_types(foo_inst, types.BooleanType)

        foo_inst = self.type_store.get_type_of_member(self.loc, u_d_instance, "foo")
        compare_types(foo_inst, types.StringType)

        u_d_instance2 = invoke(None, u_d)

        foo_inst = self.type_store.get_type_of_member(self.loc, u_d_instance2, "bar")
        assert_if_not_error(foo_inst)
        foo_inst = self.type_store.get_type_of_member(self.loc, u_d_instance2, "foo")
        compare_types(foo_inst, types.StringType)

    def test_structural_reflection_modifiable_instances_methods(self):
        class Foo:
            def __init__(self, localization):
                pass

        def func(self, localization):
            return getattr(self, "class_property")

        def obj_func(self, localization):
            return getattr(self, "instance_property")

        foo_class = Foo
        self.type_store.set_type_of_member(self.loc, foo_class, "class_property", int)
        self.type_store.set_type_of_member(self.loc, foo_class, "class_method", types.MethodType(func, foo_class))

        foo_instance = invoke(None, foo_class)
        self.type_store.set_type_of_member(self.loc, foo_instance, "instance_property", str)
        self.type_store.set_type_of_member(self.loc, foo_instance, "instance_method",
                                           types.MethodType(obj_func, foo_instance))

        # Class
        compare_types(self.type_store.get_type_of_member(self.loc, foo_class, "class_property"), types.IntType)
        class_method = self.type_store.get_type_of_member(self.loc, foo_class, "class_method")
        compare_types(type(class_method), types.MethodType)
        compare_types(invoke(None, class_method), types.IntType)

        # Instance
        compare_types(self.type_store.get_type_of_member(self.loc, foo_instance, "class_property"), types.IntType)
        class_method = self.type_store.get_type_of_member(self.loc, foo_instance, "class_method")
        compare_types(type(class_method), types.MethodType)
        compare_types(invoke(None, class_method), types.IntType)

        compare_types(self.type_store.get_type_of_member(self.loc, foo_instance, "instance_property"), types.StringType)
        instance_method = self.type_store.get_type_of_member(self.loc, foo_instance, "instance_method")
        compare_types(type(instance_method), types.MethodType)
        compare_types(invoke(None, instance_method), types.StringType)

    def test_dynamic_inheritance_change_class_hierarchy_builtin(self):
        list_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'list')
        res = change_base_types(self.loc, list_, (self.old_Base2_class,))
        assert_if_not_error(res)

        range_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'range')
        res = change_base_types(self.loc, range_, (self.old_Base2_class,))
        assert_if_not_error(res)

    def test_dynamic_inheritance_change_class_hierarchy_user_defined(self):
        u_d = ExtraTypeDefinitions
        u_d_instance = invoke(None, u_d)

        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_att")
        assert_if_not_error(temp)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_method")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

        res = change_base_types(self.loc, u_d, (self.old_Base2_class,))
        compare_types(type(res), types.NoneType)

        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_att")
        compare_types(type(temp), types.StringType)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_method")
        compare_types(type(temp), types.MethodType)
        res = invoke(None, temp)
        compare_types(type(res), types.IntType)

        res = change_base_types(self.loc, u_d, tuple())
        compare_types(type(res), types.NoneType)

    def test_dynamic_inheritance_change_type_builtin(self):
        list_ = self.type_store.get_type_of_member(self.loc, self.builtins_, 'list')

        list_inst = invoke(None, list_)
        res = change_type(self.loc, list_inst, (self.old_Base2_class,))

        assert_if_not_error(res)

    def test_dynamic_inheritance_change_type_user_defined(self):
        u_d = ExtraTypeDefinitions
        u_d_instance = invoke(None, u_d)
        u_d_instance2 = invoke(None, u_d)

        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_att")
        assert_if_not_error(temp)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_method")
        assert_if_not_error(temp)
        res = invoke(None, temp)
        assert_if_not_error(res)

        res = change_type(self.loc, u_d_instance, self.old_Base2_class)
        compare_types(type(res), types.NoneType)

        # Old-new tyle conflict
        res = change_type(self.loc, u_d_instance, self.new_Base2_class)
        assert_if_not_error(res)

        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_att")
        compare_types(type(temp), types.StringType)

        # Test modified instance and other instances and class
        temp = self.type_store.get_type_of_member(self.loc, u_d_instance, "base2_method")
        compare_types(type(temp), types.MethodType)
        res = invoke(None, temp)
        compare_types(type(res), types.IntType)

        temp = self.type_store.get_type_of_member(self.loc, u_d_instance2, "base2_att")
        assert_if_not_error(temp)

        # Test modified instance and other instances and class
        res = self.type_store.get_type_of_member(self.loc, u_d_instance2, "base2_method")
        assert_if_not_error(res)
        res = invoke(None, temp)
        assert_if_not_error(res)
