#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.contexts.context import Context
from stypy.type_inference_programs import stypy_interface
from testing.model_testing.model_testing_common import *


class TestTuples(unittest.TestCase):
    def __obtain_tuple_item(self, tuple_name, return_var_name):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), tuple_name)
        __temp_2 = get_contained_elements_type(__temp_1)
        self.type_store.set_type_of(Localization(__file__), return_var_name, __temp_2)

    def __create_tuple(self, type_store, name, types):
        tuple_ = stypy_interface.get_builtin_python_type(self.loc, "tuple")

        list_ = stypy_interface.get_builtin_python_type_instance(self.loc, "list")
        append_method = self.type_store.get_type_of_member(self.loc, list_, "append")
        for type in types:
            stypy_interface.invoke(self.loc, append_method, type)

        tuple_ = invoke(self.loc, tuple_, list_)
        type_store.set_type_of(self.loc, name, tuple_)

    # ################################################## TESTING FUNCTIONS #################################################

    def setUp(self):
        # Create a type store
        self.type_store = Context(None, __file__)
        self.loc = Localization(__file__)

        # Create a list
        # l = [1, 2, "a"]
        self.__create_tuple(self.type_store, "sample_tuple1", [int(), str()])
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()

    def test_tuple_creation(self):
        tuple_ = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        getitem = self.type_store.get_type_of_member(self.loc, tuple_, "__getitem__")
        items = invoke(self.loc, getitem, stypy_interface.get_builtin_python_type_instance(None, 'int'))

        compare_types(tuple_, tuple)
        compare_types(items, [int(), str()])

    def test_tuple_get_attribute(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__doc__")

        compare_types(type(__temp_2), str)

    def test_tuple_get_method(self):
        temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        temp_2 = self.type_store.get_type_of_member(Localization(__file__), temp_1, "count")

        assert_equal_type_name(type(temp_2), "builtin_function_or_method")

    def test_tuple_get_invalid_member(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "invalid_member_name")

        assert_if_not_error(__temp_2)

    def test_tuple___getitem__(self):
        generic_1parameter_test(self.type_store, "sample_tuple1", "__getitem__", [int(), long()],
                                [[int(), str()], [int(), str()]],
                                [list, tuple], 1)

    def test_create_and_getitem(self):
        self.__create_tuple(self.type_store, "sample_tuple2", [float, bool, str])
        self.__obtain_tuple_item("sample_tuple2", "x")
        self.assertNotEqual(self.type_store.get_type_of(Localization(__file__), "x"), [int, str])
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [float, bool, str])

    def test_tuple_add_operator(self):
        # l3 = l + l2
        self.__create_tuple(self.type_store, "sample_tuple2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_tuple2")

        self.__obtain_tuple_item("sample_tuple1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_tuple_item("sample_tuple2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "+", __temp_1, __temp_2)
        self.type_store.set_type_of(Localization(__file__), "sample_tuple3", __temp_3)

        self.__obtain_tuple_item("sample_tuple1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_tuple_item("sample_tuple2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        self.__obtain_tuple_item("sample_tuple3", "elementsl3")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl3"),
                      [int(), float(), False, str()])

    def test_tuple_invalid_tuple_operator(self):
        self.__create_tuple(self.type_store, "sample_tuple2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_tuple2")

        self.__obtain_tuple_item("sample_tuple1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_tuple_item("sample_tuple2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "invalid_operator", __temp_1,
                                                   __temp_2)
        assert_if_not_error(__temp_3)

    def test_tuple_non_applicable_tuple_operator(self):
        self.__create_tuple(self.type_store, "sample_tuple2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_tuple2")

        self.__obtain_tuple_item("sample_tuple1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_tuple_item("sample_tuple2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "*", __temp_1,
                                                   __temp_2)
        assert_if_not_error(__temp_3)

    def test_tuple___getslice__(self):
        generic_2parameter_test(self.type_store, "sample_tuple1", "__getslice__",
                                [
                                    [int(), int()],
                                    [long(), False]
                                ],
                                [tuple, tuple],
                                [
                                    [list, float],
                                    [dict, tuple]
                                ], 1)

    def test_tuple___iter__(self):
        # x = l.__iter__()
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "__iter__")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "tupleiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), str()])

    def test_tuple__getattribute__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__getattribute__")

        __temp_69 = invoke(Localization(__file__), __temp_68,
                           stypy_interface.get_builtin_python_type_instance(None, 'str'))

        self.assertTrue(len(dir(tuple)) == len(__temp_69.types))

        self.__obtain_tuple_item("sample_tuple1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

    def test_tuple__setattr__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_tuple1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__setattr__")
        __temp_69 = invoke(Localization(__file__), __temp_68,
                           stypy_interface.get_builtin_python_type_instance(None, 'str', 'foo'),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'))

        assert_if_not_error(__temp_69)

    def test_tuple__getnewargs__(self):
        self.__create_tuple(self.type_store, "sample_tuple2", [float, bool, str, int, int])

        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_tuple2")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getnewargs__")

        __temp_3 = invoke(Localization(__file__), __temp_2)
        compare_types(__temp_3, tuple)
        compare_types(get_contained_elements_type(__temp_3), tuple)
        compare_types(get_contained_elements_type(get_contained_elements_type(__temp_3)),
                      get_contained_elements_type(__temp_1))
