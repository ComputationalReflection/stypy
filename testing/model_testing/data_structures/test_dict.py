#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types
import unittest

from stypy.contexts.context import Context
from stypy.type_inference_programs import stypy_interface
from stypy.types.type_containers import set_contained_elements_type
from testing.model_testing.model_testing_common import *


class TestDicts(unittest.TestCase):
    def __create_dict(self, dict_name, list_keys, list_values):
        sample_dict = stypy_interface.get_builtin_python_type_instance(None, "dict")

        for i in range(len(list_keys)):
            add_key_and_value_type(sample_dict, list_keys[i], list_values[i])

        self.type_store.set_type_of(Localization(__file__), dict_name, sample_dict)

    def setUp(self):
        # Create a type store
        self.type_store = Context(None, __file__)

        # Create a list
        # l = [1, 2, "a"]
        self.__create_dict("sample_dict1", [int(), True], [str(), int()])

    def test_dict_creation(self):
        compare_types(type(self.type_store.get_type_of(Localization(__file__), "sample_dict1")),
                      type(stypy_interface.get_builtin_python_type_instance(None, "dict")))

    def test_dict_get_attribute(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__doc__")

        compare_types(type(__temp_2), str)

    def test_dict_get_method(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "get")

        assert_equal_type_name(type(__temp_2), "builtin_function_or_method")

    def test_dict_invalid_member(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "invalid_member_name")

        assert_if_not_error(__temp_2)

    def test_dict_getitem_parameters_error(self):
        # x = l[3, "a"] #error
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int(), str())

        assert_if_not_error(__temp_3)

    def test_dict_getitem_error(self):
        # x = l[3] #int or bool are valid keys
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, float(5))

        assert_if_not_error(__temp_3)

    def test_dict_getitem_ok(self):
        # x = l[3] #int or bool
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), str())

        __temp_3 = invoke(Localization(__file__), __temp_2, True)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), int())

    def test_dict_clear(self):
        # x = l[3] #int or str
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "clear")
        __temp_3 = invoke(Localization(__file__), __temp_2)

        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        assert_if_not_error(__temp_3)

    def __obtain_dict_item(self, dict_name, param_type, return_var_name):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), dict_name)
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, param_type)
        self.type_store.set_type_of(Localization(__file__), return_var_name, __temp_3)

    def test_dict_setitem_ok(self):
        # x = l[3] #int or bool
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__setitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, str(), False)
        __temp_3 = invoke(Localization(__file__), __temp_2, int(), float())

        self.__obtain_dict_item("sample_dict1", int(), "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [str(), float()])

        self.__obtain_dict_item("sample_dict1", str(), "x")
        compare_types(type(self.type_store.get_type_of(Localization(__file__), "x")), bool)

        self.__obtain_dict_item("sample_dict1", True, "x")
        compare_types(type(self.type_store.get_type_of(Localization(__file__), "x")), int)

    def test_dict_copy(self):
        # x = l[3] #int or str
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "copy")
        __temp_3 = invoke(Localization(__file__), __temp_2)
        self.type_store.set_type_of(Localization(__file__), "dict_copy", __temp_3)

        # Modify the original, check if the copy is independent
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__setitem__")
        __temp_4 = invoke(Localization(__file__), __temp_2, int(), float())

        self.__obtain_dict_item("dict_copy", int(), "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), str())

        self.__obtain_dict_item("sample_dict1", int(), "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [str(), float()])

    def test_dict_get(self):
        # x = l[3] #int or str
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "get")
        __temp_3 = invoke(Localization(__file__), __temp_2, str())  # type don't exist in key types
        assert __temp_3 == None

        __temp_3 = invoke(Localization(__file__), __temp_2, int())  # type exist in key types
        compare_types(__temp_3, str())

        __temp_3 = invoke(Localization(__file__), __temp_2, float(),
                          list())  # type don't exist in key types, return alternative

        compare_types(__temp_3, list())

    def test_dict_iterator(self):
        # x = l.__iter__()
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "__iter__")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "dictionary-keyiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), True])

    def test_dict_iterkeys(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "iterkeys")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "dictionary-keyiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), True])

    def test_dict_itervalues(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "itervalues")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "dictionary-valueiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), str()])

    def test_dict_iteritems(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "iteritems")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "dictionary-itemiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      tuple)

        compare_types(get_contained_elements_type(self.type_store.get_type_of(Localization(__file__), "item")),
                      [int(), True, str()])

    def test_dict_values(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "values")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"),
                      list)

        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "__getitem__")
        __temp_40 = invoke(Localization(__file__), __temp_39, int())

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), str()])

    def test_dict_keys(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "keys")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"),
                      list)

        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "__getitem__")
        __temp_40 = invoke(Localization(__file__), __temp_39, int())

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), True])

    def test_dict_items(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "items")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"),
                      list)

        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "__getitem__")
        __temp_40 = invoke(Localization(__file__), __temp_39, int())

        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      tuple)

        compare_types(get_contained_elements_type(self.type_store.get_type_of(Localization(__file__), "item")),
                      [int(), True, str()])

    #
    def test_dict_viewkeys(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "viewkeys")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(type(self.type_store.get_type_of(Localization(__file__), "x")),
                               "dict_keys")

    def test_dict_viewvalues(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "viewvalues")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(type(self.type_store.get_type_of(Localization(__file__), "x")),
                               "dict_values")

    def test_dict_viewitems(self):
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "viewitems")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(type(self.type_store.get_type_of(Localization(__file__), "x")),
                               "dict_items")

    def test_dict_popitem(self):
        # x = l.__iter__()
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "popitem")
        __temp_25 = invoke(Localization(__file__), __temp_24)

        self.type_store.set_type_of(Localization(__file__), "item", __temp_25)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      tuple)

        compare_types(get_contained_elements_type(self.type_store.get_type_of(Localization(__file__), "item")),
                      [int(), str(), True])

    def test_dict__getattribute__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__getattribute__")

        __temp_69 = invoke(Localization(__file__), __temp_68, str())

        assert_equal_type_name(type(__temp_69), "UnionType")
        self.assertEquals(len(__temp_69.types), 45)

        self.__obtain_dict_item("sample_dict1", int(), "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), str())

    def test_dict__setattr__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__setattr__")
        __temp_69 = invoke(Localization(__file__), __temp_68, "foo", int())

        assert_if_not_error(__temp_69)

    def test_dict_setitem(self):
        # Obtaing previous values
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), str())

        __temp_3 = invoke(Localization(__file__), __temp_2, True)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), int())

        # Set item (new entry, existing entry)
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__setitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int(), float())

        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__setitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, float(4), list)

        # Obtain new values
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [str(), float()])

        __temp_3 = invoke(Localization(__file__), __temp_2, True)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), int())

        __temp_3 = invoke(Localization(__file__), __temp_2, float(4))
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), list)

    def test_dict_update(self):
        # Obtaing previous values
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), str())

        __temp_3 = invoke(Localization(__file__), __temp_2, True)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), int())

        # Update item (new entry, existing entry)
        self.__create_dict("sample_dict2", [int(), True, float(4)], [list, tuple, dict])

        sample_dict = self.type_store.get_type_of(Localization(__file__), "sample_dict2")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "update")
        __temp_3 = invoke(Localization(__file__), __temp_2, sample_dict)

        # Obtain new values
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [str(), list])

        __temp_3 = invoke(Localization(__file__), __temp_2, True)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), tuple])

        __temp_3 = invoke(Localization(__file__), __temp_2, float(4))
        self.type_store.set_type_of(Localization(__file__), "x", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), dict)

    def test_dict_setdefault(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "setdefault")
        __temp_3 = invoke(Localization(__file__), __temp_2, str(),
                          False)  # type don't exist in key types

        compare_types(__temp_3, False)

        __temp_3 = invoke(Localization(__file__), __temp_2, int())  # type exist in key types
        compare_types(__temp_3, str())

        __temp_3 = invoke(Localization(__file__), __temp_2, int(),
                          list)  # type exist in key types, return alternative

        compare_types(__temp_3, [str(), list])

        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, str())
        compare_types(__temp_3, False)

        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__getitem__")
        __temp_3 = invoke(Localization(__file__), __temp_2, int())
        compare_types(__temp_3, [str(), list])

    def test_dict_fromkeys(self):
        # x = l[3] #int or str
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_dict1")

        # Test objects
        self.__create_dict("other_dict", [bool], [tuple])

        other_list = stypy_interface.get_builtin_python_type_instance(None, "list")
        union = UnionType.add(str, float)
        set_contained_elements_type(other_list, union)

        self.type_store.set_type_of(Localization(__file__), "other_list", other_list)

        other_object = int()
        self.type_store.set_type_of(Localization(__file__), "other_object", other_object)

        iter_call = self.type_store.get_type_of_member(Localization(__file__), other_list, "__iter__")
        iterator = invoke(Localization(__file__), iter_call)
        self.type_store.set_type_of(Localization(__file__), "iterator", iterator)

        # There are several cases:
        # A dictionary: Return a copy
        param = self.type_store.get_type_of(Localization(__file__), "other_dict")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, bool)
        compare_types(ret, tuple)

        # A dictionary and any other object: {<each dict key>: other object}
        param = self.type_store.get_type_of(Localization(__file__), "other_dict")
        param2 = self.type_store.get_type_of(Localization(__file__), "other_object")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param, param2)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, bool)
        compare_types(ret, [tuple, int()])

        # A list or a tuple: {each structure element type: None}
        param = self.type_store.get_type_of(Localization(__file__), "other_list")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param, types.NoneType)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, str)
        compare_types(ret, None)
        ret = invoke(Localization(__file__), getitem_call, float)
        compare_types(ret, None)

        # A list or a tuple and any other object: {<each dict key>: other object}
        param = self.type_store.get_type_of(Localization(__file__), "other_list")
        param2 = self.type_store.get_type_of(Localization(__file__), "other_object")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param, param2)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, str)
        compare_types(ret, self.type_store.get_type_of(Localization(__file__), "other_object"))
        ret = invoke(Localization(__file__), getitem_call, float)
        compare_types(ret, self.type_store.get_type_of(Localization(__file__), "other_object"))

        # Any __iter__ object: {each structure element type: None}
        param = self.type_store.get_type_of(Localization(__file__), "iterator")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param, types.NoneType)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, str)
        compare_types(ret, None)
        ret = invoke(Localization(__file__), getitem_call, float)
        compare_types(ret, None)

        # Any __iter__ object and any other object: {each structure element type: other object}
        param = self.type_store.get_type_of(Localization(__file__), "iterator")
        param2 = self.type_store.get_type_of(Localization(__file__), "other_object")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "fromkeys")
        __temp_3 = invoke(Localization(__file__), __temp_2, param, param2)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_3)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"),
                      dict)

        getitem_call = self.type_store.get_type_of_member(Localization(__file__), __temp_3, "__getitem__")
        ret = invoke(Localization(__file__), getitem_call, str)
        compare_types(ret, self.type_store.get_type_of(Localization(__file__), "other_object"))
        ret = invoke(Localization(__file__), getitem_call, float)
        compare_types(ret, self.type_store.get_type_of(Localization(__file__), "other_object"))
