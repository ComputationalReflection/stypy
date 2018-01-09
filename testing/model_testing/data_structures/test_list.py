#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from stypy.contexts.context import Context
from stypy.type_inference_programs import stypy_interface
from stypy.types.type_intercession import get_member
from testing.model_testing.model_testing_common import *
import types

class TestLists(unittest.TestCase):
    def __obtain_list_item(self, list_name, return_var_name):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), list_name)
        __temp_2 = get_contained_elements_type(__temp_1)
        self.type_store.set_type_of(Localization(__file__), return_var_name, __temp_2)

    def __create_list(self, type_store, name, types):
        list_ = stypy_interface.get_builtin_python_type_instance(self.loc, "list")
        append_method = get_member(self.loc, list_, "append")
        for type_ in types:
            stypy_interface.invoke(self.loc, append_method, type_)

        # StypyTypeError.print_error_msgs()
        type_store.set_type_of(self.loc, name, list_)

    # ################################################## TESTING FUNCTIONS #################################################

    def setUp(self):
        # Create a type store
        self.type_store = Context(None, __file__)
        self.loc = Localization(__file__)

        # Create a list
        # l = [1, 2, "a"]
        self.__create_list(self.type_store, "sample_list1", [int(), int(), str()])
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()

    def test_list_creation(self):
        list_ = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        getitem = get_member(self.loc, list_, "__getitem__")
        items = stypy_interface.invoke(self.loc, getitem, stypy_interface.get_builtin_python_type_instance(None, 'int'))

        compare_types(list_, list)
        compare_types(items, [int(), str()])

    def test_list_get_attribute(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "__doc__")

        compare_types(type(__temp_2), str)

    def test_list_get_method(self):
        temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        temp_2 = self.type_store.get_type_of_member(Localization(__file__), temp_1, "append")

        assert_equal_type_name(type(temp_2), "builtin_function_or_method")

    def test_list_get_invalid_member(self):
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of_member(Localization(__file__), __temp_1, "invalid_member_name")

        assert_if_not_error(__temp_2)

    def test_list___getitem__(self):
        generic_1parameter_test(self.type_store, "sample_list1", "__getitem__", [int(), long()],
                                [[int(), str()], [int(), str()]],
                                [list, tuple], 1)

    def test_list_append(self):
        generic_1parameter_test(self.type_store, "sample_list1", "append", [float(), long()],
                                [None, None], [], 0)
        self.__obtain_list_item("sample_list1", "x")

        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str(), float(), long()])

    def test_list_sort(self):
        generic_0parameter_test(self.type_store, "sample_list1", "sort", None)

        self.__obtain_list_item("sample_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

        class TestingClass:
            def __call__(self, localization):
                pass

        class AnotherClass:
            def __call__(self, localization):
                pass

        class Empty:
            pass

        testingClass = TestingClass()
        anotherClass = AnotherClass()
        emptyClass = Empty()

        generic_1parameter_test(self.type_store, "sample_list1", "sort",
                                [testingClass,
                                 anotherClass],
                                [None, None],
                                [Empty()], 1, 0)

        generic_2parameter_test(self.type_store, "sample_list1", "sort",
                                [
                                    [testingClass,
                                     testingClass],
                                    [anotherClass,
                                     anotherClass]
                                ],
                                [None, None],
                                [
                                    [emptyClass,
                                     emptyClass]
                                ], 1)

    def test_create_and_getitem(self):
        self.__create_list(self.type_store, "sample_list2", [float, bool, str])
        self.__obtain_list_item("sample_list2", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [float, bool, str])

    def test_list_add_operator(self):
        # l3 = l + l2
        self.__create_list(self.type_store, "sample_list2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_list2")

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "+", __temp_1, __temp_2)
        self.type_store.set_type_of(Localization(__file__), "sample_list3", __temp_3)

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        self.__obtain_list_item("sample_list3", "elementsl3")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl3"),
                      [int(), float(), False, str()])

    def test_list_invalid_list_operator(self):
        # l3 = l + l2
        self.__create_list(self.type_store, "sample_list2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_list2")

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "invalid_operator", __temp_1,
                                                   __temp_2)
        assert_if_not_error(__temp_3)

    def test_list_non_applicable_list_operator(self):
        # l3 = l + l2
        self.__create_list(self.type_store, "sample_list2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_list2")

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "*", __temp_1, __temp_2)

        assert_if_not_error(__temp_3)

    def test_list___getslice__(self):
        list_ = list()
        dict_ = dict()
        tuple_ = tuple()
        generic_2parameter_test(self.type_store, "sample_list1", "__getslice__",
                                [
                                    [int(), int()],
                                    [long(), True]
                                ],
                                [list, list],
                                [
                                    [list, float],
                                    [dict, tuple]
                                ], 1)

    def test_list_iadd_operator(self):
        # l += l2
        self.__create_list(self.type_store, "sample_list2", [float(), False, str()])
        __temp_1 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_2 = self.type_store.get_type_of(Localization(__file__), "sample_list2")

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"), [int(), str()])

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        __temp_3 = stypy_interface.python_operator(Localization(__file__), "+=", __temp_1, __temp_2)

        self.__obtain_list_item("sample_list2", "elementsl2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl2"), [float(), False, str()])

        self.__obtain_list_item("sample_list1", "elementsl1")
        compare_types(self.type_store.get_type_of(Localization(__file__), "elementsl1"),
                      [int(), float(), False, str()])

    def test_list___iter__(self):
        # x = l.__iter__()
        __temp_23 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_24 = self.type_store.get_type_of_member(Localization(__file__), __temp_23, "__iter__")
        __temp_25 = invoke(Localization(__file__), __temp_24)
        self.type_store.set_type_of(Localization(__file__), "x", __temp_25)

        assert_equal_type_name(self.type_store.get_type_of(Localization(__file__), "x"),
                               "listiterator")

        # item = x.next()
        __temp_38 = self.type_store.get_type_of(Localization(__file__), "x")
        __temp_39 = self.type_store.get_type_of_member(Localization(__file__), __temp_38, "next")
        __temp_40 = invoke(Localization(__file__), __temp_39)
        self.type_store.set_type_of(Localization(__file__), "item", __temp_40)

        compare_types(self.type_store.get_type_of(Localization(__file__), "item"), [int(), str()])

    def test_list_setitem(self):
        # Sample class to perform tests
        class C:
            pass

        C_ = C()
        self.__obtain_list_item("sample_list1", "after")
        compare_types(self.type_store.get_type_of(Localization(__file__), "after"), [int(), str()])

        generic_2parameter_test(self.type_store, "sample_list1", "__setitem__",
                                [
                                    [int(), C_],
                                    [long(), dict()]
                                ],
                                [None, None],
                                [
                                    [list, float],
                                    [dict, tuple]
                                ], 1, 1)

        self.__obtain_list_item("sample_list1", "before")

        compare_types(self.type_store.get_type_of(Localization(__file__), "after"),
                      [int(), str(), C_, dict(), dict, float, tuple, list])
        compare_types(self.type_store.get_type_of(Localization(__file__), "before"),
                      [int(), str(), C_, dict(), dict, float, tuple, list])

    def test_list_setslice(self):
        self.__create_list(self.type_store, "temp_list1", [int()])
        self.__create_list(self.type_store, "temp_list2", [False])
        self.__create_list(self.type_store, "temp_list3", [dict()])

        self.__obtain_list_item("temp_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), int())

        # Invalid call
        __temp_47 = self.type_store.get_type_of(Localization(__file__), "temp_list1")
        __temp_48 = self.type_store.get_type_of_member(Localization(__file__), __temp_47, "__setslice__")
        __temp_49 = invoke(Localization(__file__), __temp_48,
                           stypy_interface.get_builtin_python_type_instance(None, 'float'),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'),
                           stypy_interface.get_builtin_python_type_instance(None, 'str'))
        assert_if_not_error(__temp_49)

        # Set a type
        __temp_47 = self.type_store.get_type_of(Localization(__file__), "temp_list1")
        __temp_48 = self.type_store.get_type_of_member(Localization(__file__), __temp_47, "__setslice__")
        __temp_49 = invoke(Localization(__file__), __temp_48,
                           stypy_interface.get_builtin_python_type_instance(None, 'int'),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'),
                           stypy_interface.get_builtin_python_type_instance(None, 'str'))

        self.__obtain_list_item("temp_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

        # Set a list
        __temp_47 = self.type_store.get_type_of(Localization(__file__), "temp_list1")
        __temp_48 = self.type_store.get_type_of_member(Localization(__file__), __temp_47, "__setslice__")
        __temp_49 = self.type_store.get_type_of(Localization(__file__), "temp_list2")
        __temp_50 = invoke(Localization(__file__), __temp_48,
                           stypy_interface.get_builtin_python_type_instance(None, 'int'),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'), __temp_49)

        self.__obtain_list_item("temp_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str(), False])

        # Set a type or a list
        union = create_union_type([float(), self.type_store.get_type_of(Localization(__file__), "temp_list2")])
        __temp_47 = self.type_store.get_type_of(Localization(__file__), "temp_list3")
        __temp_48 = self.type_store.get_type_of_member(Localization(__file__), __temp_47, "__setslice__")
        __temp_50 = invoke(Localization(__file__), __temp_48,
                           stypy_interface.get_builtin_python_type_instance(None, 'int'),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'), union)

        self.__obtain_list_item("temp_list3", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [dict(), float(), False])

    def test_list_extend(self):
        # l.extend(l2)
        self.__create_list(self.type_store, "temp_list1", [int])
        self.__create_list(self.type_store, "temp_list2", [bool])
        self.__create_list(self.type_store, "temp_list3", [float])

        generic_1parameter_test(self.type_store, "temp_list1", "extend",
                                [self.type_store.get_type_of(Localization(__file__), "temp_list2"),
                                 self.type_store.get_type_of(Localization(__file__), "temp_list3")],
                                [None,
                                 None], [int, float], 1)

        self.__obtain_list_item("temp_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int, bool, float])

        self.__obtain_list_item("temp_list2", "x2")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x2"), bool)

    def test_list_insert(self):
        list_ = list()
        tuple_ = tuple()
        generic_2parameter_test(self.type_store, "sample_list1", "insert",
                                [
                                    [int(), list_],
                                    [long(), tuple_]
                                ],
                                [None,
                                 None],
                                [
                                    [str(), str()],
                                    [tuple_, float]
                                ], 1, 1)

        self.__obtain_list_item("sample_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"),
                      [int(), str(), float, list_, tuple_])

    def test_list__getattribute__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_list1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__getattribute__")

        __temp_69 = invoke(Localization(__file__), __temp_68,
                           stypy_interface.get_builtin_python_type_instance(None, 'str'))

        assertTrue(len(dir(list)) - 1 == len(__temp_69.types))

        # <type 'wrapper_descriptor'> \/ <type 'type'> \/ <type 'str'> \/ <type 'method_descriptor'> \/
        # <type 'NoneType'> \/ <type 'builtin_function_or_method'>

        self.__obtain_list_item("sample_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

    def test_list___delslice__(self):
        generic_2parameter_test(self.type_store, "sample_list1", "__delslice__",
                                [
                                    [int(), int()],
                                    [long(), long()]
                                ],
                                [None, None],
                                [
                                    [str, str],
                                    [tuple, list]
                                ], 1)

        self.__obtain_list_item("sample_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

    def test_list_delslice_conditional_rules(self):
        def __trunc__(self, localization):
            return int()

        class Cond:
            pass

        rightinstance = Cond()
        rightinstance.__trunc__ = types.MethodType(__trunc__, rightinstance)

        wronginstance = Cond()

        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_list1")
        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__delslice__")
        __temp_69 = invoke(Localization(__file__), __temp_68, rightinstance,
                           stypy_interface.get_builtin_python_type_instance(None, 'int'))

        compare_types(__temp_69, None)  # , "{0} == {1}".format(__temp_69, types.NoneType))

        # TypeWarning.print_warning_msgs()

        __temp_70 = invoke(Localization(__file__), __temp_68, wronginstance,
                           stypy_interface.get_builtin_python_type_instance(None, 'int'))

        assert_if_not_error(__temp_70)

        # TypeWarning.print_warning_msgs()

        self.__obtain_list_item("sample_list1", "x")
        compare_types(self.type_store.get_type_of(Localization(__file__), "x"), [int(), str()])

    def test_list__setattr__(self):
        __temp_67 = self.type_store.get_type_of(Localization(__file__), "sample_list1")

        __temp_68 = self.type_store.get_type_of_member(Localization(__file__), __temp_67, "__setattr__")
        __temp_69 = invoke(Localization(__file__), __temp_68,
                           stypy_interface.get_builtin_python_type_instance(None, 'str', "att"),
                           stypy_interface.get_builtin_python_type_instance(None, 'int'))

        assert_if_not_error(__temp_69)
