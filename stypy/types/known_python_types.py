#!/usr/bin/env python
# -*- coding: utf-8 -*-
import operator
import os
import types

"""
File to store known Python language types (not defined in library modules) that are not listed in the types module.
Surprisingly, the types module omits several types that Python uses when dealing with commonly used structures, such
as iterators and member descriptors. As we need these types to specify type rules, type instances and, in general,
working with them, we created this file to list these types and its instances.
"""

# Native types that will use a simplified str representation (only its name) on stypy output
simple_python_types = [
    None,
    types.NoneType,
    type,
    bool,
    int,
    long,
    float,
    complex,
    str,
    unicode,
]

types_without_value = [
    bool,
    int,
    long,
    float,
    complex,
]

"""
Internal python elements to generate various sample type values
"""

class Foo:
    qux = 3

    def bar(self):
        pass


def func():
    pass


def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1


foo = Foo()

# ####################################

"""
This is the table that stypy uses to associate Python types with its textual name and a sample value.
This table is used for generating code automatically: type rules and data type wrappers that checks
members type.

Is is a <type object>: <tuple> dictionary.

The tuple contains a type name (to put into Python generated code) and a sample value of this type.
Type names are used to represent types into generated Python code. Note that not all Python type names can
be directly used into type declarations, as some contain the "-" character, not valid as part of a type
name in Python syntax. Sample values are only used when a call is done with the FakeParamValues call handler.
"""
known_python_type_typename_samplevalues = {
    types.NoneType: ("types.NoneType", None),
    type: ("type", type(int)),
    bool: ("bool", True),
    int: ("int", 1),
    long: ("long", 1L),
    float: ("float", 1.0),
    complex: ("complex", 1j),
    str: ("str", "foo"),
    unicode: ("unicode", u"foo"),
    tuple: ("tuple", (1, 2)),
    list: ("list", [1, 2, 3, 4, 5]),
    dict: ("dict", {"a": 1, "b": 2}),
    set: ("set", {4, 5}),
    type(func): ("types.FunctionType", func),
    types.LambdaType: ("types.LambdaType", lambda x: x),
    types.GeneratorType: ("types.GeneratorType", firstn(4)),
    types.CodeType: ("types.CodeType", func.func_code),
    types.BuiltinFunctionType: ("types.BuiltinFunctionType", len),
    types.ModuleType: ("types.ModuleType", types.ModuleType('foo')),
    file: ("file", file(os.path.dirname(os.path.realpath(__file__)) + "/foo.txt", "w")),
    xrange: ("xrange", xrange(1, 3)),
    types.SliceType: ("slice", slice(4, 3, 2)),
    buffer: ("buffer", buffer("r")),
    types.DictProxyType: ("types.DictProxyType", int.__dict__),
    types.ClassType: ("types.ClassType", Foo),
    types.InstanceType: ("types.InstanceType", foo),
    types.MethodType: ("types.MethodType", foo.bar),
    type(iter("abc")): ("iter", iter("abc")),

    bytearray: ("bytearray", bytearray("test")),
    classmethod: ("classmethod", classmethod(Foo.bar)),
    enumerate: ("enumerate", enumerate([1, 2, 3])),
    frozenset: ("frozenset", frozenset([1, 2, 3])),
    memoryview: ("memoryview", memoryview(buffer("foo"))),
    object: ("object", object()),
    property: ("property", property(Foo.qux)),
    staticmethod: ("staticmethod", staticmethod(Foo.bar)),
    super: ("super", super(type(Foo))),
    reversed: ("reversed", reversed((1, 2))),

    type(iter((1, 2, 3, 4, 5))): ("ExtraTypeDefinitions.tupleiterator", (1, 2, 3, 4, 5).__iter__()),
    type(iter(xrange(1))): ("ExtraTypeDefinitions.rangeiterator", iter(xrange(1))),
    type(iter([1, 2])): ("ExtraTypeDefinitions.listiterator", [1, 2, 3, 4, 5].__iter__()),
    type(iter(int, 1)): ("ExtraTypeDefinitions.callable_iterator", iter(int, 1)),
    type(iter(reversed([1, 2]))): ("ExtraTypeDefinitions.listreverseiterator", iter(reversed([1, 2]))),
    type(operator.methodcaller(0)): ("ExtraTypeDefinitions.methodcaller", operator.methodcaller(0)),
    type(operator.itemgetter(0)): ("ExtraTypeDefinitions.itemgetter", operator.itemgetter(0)),
    type(operator.attrgetter(0)): ("ExtraTypeDefinitions.attrgetter", operator.attrgetter(0)),
    type(iter(bytearray("test"))): ("ExtraTypeDefinitions.bytearray_iterator", iter(bytearray("test"))),

    type({"a": 1, "b": 2}.viewitems()): ("ExtraTypeDefinitions.dict_items", {"a": 1, "b": 2}.viewitems()),
    type({"a": 1, "b": 2}.viewkeys()): ("ExtraTypeDefinitions.dict_keys", {"a": 1, "b": 2}.viewkeys()),
    type({"a": 1, "b": 2}.viewvalues()): ("ExtraTypeDefinitions.dict_values", {"a": 1, "b": 2}.viewvalues()),

    type(iter({"a": 1, "b": 2})): ("ExtraTypeDefinitions.dictionary_keyiterator", iter({"a": 1, "b": 2})),
    type({"a": 1, "b": 2}.iteritems()): ("ExtraTypeDefinitions.dictionary_itemiterator", {"a": 1}.iteritems()),
    type({"a": 1, "b": 2}.itervalues()): ("ExtraTypeDefinitions.dictionary_valueiterator", {"a": 1}.itervalues()),

    type(iter(bytearray("test"))): ("ExtraTypeDefinitions.bytearray_iterator", iter(bytearray("test"))),

    type(ArithmeticError.message): ("ExtraTypeDefinitions.getset_descriptor", ArithmeticError.message),
    type(IOError.errno): ("ExtraTypeDefinitions.member_descriptor", IOError.errno),
    type(u"foo"._formatter_parser()): ("ExtraTypeDefinitions.formatteriterator", type(u"foo"._formatter_parser()))
}


class ExtraTypeDefinitions(object):
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


def get_sample_instance_for_type(type_name):
    """
    Gets an instance of the passed type
    :param type_name:
    :return:
    """
    try:
        type_ = getattr(ExtraTypeDefinitions, type_name)
        return known_python_type_typename_samplevalues[type_][1]
    except:
        try:
            return known_python_type_typename_samplevalues[getattr(types, type_name)][1]
        except:
            return known_python_type_typename_samplevalues[__builtins__[type_name]][1]


unique_object_id = 0
unique_instance_types = [tuple]


def needs_unique_instance(type_):
    """
    Tuples do not generate new instances as they are immutable
    :param type_:
    :return:
    """
    return type_ in unique_instance_types


def get_unique_instance(type_):
    """
    Gets a unique sample instance of a certain type
    :param type_:
    :return:
    """
    global unique_object_id
    ret = None

    if type_ is tuple:
        ret = tuple([unique_object_id])

    unique_object_id += 1
    return ret
