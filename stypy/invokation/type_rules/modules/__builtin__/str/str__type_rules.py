#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    'str': [
        ((), str),
        ((Number,), str),
        ((Str,), str),
        ((IterableObject,), str),
        ((ByteSequence,), str),
        ((Has__str__,), str),
    ],

    '__repr__': [
        ((), str),
    ],

    'islower': [
        ((), bool),
    ],

    'upper': [
        ((), str),
    ],

    '__getslice__': [
        ((Integer, Integer), str),
        ((Overloads__trunc__, Integer), str),
        ((Integer, Overloads__trunc__), str),
        ((Overloads__trunc__, Overloads__trunc__), str),
    ],

    'startswith': [
        ((Str,), bool),
        ((bytearray,), bool),
        ((Str, Integer), bool),
        ((bytearray, Integer), bool),
        ((Str, types.NoneType), bool),
        ((bytearray, types.NoneType), bool),
    ],

    'lstrip': [
        ((str,), str),
        ((types.NoneType,), str),
        ((unicode,), unicode),
    ],

    '__str__': [
        ((), str),
    ],

    'rpartition': [
        ((Str,), tuple),
        ((bytearray,), tuple),
    ],

    'replace': [
        ((Str, Str,), str),
        ((bytearray, Str,), str),
        ((Str, bytearray), str),
        ((bytearray, bytearray,), str),
    ],

    'isdigit': [
        ((), bool),
    ],

    'endswith': [
        ((Str,), bool),
        ((bytearray,), bool),
        ((Str, Integer), bool),
        ((bytearray, Integer), bool),
        ((Str, types.NoneType), bool),
        ((bytearray, types.NoneType), bool),
    ],

    'splitlines': [
        ((), list),
        ((Integer,), list),
        ((Overloads__trunc__,), list),
    ],

    'rfind': [
        ((Str,), bool),
        ((bytearray,), bool),
        ((Str, Integer), bool),
        ((bytearray, Integer), bool),
        ((Str, types.NoneType), bool),
        ((bytearray, types.NoneType), bool),
    ],

    'strip': [
        ((), str),
        ((Str,), str),
        ((types.NoneType,), str),
    ],

    '__rmul__': [
        ((Integer,), str),
    ],

    '__lt__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],

    '__getnewargs__': [
        ((), tuple),
    ],

    '__rmod__': [
        ((AnyType,), types.NotImplementedType),
    ],

    '__init__': [
        ((VarArgs,), types.NoneType),
    ],

    'index': [
        ((unicode,), int),
        ((str,), int),
        ((unicode, types.NoneType), int),
        ((str, types.NoneType), int),
        ((unicode, Integer), int),
        ((str, Integer), int),
        ((unicode, Integer, Integer), int),
        ((str, Integer, Integer), int),
        ((unicode, Integer, types.NoneType), int),
        ((str, Integer, types.NoneType), int),
    ],

    '__new__': [
        ((SubtypeOf(str),), str),
        ((SubtypeOf(str), VarArgs), str),
    ],

    'isalnum': [
        ((), bool),
    ],

    '__contains__': [
        ((unicode,), bool),
        ((str,), bool),
    ],

    'rindex': [
        ((unicode,), int),
        ((str,), int),
        ((unicode, types.NoneType), int),
        ((str, types.NoneType), int),
        ((unicode, Integer), int),
        ((str, Integer), int),
        ((unicode, Integer, Integer), int),
        ((str, Integer, Integer), int),
        ((unicode, Integer, types.NoneType), int),
        ((str, Integer, types.NoneType), int),
    ],

    'capitalize': [
        ((), str),
    ],

    'find': [
        ((Str,), int),
        ((bytearray,), int),
        ((Str, Integer), int),
        ((bytearray, Integer), int),
        ((Str, types.NoneType), int),
        ((bytearray, types.NoneType), int),
    ],

    'decode': [
        ((), unicode),
    ],

    'isalpha': [
        ((), bool),
    ],

    'split': [
        ((), list),
        ((Str,), list),
        ((types.NoneType,), list),
        ((bytearray,), list),
        ((Str, Integer), list),
        ((types.NoneType, Integer), list),
        ((bytearray, Integer), list),
        ((Str, Overloads__trunc__), list),
        ((types.NoneType, Overloads__trunc__), list),
        ((bytearray, Overloads__trunc__), list),
    ],

    'rstrip': [
        ((str,), str),
        ((types.NoneType,), str),
        ((unicode,), unicode),
    ],

    'encode': [
        ((str,), str),
    ],

    '_formatter_parser': [
        ((str,), ExtraTypeDefinitions.formatteriterator),
    ],

    'translate': [
        ((str,), str),
        ((str, str), str),
    ],

    'isspace': [
        ((), bool),
    ],

    '__len__': [
        ((), int),
    ],

    '__ne__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],

    '__getitem__': [
        ((Integer,), str),
        ((slice,), str),
    ],

    'format': [
        ((str,), str),
        ((VarArgs,), str),
    ],

    'ljust': [
        ((Integer,), str),
        ((Overloads__trunc__,), str),
        ((Integer, str), str),
        ((Overloads__trunc__, str), str),
    ],

    'rjust': [
        ((Integer,), str),
        ((Overloads__trunc__,), str),
        ((Integer, str), str),
        ((Overloads__trunc__, str), str),
    ],

    'swapcase': [
        ((), str),
    ],

    '__subclasshook__': [
        ((), type),
        ((AnyType,), type),
    ],

    'zfill': [
        ((Integer,), str),
        ((Overloads__trunc__,), str),
    ],

    '__add__': [
        ((bytearray,), bytearray),
        ((unicode,), unicode),
        ((str,), str),
    ],

    '__gt__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],

    '__eq__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],

    '__sizeof__': [
        ((), int),
    ],

    'count': [
        ((Str,), int),
        ((bytearray,), int),
        ((Str, Integer), int),
        ((bytearray, Integer), int),
        ((Str, types.NoneType), int),
        ((bytearray, types.NoneType), int),
    ],

    'lower': [
        ((), str),
    ],

    'join': [
        ((IterableDataStructureWithTypedElements(str, UndefinedType),), str),
    ],

    'center': [
        ((Integer,), str),
        ((Overloads__trunc__,), str),
        ((Integer, str), str),
        ((Overloads__trunc__, str), str),
    ],

    '__mod__': [
        ((types.DictProxyType,), str),
        ((types.InstanceType,), str),
        ((buffer,), str),
        ((dict,), str),
        ((bytearray,), str),
        ((list,), str),
        ((memoryview,), str),
    ],

    'partition': [
        ((Str,), tuple),
        ((bytearray,), tuple),
    ],

    'rsplit': [
        ((), list),
        ((Str,), list),
        ((types.NoneType,), list),
        ((bytearray,), list),
        ((Str, Integer), list),
        ((types.NoneType, Integer), list),
        ((bytearray, Integer), list),
        ((Str, Overloads__trunc__), list),
        ((types.NoneType, Overloads__trunc__), list),
        ((bytearray, Overloads__trunc__), list),
    ],

    'expandtabs': [
        ((), str),
        ((Integer,), str),
        ((Overloads__trunc__,), str),
    ],

    'istitle': [
        ((), bool),
    ],

    '__le__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],

    '__mul__': [
        ((Integer,), str),
    ],

    '_formatter_field_name_split': [
        ((), tuple),
    ],

    '__hash__': [
        ((), int),
    ],

    'title': [
        ((), str),
    ],

    'isupper': [
        ((), bool),
    ],

    '__ge__': [
        ((str,), bool),
        ((AnyType,), types.NotImplementedType),
    ],
}
