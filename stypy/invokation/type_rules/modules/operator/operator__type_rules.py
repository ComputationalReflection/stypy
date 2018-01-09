#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *


def same_rules_as(key):
    return type_rules_of_members[key]


type_rules_of_members = {
    '_compare_digest': [
        ((ByteSequence, ByteSequence), bool),
        ((unicode, unicode), bool),
    ],

    '__abs__': [
        ((bool,), int),
        ((complex,), float),
        ((Number,), TypeOfParam(1)),
        ((Overloads__abs__,), DynamicType),
    ],

    '__add__': [
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((str, bytearray), bytearray),
        ((str, Str), TypeOfParam(2)),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, float), float),
        ((complex, Number), complex),
        ((buffer, buffer), str),
        ((buffer, bytearray), str),
        ((buffer, Str), str),
        ((bytearray, ByteSequence), bytearray),
        ((unicode, buffer), unicode),
        ((unicode, Str), unicode),
        ((tuple, tuple), DynamicType),
        ((numpy.ndarray, Number), TypeOfParam(1)),
        ((Number, numpy.ndarray), TypeOfParam(2)),
        ((Overloads__add__, AnyType), DynamicType),
        ((Overloads__iadd__, AnyType), DynamicType),
        ((AnyType, Overloads__radd__), DynamicType),
    ],

    'and_keyword': [
        ((bool, Integer), TypeOfParam(2)),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((types.NoneType, AnyType), types.NoneType),
        ((AnyType, types.NoneType), TypeOfParam(1)),
        ((AnyType, bool), bool),
        ((bool, AnyType), bool),
        ((AnyType, AnyType), TypeOfParam(2)),
    ],

    '__and__': [
        ((bool, Integer), TypeOfParam(2)),
        ((numpy.ndarray, AnyType), DynamicType),
        ((AnyType, numpy.ndarray), DynamicType),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__and__, AnyType), DynamicType),
        ((AnyType, Overloads__rand__), DynamicType)
    ],

    '__concat__': [
        ((buffer, buffer), str),
        ((buffer, bytearray), str),
        ((buffer, Str), str),
        ((bytearray, ByteSequence), bytearray),
        ((unicode, buffer), unicode),
        ((unicode, Str), unicode),
        ((str, bytearray), bytearray),
        ((str, Str), TypeOfParam(2)),
        ((list, list), list),
        ((tuple, tuple), tuple),
    ],

    '__contains__': [
        ((Str, Str), bool),
        ((IterableObject, AnyType), bool),
        ((Has__contains__, AnyType), DynamicType),
    ],

    '__delitem__': [
        ((dict, AnyType), types.NoneType),
        ((bytearray, Integer), types.NoneType),
        ((bytearray, slice), types.NoneType),
        ((list, Integer), types.NoneType),
        ((list, slice), types.NoneType),
        ((Has__delitem__, AnyType), types.NoneType),
    ],

    '__delslice__': [
        ((IterableDataStructure, Integer, Integer), types.NoneType),
        ((IterableDataStructure, Overloads__trunc__, Integer), types.NoneType),
        ((IterableDataStructure, Integer, Overloads__trunc__), types.NoneType),
        ((IterableDataStructure, Overloads__trunc__, Overloads__trunc__), types.NoneType),
        ((Has__delslice__, AnyType, AnyType), DynamicType)
    ],

    '__div__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, float), float),
        ((int, bool), int),
        ((int, Integer), int),
        ((int, RealNumber), float),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((Overloads__div__, AnyType), DynamicType),
        ((Overloads__idiv__, AnyType), DynamicType),
        ((AnyType, Overloads__rdiv__), DynamicType),
    ],

    '__eq__': [
        ((Number, Number), bool),
        ((Str, Number), bool),
        ((numpy.ndarray, numpy.ndarray), DynamicType),
        ((IterableObject, IterableObject), bool),
        ((Overloads__eq__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__eq__", "__cmp__"], 1), AnyType), bool)
    ],

    '__floordiv__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, complex), complex),
        ((long, Integer), long),
        ((long, float), float),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, complex), complex),
        ((float, Number), float),
        ((Overloads__floordiv__, AnyType), DynamicType),
        ((Overloads__ifloordiv__, AnyType), DynamicType),
        ((AnyType, Overloads__rfloordiv__), DynamicType)
    ],

    '__ge__': [
        ((complex, complex), StypyTypeError(None, "Complex numbers do not accept sorting", False)),
        ((Number, Number), bool),
        ((IterableDataStructure, IterableDataStructure), bool),
        ((Overloads__ge__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__ge__", "__cmp__"], 1), AnyType), bool)
    ],

    '__getslice__': [
        ((IterableDataStructure, Integer, Integer), DynamicType),
        ((IterableDataStructure, Overloads__trunc__, Integer), DynamicType),
        ((IterableDataStructure, Integer, Overloads__trunc__), DynamicType),
        ((IterableDataStructure, Overloads__trunc__, Overloads__trunc__), DynamicType),
        ((Str, Integer, Integer), DynamicType),
        ((Str, Overloads__trunc__, Integer), DynamicType),
        ((Str, Integer, Overloads__trunc__), DynamicType),
        ((Str, Overloads__trunc__, Overloads__trunc__), DynamicType),
        ((Has__getslice__, AnyType, AnyType), DynamicType)
    ],

    '__gt__': [
        ((complex, complex), StypyTypeError(None, "Complex numbers do not accept sorting", False)),
        ((Number, Number), bool),
        ((IterableDataStructure, IterableDataStructure), bool),
        ((Overloads__gt__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__gt__", "__cmp__"], 1), AnyType), bool)
    ],

    '__getitem__': [
        ((dict, AnyType), DynamicType),
        ((bytearray, Integer), int),
        ((bytearray, slice), bytearray),
        ((bytearray, CastsToIndex), int),
        ((list, Integer), DynamicType),
        ((list, CastsToIndex), DynamicType),
        ((list, slice), list),
        ((Has__getitem__, Integer), DynamicType),
        ((Has__getitem__, slice), DynamicType),
        ((Has__getitem__, CastsToIndex), DynamicType),
    ],

    '__iadd__': [
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((str, bytearray), bytearray),
        ((str, Str), TypeOfParam(2)),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, float), float),
        ((complex, Number), complex),
        ((buffer, buffer), str),
        ((buffer, bytearray), str),
        ((buffer, Str), str),
        ((bytearray, ByteSequence), bytearray),
        ((unicode, buffer), unicode),
        ((unicode, Str), unicode),
        ((Overloads__iadd__, AnyType), DynamicType),
        ((Number, Number), TypeOfParam(1)),
    ],

    '__iand__': [
        ((bool, Integer), TypeOfParam(2)),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__iand__, AnyType), DynamicType)
    ],

    '__iconcat__': lambda: same_rules_as('__concat__'),

    '__idiv__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, float), float),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((Overloads__idiv__, AnyType), DynamicType),
    ],

    '__ifloordiv__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, complex), complex),
        ((long, Integer), long),
        ((long, float), float),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, complex), complex),
        ((float, Number), float),
        ((Overloads__ifloordiv__, AnyType), DynamicType),
    ],

    '__ilshift__': [
        ((bool, bool), int),
        ((bool, long), long),
        ((bool, int), int),
        ((long, Integer), long),
        ((int, bool), int),
        ((int, long), long),
        ((int, int), int),
        ((Overloads__ilshift__, AnyType), DynamicType),
        ((Integer, numpy.ndarray), TypeOfParam(2))
    ],

    '__imod__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((Str, types.DictProxyType), TypeOfParam(1)),
        ((Str, types.InstanceType), TypeOfParam(1)),
        ((Str, buffer), TypeOfParam(1)),
        ((Str, dict), TypeOfParam(1)),
        ((Str, bytearray), TypeOfParam(1)),
        ((Str, list), TypeOfParam(1)),
        ((Str, memoryview), TypeOfParam(1)),
        ((long, bool), long),
        ((long, Number), TypeOfParam(2)),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, bool), float),
        ((float, complex), complex),
        ((float, RealNumber), TypeOfParam(2)),
        ((Overloads__imod__, AnyType), DynamicType),
    ],

    '__imul__': [
        ((bool, bool), int),
        ((bool, complex), complex),
        ((bool, buffer), str),
        ((bool, bytearray), bytearray),
        ((bool, Str), TypeOfParam(2)),
        ((bool, RealNumber), TypeOfParam(2)),
        ((bool, list), TypeOfParam(2)),
        ((bool, tuple), TypeOfParam(2)),
        ((complex, Number), complex),
        ((buffer, Integer), str),
        ((bytearray, Integer), bytearray),
        ((unicode, Integer), unicode),
        ((str, Integer), str),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, buffer), str),
        ((long, bytearray), bytearray),
        ((long, Str), TypeOfParam(2)),
        ((long, list), TypeOfParam(2)),
        ((long, float), float),
        ((long, tuple), TypeOfParam(2)),
        ((list, Integer), TypeOfParam(1)),
        ((int, bool), int),
        ((int, complex), complex),
        ((int, buffer), str),
        ((int, bytearray), bytearray),
        ((int, Str), TypeOfParam(2)),
        ((int, RealNumber), TypeOfParam(2)),
        ((int, list), TypeOfParam(2)),
        ((int, tuple), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((tuple, Integer), tuple),
        ((Overloads__imul__, AnyType), DynamicType),
    ],

    '__index__': [
        ((Integer,), TypeOfParam(1)),
        ((CastsToIndex,), DynamicType)
    ],

    '__inv__': [
        ((bool,), int),
        ((Integer,), TypeOfParam(1)),
        ((Overloads__invert__,), DynamicType),
    ],

    '__invert__': lambda: same_rules_as('__inv__'),

    '__ior__': [
        ((bool, Integer), TypeOfParam(2)),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__ior__, AnyType), DynamicType),
    ],

    '__ipow__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, Integer), TypeOfParam(1)),
        ((long, Number), TypeOfParam(2)),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((Overloads__ipow__, AnyType), DynamicType),
    ],

    '__irepeat__': lambda: same_rules_as('__repeat__'),

    '__irshift__': [
        ((bool, bool), int),
        ((bool, long), long),
        ((bool, int), int),
        ((long, Integer), long),
        ((int, bool), int),
        ((int, long), long),
        ((int, int), int),
        ((Overloads__irshift__, AnyType), DynamicType),
        ((numpy.ndarray, Integer), TypeOfParam(1))
    ],

    '__isub__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, RealNumber), long),
        ((long, complex), complex),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((IterableObject, IterableObject), set),
        ((Overloads__isub__, AnyType), DynamicType),
    ],

    '__itruediv__': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, complex), complex),
        ((complex, Number), complex),
        ((Overloads__itruediv__, AnyType), DynamicType),
    ],

    '__ixor__': [
        ((bool, Integer), TypeOfParam(2)),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__ixor__, AnyType), DynamicType),
    ],

    '__le__': [
        ((complex, complex), StypyTypeError(None, "Complex numbers do not accept sorting", False)),
        ((Number, Number), bool),
        ((IterableDataStructure, IterableDataStructure), bool),
        ((Overloads__le__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__le__", "__cmp__"], 1), AnyType), bool)
    ],

    '__lshift__': [
        ((bool, bool), int),
        ((bool, long), long),
        ((bool, int), int),
        ((long, Integer), long),
        ((int, bool), int),
        ((int, long), long),
        ((int, int), int),
        ((Overloads__lshift__, AnyType), DynamicType),
        ((Overloads__ilshift__, AnyType), DynamicType),
        ((AnyType, Overloads__rlshift__), DynamicType),
        ((Integer, numpy.ndarray), TypeOfParam(2))
    ],

    '__lt__': [
        ((complex, complex), StypyTypeError(None, "Complex numbers do not accept sorting", False)),
        ((Number, Number), bool),
        ((IterableDataStructure, IterableDataStructure), bool),
        ((Overloads__lt__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__lt__", "__cmp__"], 1), AnyType), bool)
    ],

    '__mod__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((Str, Number), TypeOfParam(1)),
        ((Str, types.DictProxyType), TypeOfParam(1)),
        ((Str, types.InstanceType), TypeOfParam(1)),
        ((Str, buffer), TypeOfParam(1)),
        ((Str, dict), TypeOfParam(1)),
        ((Str, bytearray), TypeOfParam(1)),
        ((Str, list), TypeOfParam(1)),
        ((Str, tuple), TypeOfParam(1)),
        ((Str, memoryview), TypeOfParam(1)),
        ((Str, TypeInstance), TypeOfParam(1)),
        ((long, bool), long),
        ((long, Number), TypeOfParam(2)),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, bool), float),
        ((float, complex), complex),
        ((float, RealNumber), float),
        ((Overloads__mod__, AnyType), DynamicType),
        ((Overloads__imod__, AnyType), DynamicType),
        ((AnyType, Overloads__rmod__), DynamicType)
    ],

    '__mul__': [
        ((bool, bool), int),
        ((bool, complex), complex),
        ((bool, buffer), str),
        ((bool, bytearray), bytearray),
        ((bool, Str), TypeOfParam(2)),
        ((bool, RealNumber), TypeOfParam(2)),
        ((bool, list), TypeOfParam(2)),
        ((bool, tuple), tuple),
        ((complex, Number), complex),
        ((buffer, Integer), str),
        ((bytearray, Integer), bytearray),
        ((unicode, Integer), unicode),
        ((str, Integer), str),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, buffer), str),
        ((long, bytearray), bytearray),
        ((long, Str), TypeOfParam(2)),
        ((long, list), TypeOfParam(2)),
        ((long, float), float),
        ((long, tuple), tuple),
        ((list, Integer), TypeOfParam(1)),
        ((int, bool), int),
        ((int, complex), complex),
        ((int, buffer), str),
        ((int, bytearray), bytearray),
        ((int, Str), TypeOfParam(2)),
        ((int, RealNumber), TypeOfParam(2)),
        ((int, list), TypeOfParam(2)),
        ((int, tuple), tuple),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((tuple, Integer), tuple),
        ((Overloads__mul__, AnyType), DynamicType),
        ((Overloads__imul__, AnyType), DynamicType),
        ((AnyType, Overloads__rmul__), DynamicType),
    ],

    '__ne__': [
        ((Number, Number), bool),
        ((IterableDataStructure, IterableDataStructure), bool),
        ((Overloads__ne__, AnyType), DynamicType),
        ((Overloads__cmp__, AnyType), DynamicType),
        ((DontHaveMember(["__ne__", "__cmp__"], 1), AnyType), bool)
    ],

    '__neg__': {
        ((bool,), int),
        ((Number,), TypeOfParam(1)),
        ((Overloads__neg__,), DynamicType),
    },

    '__not__': {
        ((AnyType,), bool),
    },

    'or_keyword': [
        ((bool, Integer), TypeOfParam(2)),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((AnyType, types.NoneType), TypeOfParam(1)),
        ((types.NoneType, AnyType), TypeOfParam(2)),
        ((AnyType, bool), TypeOfParam(1)),
        ((bool, AnyType), bool),
        ((AnyType, AnyType), TypeOfParam(1)),
    ],

    '__or__': [
        ((bool, Integer), TypeOfParam(2)),
        ((numpy.ndarray, AnyType), DynamicType),
        ((AnyType, numpy.ndarray), DynamicType),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__or__, AnyType), DynamicType),
        ((AnyType, Overloads__ror__), DynamicType)
    ],

    '__pos__': {
        ((Number,), TypeOfParam(1)),
        ((Overloads__pos__,), DynamicType),
    },

    '__pow__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, Integer), TypeOfParam(1)),
        ((long, Number), TypeOfParam(2)),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((numpy.ndarray, Integer), DynamicType),
        ((Integer, numpy.ndarray), DynamicType),
        ((Overloads__pow__, AnyType), DynamicType),
        ((Overloads__ipow__, AnyType), DynamicType),
        ((AnyType, Overloads__rpow__), DynamicType)
    ],

    '__repeat__': [
        ((buffer, Integer), str),
        ((bytearray, Integer), TypeOfParam(1)),
        ((bytearray, Overloads__trunc__), TypeOfParam(1)),
        ((unicode, Integer), TypeOfParam(1)),
        ((str, Integer), TypeOfParam(1)),
        ((list, Integer), TypeOfParam(1)),
        ((tuple, Integer), TypeOfParam(1)),
    ],

    '__rshift__': [
        ((bool, bool), int),
        ((bool, long), long),
        ((bool, int), int),
        ((long, Integer), long),
        ((int, bool), int),
        ((int, long), long),
        ((int, int), int),
        ((Overloads__rshift__, AnyType), DynamicType),
        ((Overloads__irshift__, AnyType), DynamicType),
        ((AnyType, Overloads__rrshift__), DynamicType),
        ((numpy.ndarray, Integer), TypeOfParam(1))
    ],

    '__setitem__': [
        ((dict, AnyType, AnyType), types.NoneType),
        ((bytearray, Integer, Integer), types.NoneType),
        ((bytearray, slice, IterableDataStructure), types.NoneType),
        ((bytearray, CastsToIndex, AnyType), types.NoneType),
        ((list, Integer, AnyType), types.NoneType),
        ((list, slice, IterableDataStructure), types.NoneType),
        ((list, CastsToIndex, AnyType), types.NoneType),
        ((Has__setitem__, Integer, AnyType), types.NoneType),
        ((Has__setitem__, slice, IterableDataStructure), types.NoneType),
        ((Has__setitem__, CastsToIndex, AnyType), types.NoneType),
    ],

    '__sub__': [
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, RealNumber), long),
        ((long, complex), complex),
        ((int, bool), int),
        ((int, Number), TypeOfParam(2)),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((numpy.ndarray, numpy.ndarray), TypeOfParam(1)),
        ((IterableObject, IterableObject), set),
        ((Overloads__sub__, AnyType), DynamicType),
        ((Overloads__isub__, AnyType), DynamicType),
        ((AnyType, Overloads__rsub__), DynamicType)
    ],

    '__truediv__': [
        ((RealNumber, RealNumber), float),
        ((RealNumber, complex), complex),
        ((complex, Number), complex),
        ((Overloads__truediv__, AnyType), DynamicType),
        ((AnyType, Overloads__rtruediv__), DynamicType)
    ],

    '__xor__': [
        ((bool, Integer), TypeOfParam(2)),
        ((numpy.ndarray, AnyType), DynamicType),
        ((AnyType, numpy.ndarray), DynamicType),
        ((IterableObject, ExtraTypeDefinitions.dict_items), set),
        ((IterableObject, ExtraTypeDefinitions.dict_keys), set),
        ((ExtraTypeDefinitions.dict_items, IterableObject), set),
        ((frozenset, frozenset), frozenset),
        ((frozenset, set), frozenset),
        ((long, Integer), long),
        ((ExtraTypeDefinitions.dict_keys, IterableObject), set),
        ((int, bool), int),
        ((int, Integer), TypeOfParam(2)),
        ((set, frozenset), set),
        ((set, set), set),
        ((Overloads__xor__, AnyType), DynamicType),
        ((Overloads__ixor__, AnyType), DynamicType),
        ((AnyType, Overloads__rxor__), DynamicType)
    ],

    # NON - PRIVATE OPERATORS

    'add': lambda: same_rules_as('__add__'),

    'and_': lambda: same_rules_as('__and__'),

    'concat': {
        ((buffer, Str), str),
        ((buffer, bytearray), str),
        ((bytearray, ByteSequence), bytearray),
        ((unicode, Str), unicode),
        ((str, bytearray), bytearray),
        ((str, unicode), unicode),
        ((str, str), str),
        ((list, list), list),
        ((tuple, tuple), tuple),
    },

    'contains': lambda: same_rules_as('__contains__'),

    'countOf': [
        ((Str,), int),
        ((IterableObject,), int)
    ],

    'delitem': lambda: same_rules_as('__delitem__'),

    'delslice': lambda: same_rules_as('__delslice__'),

    'div': lambda: same_rules_as('__div__'),

    'eq': lambda: same_rules_as('__eq__'),

    'floordiv': lambda: same_rules_as('__floordiv__'),

    'ge': lambda: same_rules_as('__ge__'),

    'getitem': lambda: same_rules_as('__getitem__'),

    'getslice': lambda: same_rules_as('__getslice__'),

    'gt': lambda: same_rules_as('__gt__'),

    'iadd': lambda: same_rules_as('__iadd__'),

    'iand': lambda: same_rules_as('__iand__'),

    'iconcat': lambda: same_rules_as('__iconcat__'),

    'idiv': lambda: same_rules_as('__idiv__'),

    'ifloordiv': lambda: same_rules_as('__ifloordiv__'),

    'ilshift': lambda: same_rules_as('__ilshift__'),

    'imod': lambda: same_rules_as('__imod__'),

    'imul': lambda: same_rules_as('__imul__'),

    'index': lambda: same_rules_as('__index__'),

    'indexOf': [
        ((IterableObject, AnyType), int)
    ],

    'inv': lambda: same_rules_as('__inv__'),

    'invert': lambda: same_rules_as('__inv__'),

    'ior': lambda: same_rules_as('__ior__'),

    'ipow': lambda: same_rules_as('__ipow__'),

    'irepeat': lambda: same_rules_as('__repeat__'),

    'irshift': lambda: same_rules_as('__irshift__'),

    'is_': [
        ((AnyType, AnyType), bool)
    ],

    'is_not': [
        ((AnyType, AnyType), bool)
    ],

    'isCallable': [
        ((AnyType,), bool),
    ],

    'isSequenceType': [
        ((AnyType,), bool)
    ],

    'isMappingType': [
        ((AnyType,), bool)
    ],

    'isNumberType': [
        ((AnyType,), bool)
    ],

    'isub': lambda: same_rules_as('__isub__'),

    'itruediv': lambda: same_rules_as('__itruediv__'),

    'ixor': lambda: same_rules_as('__ixor__'),

    'le': lambda: same_rules_as('__le__'),

    'lshift': lambda: same_rules_as('__lshift__'),

    'lt': lambda: same_rules_as('__lt__'),

    'mod': lambda: same_rules_as('__mod__'),

    'mul': lambda: same_rules_as('__mul__'),

    'ne': lambda: same_rules_as('__ne__'),

    'neg': lambda: same_rules_as('__neg__'),

    'not_': lambda: same_rules_as('__not__'),

    'or_': lambda: same_rules_as('__or__'),

    'pos': lambda: same_rules_as('__pos__'),

    'pow': lambda: same_rules_as('__pow__'),

    'repeat': lambda: same_rules_as('__repeat__'),

    'rshift': lambda: same_rules_as('__rshift__'),

    'sequenceIncludes': [
        ((IterableObject, AnyType), bool)
    ],

    'setitem': lambda: same_rules_as('__setitem__'),

    'sub': lambda: same_rules_as('__sub__'),

    'truediv': lambda: same_rules_as('__truediv__'),

    'truth': [
        ((AnyType,), bool)
    ],

    'xor': lambda: same_rules_as('__xor__'),
}
