#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions

type_rules_of_members = {
    'iteritems': [
        ((), ExtraTypeDefinitions.dictionary_itemiterator),
    ],

    'pop': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), TypeOfParam(2)),
    ],

    'has_hey': [
        ((AnyType,), bool),
    ],

    'viewkeys': [
        ((), ExtraTypeDefinitions.dict_keys),
    ],

    '__lt__': [
        ((AnyType,), types.NotImplementedType)
    ],

    '__sizeof__': [
        (), int,
    ],

    'dict': [
        ((), types.NoneType),
        ((dict,), types.NoneType),
        ((types.DictProxyType,), types.NoneType),
        ((IterableDataStructure,), types.NoneType),

    ],

    'viewitems': [
        ((), ExtraTypeDefinitions.dict_items),
    ],

    '__setattr__': [
        ((Str, AnyType), types.NoneType)
    ],

    '__reduce__': [
        ((Integer,), tuple),
        ((Overloads__trunc__,), tuple),
        ((Has__mro__, Integer), DynamicType),
        ((Has__class__, Integer), DynamicType),
    ],

    '__reduce_ex__': [
        ((Integer,), tuple),
        ((Overloads__trunc__,), tuple),
        ((Has__mro__, Integer), DynamicType),
        ((Has__class__, Integer), DynamicType),
    ],

    '__new__': [
        ((SubtypeOf(dict),), dict),
        # (type, VarArgType): first_param_is_a_subtype_of('dict', dict),
    ],

    '__contains__': [
        ((AnyType,), bool),
    ],

    '__cmp__': [
        ((dict,), int),
    ],

    'itervalues': [
        ((), DynamicType),
    ],

    '__len__': [
        ((), int),
    ],

    '__ne__': [
        ((AnyType,), types.NotImplementedType),
    ],

    '__getitem__': [
        ((AnyType,), DynamicType),
    ],

    'get': [
        ((AnyType,), DynamicType),
        ((AnyType, AnyType), TypeOfParam(2)),
    ],

    'keys': [
        ((), list),
    ],

    'update': [
        ((), types.NoneType),
        ((IterableDataStructure,), types.NoneType),
    ],

    '__setitem__': [
        ((AnyType, AnyType), types.NoneType),
    ],

    '__gt__': [
        ((AnyType,), types.NotImplementedType),
    ],

    'popitem': [
        ((), tuple),
    ],

    'copy': [
        ((), dict),
    ],

    '__eq__': [
        ((AnyType,), types.NotImplementedType),
    ],

    'iterkeys': [
        ((), ExtraTypeDefinitions.dictionary_keyiterator),
    ],

    '__delitem__': [
        ((AnyType,), types.NoneType),
    ],

    'setdefault': [
        ((AnyType,), types.NoneType),
        ((AnyType, AnyType), types.NoneType),
    ],

    'viewvalues': [
        ((), ExtraTypeDefinitions.dict_values),
    ],

    'items': [
        ((), list),
    ],

    'clear': [
        ((), types.NoneType),
    ],

    '__iter__': [
        ((), ExtraTypeDefinitions.dictionary_keyiterator),
    ],

    '__le__': [
        ((AnyType,), types.NotImplementedType),
    ],

    'values': [
        ((), list),
    ],

    '__ge__': [
        ((AnyType,), types.NotImplementedType),
    ],
}
