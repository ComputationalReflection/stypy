#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    '__str__': [
        ((), str),
    ],

    '__getattribute__': [
        ((Str,), DynamicType)
    ],

    '__init__': [
        ((), types.NoneType),
        ((AnyType,), types.NoneType),
        ((AnyType, AnyType), types.NoneType),
        ((AnyType, AnyType, AnyType), types.NoneType),
        #  (list, DynamicType, DynamicType, DynamicType, VarArgType): types.NoneType, TODO
    ],

    '__setattr__': [
        ((Str, AnyType), types.NoneType)
    ],

    '__repr__': [
        ((), str),
    ],

    # '__new__': [
    #     ((SubtypeOf(list)), list),
    #     # (type, VarArgType): first_param_is_a_subtype_of('list', list), TODO
    # ],

    '__format__': [
        ((Str,), str),
    ],

    '__subclasshook__': [
        ((), types.NoneType),
    ],

    '__reduce__': [
        ((Has__mro__, Integer), types.NoneType),
        ((Has__class__, Integer), types.NoneType),
        ((Has__mro__, Overloads__trunc__), types.NoneType),
        ((Has__class__, Overloads__trunc__), types.NoneType),
    ],

    '__reduce_ex__': [
        ((Has__mro__, Integer), types.NoneType),
        ((Has__class__, Integer), types.NoneType),
        ((Has__mro__, Overloads__trunc__), types.NoneType),
        ((Has__class__, Overloads__trunc__), types.NoneType),
    ],

    '__delattr__': [
        ((Str,), types.NoneType),
    ],

    '__sizeof__': [
        ((), int)
    ],
}
