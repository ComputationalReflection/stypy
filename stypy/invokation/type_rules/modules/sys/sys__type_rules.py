#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from stypy.invokation.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    'getfilesystemencoding': [
        ((), str),
    ],
    'getprofile': [
        ((), types.NoneType),
    ],
    'exc_clear': [
        ((), types.NoneType),
    ],
    'getrefcount': [
        ((AnyType,), int),
    ],
    '_clear_type_cache': [
        ((), types.NoneType),
    ],
    'excepthook': [
        ((AnyType, AnyType, AnyType), types.NoneType),
    ],
    'getwindowsversion': [
        ((), sys.getwindowsversion),
    ],
    '__excepthook__': [
        ((AnyType, AnyType, AnyType), types.NoneType),
    ],
    'gettrace': [
        ((), types.MethodType),
    ],
    'getrecursionlimit': [
        ((), int),
    ],
    '_current_frames': [
        ((), dict),
    ],
    'call_tracing': [
        ((Has__call__, tuple), UndefinedType),
    ],
    'callstats': [
        ((), types.NoneType),
    ],
    'setcheckinterval': [
        ((Integer,), types.NoneType),
        ((Overloads__trunc__,), types.NoneType),
    ],
    'getdefaultencoding': [
        ((), str),
    ],
    'getcheckinterval': [
        ((), int),
    ],
    'settrace': [
        ((types.NoneType,), types.NoneType),
        ((Has__call__,), types.NoneType),
    ],
    'setprofile': [
        ((types.NoneType,), types.NoneType),
        ((Has__call__,), types.NoneType),
    ],
    'displayhook': [
        ((AnyType,), types.NoneType),
    ],
    'exitfunc': [
        ((), types.NoneType),
    ],
    'getsizeof': [
        ((AnyType,), int),
        ((AnyType, AnyType), int),
    ],
    '__displayhook__': [
        ((AnyType,), types.NoneType),
    ],
    '_getframe': [
        ((), DynamicType),
        ((Integer,), DynamicType),
        ((Overloads__trunc__,), DynamicType),
    ],
    'exc_info': [
        ((), tuple),
    ],
    'exit': [
        ((), UndefinedType),
        ((AnyType,), UndefinedType),
    ],
    'setrecursionlimit': [
        ((), types.NoneType),
        ((Integer,), types.NoneType),
        ((Overloads__trunc__,), types.NoneType),
    ],
}
