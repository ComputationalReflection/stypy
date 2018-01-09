# ----------------------------------
# Python struct_time members type rules
# ----------------------------------


import time

from stypy.python_lib.type_rules.type_groups.type_group_generator import *

type_rules_of_members = {
    '__getslice__': [
        ((Integer, Integer), tuple),
        ((Integer, Overloads__trunc__), tuple),
        ((Overloads__trunc__, Integer), tuple),
        ((Overloads__trunc__, Overloads__trunc__), tuple),
    ],
    '__reduce__': [
        ((), tuple),
    ],
    '__rmul__': [
        ((Integer,), tuple),
    ],
    '__lt__': [
        ((AnyType,), bool),
    ],
    '__sizeof__': [
        ((), int),
    ],
    'struct_time': [
        ((time.struct_time,), types.NoneType),
        ((time.struct_time, time.struct_time), types.NoneType),
        ((time.struct_time, time.struct_time, time.struct_time), types.NoneType),
        ((time.struct_time, time.struct_time, time.struct_time, time.struct_time), types.NoneType),
    ],
    '__new__': [
        ((SubtypeOf(time.struct_time),), time.struct_time),
    ],
    '__contains__': [
        ((AnyType,), bool),
    ],
    '__len__': [
        ((), int),
    ],
    '__mul__': [
        ((Integer,), tuple),
    ],
    '__ne__': [
        ((AnyType,), bool),
    ],
    '__getitem__': [
        ((Integer,), DynamicType),
    ],
    '__subclasshook__': [
        ((), type),
    ],
    '__add__': [
        ((tuple,), tuple),
    ],
    '__gt__': [
        ((AnyType,), bool),
    ],
    '__eq__': [
        ((AnyType,), bool),
    ],
    '__le__': [
        ((AnyType,), bool),
    ],
    '__repr__': [
        ((), str),
    ],
    '__hash__': [
        ((), int),
    ],
    '__ge__': [
        ((AnyType,), bool),
    ],
}
