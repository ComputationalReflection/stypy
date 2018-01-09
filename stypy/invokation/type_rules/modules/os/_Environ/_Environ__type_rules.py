# ----------------------------------
# Python _Environ members type rules
# ----------------------------------


import types
from stypy.invokation.type_rules.type_groups.type_group_generator import *
import os
from os import _Environ



type_rules_of_members = {
    '__getitem__': [
        ((Str,), Str),
    ],
    '__setitem__': [
        ((Str, Str), types.NoneType),
    ],
    'get': [
        ((Str, Str), str),
    ],
    '__contains__': [
        ((Str, ), bool),
    ]

    # '__hash__': [
    # ],
    # '__repr__': [
    # ],
    # '__init__': [
    #     ((os._Environ,), types.NoneType),
    #     ((os._Environ, os._Environ), types.NoneType),
    #     ((os._Environ, os._Environ, os._Environ), types.NoneType),
    #     ((os._Environ, os._Environ, os._Environ, os._Environ), types.NoneType),
    # ],
}
