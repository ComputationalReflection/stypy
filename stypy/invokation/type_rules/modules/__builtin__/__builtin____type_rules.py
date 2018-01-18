#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.invokation.type_rules.type_groups.type_group_generator import *
from stypy.types.known_python_types import ExtraTypeDefinitions
from stypy.types.undefined_type import UndefinedType

type_rules_of_members = {
    'bytearray': [
        ((), bytearray),
        ((IterableDataStructureWithTypedElements(Integer, Overloads__trunc__),), bytearray),
        ((Integer,), bytearray),
        ((Str,), bytearray),
    ],

    'all': [
        ((IterableObject,), bool),
        ((Str,), bool),
    ],

    'set': [
        ((), set),
        ((IterableObject,), set),
        ((Str,), set),
    ],

    'vars': [
        ((), dict),
        ((Has__dict__,), dict),
    ],

    'bool': [
        ((), bool),
        ((AnyType,), bool),
    ],

    'float': [
        ((), float),
        ((Number,), float),
        ((Str,), float),
        ((CastsToFloat,), float),
    ],

    '__import__': [
        ((Str,), types.ModuleType),
        ((Str, AnyType), types.ModuleType),
        ((Str, AnyType, AnyType), types.ModuleType),
        ((Str, AnyType, AnyType, AnyType), types.ModuleType),
        ((Str, AnyType, AnyType, AnyType, Integer), types.ModuleType),
    ],

    'unicode': [
        ((), unicode),
        ((Has__str__,), unicode),
        ((AnyType,), DynamicType),
    ],

    'enumerate': [
        ((Str,), enumerate),
        ((Str, Integer), enumerate),
        ((IterableObject,), enumerate),
        ((Has__iter__,), enumerate),
        ((IterableObject, Integer), enumerate),
        ((Has__iter__, Integer), enumerate)
    ],

    'reduce': [
        ((Has__call__, Str,), DynamicType),
        ((Has__call__, IterableObject,), DynamicType),
        ((Has__call__, Str, AnyType,), DynamicType),
        ((Has__call__, IterableObject, AnyType,), DynamicType),
    ],

    'list': [
        ((), list),
        ((IterableObject,), list),
        ((Str,), list),
    ],

    'coerce': [
        ((Number, Number), tuple),
    ],

    'intern': [
        ((str,), str),
    ],

    'globals': [
        ((), dict),
    ],

    'issubclass': [
        ((Type, Type), bool),
    ],

    'divmod': [
        ((Number, Number), tuple),
        ((Overloads__divmod__, Number), tuple),
        ((Number, Overloads__rdivmod__), tuple),
    ],

    'file': [
        ((Str,), file),
        ((Str, Str), file),
        ((Str, Str, Integer), file),
        ((Str, Str, Overloads__trunc__), file)
    ],

    'unichr': [
        ((Integer,), unicode),
        ((Overloads__trunc__,), unicode),
    ],

    'apply': [
        ((Has__call__,), DynamicType),
        ((Has__call__, tuple), DynamicType),
        ((Has__call__, tuple, dict), DynamicType)
    ],

    'isinstance': [
        ((AnyType, AnyType), bool),
    ],

    'next': [
        ((Has__next,), DynamicType),
        ((Has__next, AnyType), DynamicType),
    ],

    'any': [
        ((IterableObject,), bool),
    ],

    'locals': [
        ((), dict),
    ],

    'filter': [
        ((Has__call__, IterableObject), DynamicType),
        ((Has__call__, Str), DynamicType),
    ],

    'slice': [
        ((AnyType,), slice),
        ((AnyType, AnyType), slice),
        ((AnyType, AnyType, AnyType), slice),
    ],

    'copyright': [
        ((), str)
    ],

    'min': [
        ((IterableObject,), DynamicType),
        ((Str,), DynamicType),
        ((IterableObject, Has__call__), DynamicType),
        ((Str, Has__call__), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, Has__call__), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, Has__call__), DynamicType),
        ((AnyType, AnyType, VarArgs,), DynamicType),
    ],

    'open': [
        ((Str,), file),
        ((Str, Str), file),
        ((Str, Str, Integer), file),
        ((Str, Str, Overloads__trunc__), file)
    ],

    'sum': [
        ((IterableDataStructureWithTypedElements(Number),), DynamicType),
        ((IterableDataStructureWithTypedElements(Number), Integer), DynamicType),
    ],

    'chr': [
        ((Integer,), str),
        ((Overloads__trunc__,), str),
    ],

    'hex': [
        ((Integer,), str),
        ((CastsToHex,), str),
    ],

    'exec': [
        ((types.CodeType,), DynamicType),
        ((types.CodeType, types.NoneType), DynamicType),
        ((types.CodeType, dict), DynamicType),
        ((types.CodeType, dict, types.NoneType), DynamicType),
        ((Str,), DynamicType),
        ((Str, dict), DynamicType),
        ((Str, dict, types.NoneType), DynamicType),
        ((Str, dict, dict), DynamicType),
        ((Str, types.NoneType, dict), DynamicType),
        ((file,), DynamicType),
        ((file, dict), DynamicType),
        ((file, dict, types.NoneType), DynamicType),
        ((file, dict, dict), DynamicType),
        ((file, types.NoneType, dict), DynamicType),
    ],

    'execfile': [
        ((Str,), DynamicType),
        ((Str, dict), DynamicType),
        ((Str, dict, dict), DynamicType),
    ],

    'long': [
        ((), long),
        ((RealNumber,), long),
        ((Overloads__trunc__,), long),
        ((CastsToLong,), long),
        ((Str,), long),
        ((Str, Integer), long),
        ((Str, Overloads__trunc__), long),
        ((RealNumber, Integer), long),
        ((RealNumber, Overloads__trunc__), long),
        ((Overloads__trunc__, Integer), long),
        ((Overloads__trunc__, Overloads__trunc__), long),
    ],

    'id': [
        ((AnyType,), long)
    ],

    'xrange': [
        ((Integer,), xrange),
        ((Overloads__trunc__,), xrange),
        ((Integer, Integer), xrange),
        ((Overloads__trunc__, Integer), xrange),
        ((Integer, Overloads__trunc__), xrange),
        ((Overloads__trunc__, Overloads__trunc__), xrange),
        ((Integer, Integer, Integer), xrange),
        ((Overloads__trunc__, Integer, Integer), xrange),
        ((Integer, Overloads__trunc__, Integer), xrange),
        ((Integer, Integer, Overloads__trunc__), xrange),
        ((Integer, Overloads__trunc__, Overloads__trunc__), xrange),
        ((Overloads__trunc__, Overloads__trunc__, Integer), xrange),
        ((Overloads__trunc__, Integer, Overloads__trunc__), xrange),
        ((Overloads__trunc__, Overloads__trunc__, Overloads__trunc__), xrange),
    ],

    'int': [
        ((), int),
        ((RealNumber,), int),
        ((Overloads__trunc__,), int),
        ((CastsToInt,), int),
        ((Str,), int),
        ((Str, Integer), int),
        ((Str, Overloads__trunc__), int),
        ((RealNumber, Integer), int),
        ((RealNumber, Overloads__trunc__), int),
        ((Overloads__trunc__, Integer), int),
        ((Overloads__trunc__, Overloads__trunc__), int),
    ],

    'getattr': [
        ((AnyType, Str), DynamicType),
        ((AnyType, Str, AnyType), DynamicType),
    ],

    'abs': [
        ((bool,), int),
        ((complex,), float),
        ((Number,), TypeOfParam(1)),
        ((Overloads__abs__,), DynamicType),
    ],

    'pow': [
        ((int, Number), TypeOfParam(2)),
        ((bool, bool), int),
        ((bool, Number), TypeOfParam(2)),
        ((complex, Number), complex),
        ((long, Integer), long),
        ((long, complex), complex),
        ((long, float), float),
        ((int, bool), int),
        ((float, RealNumber), float),
        ((float, complex), complex),
        ((bool, bool, bool), int),
        ((bool, bool, types.NoneType), int),
        ((bool, bool, Integer), TypeOfParam(2)),
        ((bool, complex, types.NoneType), complex),
        ((bool, long, bool), long),
        ((bool, long, types.NoneType), long),
        ((bool, long, Integer), TypeOfParam(1)),
        ((bool, int, bool), int),
        ((bool, int, types.NoneType), int),
        ((bool, int, Integer), TypeOfParam(2)),
        ((bool, float, types.NoneType), float),
        ((complex, Number, types.NoneType), complex),
        ((long, bool, Integer), long),
        ((long, bool, types.NoneType), long),
        ((long, complex, types.NoneType), complex),
        ((long, long, Integer), long),
        ((long, long, types.NoneType), long),
        ((long, int, Integer), long),
        ((long, int, types.NoneType), long),
        ((long, float, types.NoneType), float),
        ((int, bool, bool), int),
        ((int, bool, types.NoneType), int),
        ((int, bool, Integer), TypeOfParam(3)),
        ((int, complex, types.NoneType), complex),
        ((int, long, Integer), long),
        ((int, long, types.NoneType), long),
        ((int, int, bool), int),
        ((int, int, types.NoneType), int),
        ((int, int, Integer), TypeOfParam(3)),
        ((int, RealNumber, types.NoneType), float),
        ((float, complex, types.NoneType), complex),
        ((Overloads__pow__, AnyType), DynamicType),
        ((Overloads__pow__, AnyType, AnyType), DynamicType),
        ((Overloads__pow__, AnyType, AnyType, AnyType), DynamicType),
    ],

    'input': [
        ((), DynamicType),
        ((AnyType,), DynamicType),
    ],

    'type': [
        ((AnyType,), TypeObjectOfParam(1)),
        ((Str, tuple, dict), DynamicType)
    ],

    'oct': [
        ((Integer,), str),
        ((CastsToOct,), str),
    ],

    'bin': [
        ((Integer,), str),
    ],

    'map': [
        ((Has__call__, IterableObject), list),
        ((Has__call__, IterableObject, IterableObject), list),
        ((Has__call__, IterableObject, IterableObject, IterableObject), list),
        ((Has__call__, Str), list),
        ((Has__call__, Str, IterableObject), list),
        ((Has__call__, IterableObject, Str), list),
        ((Has__call__, Str, Str), list),
        ((Has__call__, Str, IterableObject, IterableObject), list),
        ((Has__call__, IterableObject, Str, IterableObject), list),
        ((Has__call__, IterableObject, IterableObject, Str), list),
        ((Has__call__, Str, Str, IterableObject), list),
        ((Has__call__, IterableObject, Str, Str), list),
        ((Has__call__, Str, IterableObject, Str), list),
        ((Has__call__, Str, Str, Str), list),
        ((Has__call__, IterableObject, VarArgs), list),
    ],

    'zip': [
        ((), list),
        ((IterableObject,), list),
        ((Str,), list),
        ((IterableObject, IterableObject), list),
        ((Str, IterableObject), list),
        ((IterableObject, Str), list),
        ((Str, Str), list),
        ((IterableObject, IterableObject, IterableObject), list),
        ((Str, IterableObject, IterableObject), list),
        ((IterableObject, Str, IterableObject), list),
        ((IterableObject, IterableObject, Str), list),
        ((Str, Str, IterableObject), list),
        ((IterableObject, Str, Str), list),
        ((Str, IterableObject, Str), list),
        ((Str, Str, Str), list),
        ((IterableObject, VarArgs), list),
        ((Str, VarArgs), list),
    ],

    'hash': [
        ((AnyType,), int),
    ],

    'format': [
        ((AnyType,), str),
        ((AnyType, Str), str),
    ],

    'max': [
        ((IterableObject,), DynamicType),
        ((Str,), DynamicType),
        ((IterableObject, Has__call__), DynamicType),
        ((Str, Has__call__), DynamicType),
        ((AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, Has__call__), DynamicType),
        ((AnyType, AnyType, AnyType), DynamicType),
        ((AnyType, AnyType, AnyType, Has__call__), DynamicType),
        ((AnyType, AnyType, VarArgs,), DynamicType),
    ],

    'reversed': [
        ((buffer,), reversed),
        ((bytearray,), reversed),
        ((Str,), reversed),
        ((list,), ExtraTypeDefinitions.listreverseiterator),
        ((tuple,), reversed),
        ((xrange,), ExtraTypeDefinitions.rangeiterator),
    ],

    'object': [
        ((), object),
    ],

    'quit': [
        ((), UndefinedType),
        ((AnyType,), UndefinedType),
    ],

    'len': [
        ((IterableObject,), int),
        ((Str,), int),
        ((Has__len__,), int),
    ],

    'repr': [
        ((Has__repr__,), str),
    ],

    'callable': [
        ((AnyType,), bool),
    ],

    'credits': [
        ((), types.NoneType),
    ],

    'tuple': [
        ((), tuple),
        ((Str,), tuple),
        ((IterableObject,), tuple)
    ],

    'eval': [
        ((types.CodeType,), DynamicType),
        ((types.CodeType, types.NoneType), DynamicType),
        ((types.CodeType, dict), DynamicType),
        ((types.CodeType, dict, types.NoneType), DynamicType),
        ((Str,), DynamicType),
        ((Str, dict), DynamicType),
        ((Str, dict, types.NoneType), DynamicType),
        ((Str, dict, dict), DynamicType),
        ((Str, types.NoneType, dict), DynamicType),
    ],

    'frozenset': [
        ((), frozenset),
        ((Str,), frozenset),
        ((IterableObject,), frozenset),
    ],

    'sorted': [
        ((IterableObject,), list),
        ((IterableObject, Has__call__), list),
        ((IterableObject, Has__call__, bool), list),
        ((IterableObject, Has__call__, Has__call__), list),
        ((IterableObject, Has__call__, Has__call__, bool), list),
        ((Str,), list),
        ((Str, Has__call__), list),
        ((Str, Has__call__, bool), list),
        ((Str, Has__call__, Has__call__), list),
        ((Str, Has__call__, Has__call__, bool), list),
    ],

    'ord': [
        ((Str,), int),
    ],

    'super': {
        ((Type,), super),
        ((Type, types.NoneType), super),
        ((Type, AnyType), super),
    },

    'hasattr': [
        ((AnyType, Str), bool)
    ],

    'delattr': [
        ((AnyType, Str), types.NoneType)
    ],

    'dict': [
        ((), dict),
        ((dict,), dict),
        ((IterableObject,), dict),
    ],

    'setattr': [
        ((AnyType, Str, AnyType), types.NoneType),
    ],

    'classmethod': [
        ((AnyType,), classmethod),
    ],

    'raw_input': [
        ((), str),
        ((AnyType,), str),
    ],

    'bytes': [
        ((), str),
        ((AnyType,), str)
    ],

    'iter': [
        ((Str,), DynamicType),
        ((IterableObject,), DynamicType),
        ((Type, AnyType), DynamicType),
        ((Has__call__, AnyType), DynamicType),
    ],

    'compile': [
        ((Str, Str, Str), types.CodeType),
        ((Str, Str, Str, Integer), types.CodeType),
        ((Str, Str, Str, Integer, Integer), types.CodeType),
    ],

    'reload': {
        ((types.ModuleType,), types.ModuleType),
    },

    'range': [
        ((Integer,), range),
        ((Overloads__trunc__,), range),
        ((Integer, Integer), range),
        ((Overloads__trunc__, Integer), range),
        ((Integer, Overloads__trunc__), range),
        ((Overloads__trunc__, Overloads__trunc__), range),
        ((Integer, Integer, Integer), range),
        ((Overloads__trunc__, Integer, Integer), range),
        ((Integer, Overloads__trunc__, Integer), range),
        ((Integer, Integer, Overloads__trunc__), range),
        ((Integer, Overloads__trunc__, Overloads__trunc__), range),
        ((Overloads__trunc__, Overloads__trunc__, Integer), range),
        ((Overloads__trunc__, Integer, Overloads__trunc__), range),
        ((Overloads__trunc__, Overloads__trunc__, Overloads__trunc__), range),
    ],

    'staticmethod': [
        ((AnyType,), staticmethod)
    ],

    'str': [
        ((), str),
        ((AnyType,), str)
    ],

    'complex': [
        ((), complex),
        ((Number,), complex),
        ((Str,), complex),
        ((Number, Number), complex),
        ((CastsToFloat,), complex),
        ((CastsToComplex,), complex),
        ((CastsToComplex, Number), complex),
        ((CastsToComplex, CastsToFloat), complex),
        ((Number, CastsToFloat), complex),
        ((CastsToFloat, Number), complex),
        ((CastsToFloat, CastsToFloat), complex),
    ],

    'property': [
        ((), property),
        ((Has__call__,), property),
        ((Has__call__, Has__call__,), property),
        ((Has__call__, Has__call__, Has__call__,), property),
        ((Has__call__, Has__call__, Has__call__, Str), property),
    ],

    'round': [
        ((RealNumber,), float),
        ((RealNumber, Integer), float),
        ((RealNumber, CastsToIndex), float),
        ((CastsToFloat,), float),
        ((CastsToFloat, Integer), float),
        ((CastsToFloat, CastsToIndex), float),
    ],

    'dir': [
        ((), list),
        ((AnyType,), list),
    ],

    'cmp': [
        ((AnyType, AnyType), int),
    ]

}
