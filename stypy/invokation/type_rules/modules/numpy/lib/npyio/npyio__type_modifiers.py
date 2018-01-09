#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, IterableDataStructure, \
    IterableDataStructureWithTypedElements, Str, DynamicType


class TypeModifiers:
    @staticmethod
    def genfromtxt(localization, proxy_obj, arguments):
        dvar = call_utilities.parse_varargs_and_kwargs(localization, arguments,
                                                       ['fname', 'dtype', 'comments', 'delimiter', 'skiprows',
                                                        'skip_header',
                                                        'skip_footer',
                                                        'converters',
                                                        'missing_values',
                                                        'filling_values',
                                                        'usecols',
                                                        'names',
                                                        'excludelist',
                                                        'deletechars',
                                                        'defaultfmt',
                                                        'autostrip',
                                                        'replace_space',
                                                        'case_sensitive',
                                                        'unpack',
                                                        'usemask',
                                                        'loose',
                                                        'invalid_raise',
                                                        'max_rows']
                                                       , {
                                                           'fname': [file, Str,
                                                                     IterableDataStructureWithTypedElements(Str)],
                                                           'dtype': type,
                                                           'comments': Str,
                                                           'delimiter': [Str, Integer],
                                                           'skiprows': Integer,
                                                           'skip_header': Integer,
                                                           'skip_footer': Integer,
                                                           'converters': IterableDataStructureWithTypedElements(
                                                               types.LambdaType),
                                                           'missing_values': IterableDataStructureWithTypedElements(
                                                               Str),
                                                           'filling_values': IterableDataStructure,
                                                           'usecols': IterableDataStructureWithTypedElements(Integer),
                                                           'names': [bool, Str,
                                                                     IterableDataStructureWithTypedElements(Str)],
                                                           'excludelist': IterableDataStructureWithTypedElements(Str),
                                                           'deletechars': Str,
                                                           'defaultfmt': Str,
                                                           'autostrip': bool,
                                                           'replace_space': Str,
                                                           'case_sensitive': [bool, Str],
                                                           'unpack': bool,
                                                           'usemask': bool,
                                                           'loose': bool,
                                                           'invalid_raise': bool,
                                                           'max_rows': Integer,
                                                       }, 'genfromtxt', 0)

        if isinstance(dvar, StypyTypeError):
            return dvar

        if 'case_sensitive' in dvar.keys():
            if Str == type(dvar['case_sensitive']):
                temp = call_utilities.check_possible_values(dvar, 'case_sensitive', ['upper', 'lower'])
                if isinstance(temp, StypyTypeError):
                    return temp

        return call_utilities.create_numpy_array(DynamicType())
