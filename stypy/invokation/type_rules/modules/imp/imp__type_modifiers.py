

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stypy.errors.type_error import StypyTypeError
from stypy.invokation.handlers import call_utilities
from stypy.invokation.type_rules.type_groups.type_group_generator import Integer, RealNumber, \
    IterableDataStructureWithTypedElements, IterableDataStructure, DynamicType
import imp

class TypeModifiers:
    @staticmethod
    def load_source(localization, proxy_obj, arguments):
        if arguments[0] != "" and len(arguments) == 1:
            return imp.load_source(arguments[0])
        if arguments[0] != "" and len(arguments) == 2:
            return imp.load_source(arguments[0], arguments[1])
        if arguments[0] != "" and len(arguments) == 3:
            return imp.load_source(arguments[0], arguments[1], arguments[2])

        return DynamicType
