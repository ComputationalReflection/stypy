#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abstract_call_handler import AbstractCallHandler
from stypy.errors.type_warning import CannotResolveTypeWarning
from stypy.invokation.type_rules.type_groups.type_groups import DynamicType
from stypy.reporting.output_formatting import format_call


class CannotResolveTypeHandler(AbstractCallHandler):
    """
    This default call handler is used when a call cannot be analyzed using either type rules or calling the generated
    type inference equivalent program. In this case, a DynamicType instance is returned, that deactivates further type analysis,
    as calls with DynamicType always return DynamicType

    Note that this call handler is a last resort measure if none of the others can work with the callable entity.
    .
    """

    def can_be_applicable_to(self, callable_):
        """
        This handler is always applicable. Therefore, it must be examined the last
        :param callable_:
        :return:
        """
        return True

    def __call__(self, applicable_rules, localization, callable_, *arguments, **keyword_arguments):
        """
        Perform the call
        :param applicable_rules: Unused
        :param localization:
        :param callable_:
        :param arguments:
        :param keyword_arguments:
        :return:
        """

        str_call = format_call(callable_, arguments, keyword_arguments)
        w = CannotResolveTypeWarning(localization, str_call)
        return DynamicType()
