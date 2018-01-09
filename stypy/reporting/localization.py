#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.errors.stack_trace import StackTrace


class Localization(object):
    """
    This class is used to store caller information on function calls. It comprises the following data of the caller:
    - Line and column of the source code that performed the call
    - Python source code file name
    - Current stack trace of calls.

    Localization objects are key to generate accurate errors. Therefore most of the calls that stypy does uses
    localization instances for this matter
    """
    current_localization = None

    def __init__(self, file_name="[Not specified]", line=-1, column=-1):
        self.stack_trace = StackTrace.instance()
        self.file_name = file_name
        self.line = line
        self.column = column

    def get_stack_trace(self):
        """
        Gets the current stack trace
        :return:
        """
        return self.stack_trace

    def set_stack_trace(self, func_name, declared_arguments, arguments):
        """
        Modifies the stored stack trace appending a new stack trace (call begins)
        :param func_name:
        :param declared_arguments:
        :param arguments:
        :return:
        """
        self.stack_trace.set(self.file_name, self.line, self.column, func_name, declared_arguments, arguments)

    def unset_stack_trace(self):
        """
        Deletes the last set stack trace (call ends)
        :return:
        """
        self.stack_trace.unset()

    def __eq__(self, other):
        """
        Compares localizations using source line, column and file
        :param other:
        :return:
        """
        return self.file_name == other.file_name and self.line == other.line and self.column == other.column

    @staticmethod
    def set_current(current_localization):
        """

        """
        Localization.current_localization = current_localization

    @staticmethod
    def get_current():
        """
        """
        return Localization.current_localization
