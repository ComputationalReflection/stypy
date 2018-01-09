#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy import stypy_parameters
from stypy.reporting.module_line_numbering import ModuleLineNumbering


class Advice(object):
    """
    An advice is a bad coding practice that may not interrupt execution right now, but may lead to potential crashed
    or misbehaviors in the future
    """
    # All the advices produced during the execution are stored here
    advices = []

    def __init__(self, localization, msg, prints_msg=True):
        """
        Creates a warning with the provided message.
        :param localization: Caller information
        :param msg: Warning message
        :param prints_msg: As TypeErrors, TypeWarnings can also be silent if reporting them is not activated
        :return:
        """
        self.msg = msg
        self.localization = localization

        # Create the message here to capture the execution point, as stack traces are dynamic.
        self.advice_msg = self.__msg()

        if prints_msg and self not in Advice.advices:
            Advice.advices.append(self)

    def __str__(self):
        return self.advice_msg

    def __eq__(self, other):
        return self.localization == other.localization

    def __format_file_name(self):
        """
        Pretty-prints file name
        :return:
        """
        file_name = self.localization.file_name.split('/')[-1]
        file_name = file_name.split('\\')[-1]
        file_name = file_name.replace(stypy_parameters.type_inference_file_directory_name, '')

        return file_name

    def __msg(self):
        """
        Composes the full warning message, using the message, the localization, current file name and
        the stack trace. If available, it also displays the source code line when the warning is produced and a
        ^ marker indicating the position within the warning line.
        :return:
        """
        file_name = self.__format_file_name()

        source_code = ModuleLineNumbering.get_line_from_module_code(self.localization.file_name, self.localization.line)
        col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
                                                                     self.localization.line, self.localization.column)
        if source_code is not None:
            return "Coding advice in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s" % \
                   (file_name, self.localization.line, self.localization.column,
                    source_code, "" + col_offset,
                    self.msg.strip(), self.localization.stack_trace)

        return "Coding advice in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
               (file_name, self.localization.line, self.localization.column,
                self.msg, self.localization.stack_trace)

    @staticmethod
    def print_advice_msgs():
        """
        Prints all the warning messages that were produced during a program analysis. Just for debugging
        :return:
        """
        for err in Advice.advices:
            print(err)

    @staticmethod
    def reset_advice_msgs():
        Advice.advices = []

    @staticmethod
    def get_advice_msgs():
        """
        Gets all the warning messages that were produced during a program analysis.
        :return: All the errors, sorted by line number
        """
        return sorted(Advice.advices, key=lambda advice: advice.localization.line)

    # ############################################### PREDEFINED ADVICES ############################################

    @staticmethod
    def syntax_warning_name_assigned_before_global_advice(localization, name):
        return Advice(localization, "SyntaxWarning: name '{0}' is assigned to before global declaration".format(
            name
        ))

    @staticmethod
    def syntax_warning_name_used_before_global_advice(localization, name):
        return Advice(localization, "SyntaxWarning: name '{0}' is used prior to global declaration".format(
            name
        ))

    @staticmethod
    def redeclared_without_usage_advice(localization, name):
        return Advice(localization, "Redeclared '{0}' defined without usage".format(
            name
        ))

    @staticmethod
    def global_not_defined_advice(localization, name):
        return Advice(localization, "Global variable '{0}' is not defined at module level".format(
            name
        ))

    @staticmethod
    def value_not_defined_advice(localization, operation, value_name):
        return Advice(localization,
                      "Could not determine a concrete value for '{0}' of the operation '{1}'".format(value_name,
                                                                                                     operation))
