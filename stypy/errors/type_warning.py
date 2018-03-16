#!/usr/bin/env python
# -*- coding: utf-8 -*-
from advice import Advice
from stypy import errors
from stypy import stypy_parameters
from stypy.reporting.localization import Localization
from stypy.reporting.module_line_numbering import ModuleLineNumbering
import type_warning_postprocessing


class TypeWarning(object):
    """
    Class to model type warnings. This means that the type of a variable has been detected as invalid, but other
    options for types of these variables exist within the execution flow that result in a correct behavior. Therefore
    all the incorrect behaviors among an operation are warnings because there is a combination that can be valid.
    """

    # In strict mode all warnings are errors and warnings itself do not exist. If we try to create one, a TypeError
    # is returned instead.
    warnings_as_errors = False

    # All the warnings produced during the execution are stored here
    warnings = []

    dynamic_type_warning_included = False

    dynamic_warning = None

    recursion_warning_included = False

    recursion_warning = None

    type_warning_limit_hit = False

    def __init__(self, localization, msg, prints_msg=True, snap=None):
        """
        Creates a warning with the provided message.
        :param localization: Caller information
        :param msg: Warning message
        :param prints_msg: As TypeErrors, TypeWarnings can also be silent if reporting them is not activated
        :return:
        """
        self.packed = False
        self.message = msg
        if localization is None:
            localization = Localization(__file__, 1, 0)

        self.localization = localization
        if snap is None:
            self.stack_trace_snapshot = localization.stack_trace.get_snapshot()
        else:
            self.stack_trace_snapshot = snap

        # Create the message here to capture the execution point, as stack traces are dynamic.
        self.warn_msg = self.__msg()

        if (stypy_parameters.MAX_TYPE_WARNINGS > 0) and len(TypeWarning.warnings) > stypy_parameters.MAX_TYPE_WARNINGS:
            TypeWarning.type_warning_limit_hit = True
            return

        if prints_msg and self not in TypeWarning.warnings:
            TypeWarning.warnings.append(self)

    @staticmethod
    def instance(localization, msg, prints_msg=True, snap=None):
        """
        Creates a TypeWarning instance (or a TypeError if strict mode is used).
        :param localization: Caller information
        :param msg: Warning message
        :param prints_msg: Silent warning?
        :param snap: Snapshot of the stack trace in which the warning is produced
        :return:
        """
        if TypeWarning.warnings_as_errors:
            return errors.type_error.TypeError(localization, msg, prints_msg)
        else:
            return TypeWarning(localization, msg, prints_msg, snap)

    def __str__(self):
        """
        Warning to str
        :return:
        """
        return self.warn_msg

    def __format_file_name(self):
        """
        Pretty-prints file name
        :return:
        """
        file_name = self.localization.file_name.replace('\\', '/')
        file_name = file_name.replace(stypy_parameters.ROOT_PATH, '')
        file_name = file_name.replace('stypy/sgmc/sgmc_cache/', '')
        file_name = file_name.replace('/sgmc/sgmc_cache', '')
        file_name = file_name.replace('site_packages/', '')
        file_name = file_name.replace('.pyc', '.py')

        return file_name

    def rebuild_message(self):
        self.warn_msg = self.__msg()

    def __msg(self):
        """
        Composes the full warning message, using the message, the localization, current file name and
        the stack trace. If available, it also displays the source code line when the warning is produced and a
        ^ marker indicating the position within the warning line.
        :return:
        """
        file_name = self.__format_file_name()

        source_code = ModuleLineNumbering.get_line_from_module_code(self.localization.file_name, self.localization.line)

        if hasattr(self.localization, 'column_offsets_for_packed_warnings'):
            col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
                                                                         self.localization.line,
                                                                         self.localization.column,
                                                                         self.localization.column_offsets_for_packed_warnings)
            col_numbers = ModuleLineNumbering.get_column_number_str(self.localization.column,
                                                                    self.localization.column_offsets_for_packed_warnings)

        else:
            col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
                                                                         self.localization.line,
                                                                         self.localization.column)
            col_numbers = "column %s" % str(self.localization.column)

        if source_code is not None:
            return "Type warning in file '%s' (line %s, %s):\n%s\n%s\n\t%s.\n\n%s" % \
                   (file_name, self.localization.line, col_numbers,
                    source_code, "" + col_offset,
                    self.message.strip(), self.stack_trace_snapshot)

        return "Type warning in file '%s' (line %s, %s):\n%s.\n\n%s" % \
               (file_name, self.localization.line, col_numbers,
                self.message, self.stack_trace_snapshot)

    @staticmethod
    def print_warning_msgs():
        """
        Prints all the warning messages that were produced during a program analysis. Just for debugging
        :return:
        """
        for err in TypeWarning.warnings:
            print (err)

    @staticmethod
    def reset_warning_msgs():
        """
        Removes all current warning messages
        :return:
        """
        TypeWarning.warnings = []

    @staticmethod
    def get_warning_msgs():
        """
        Gets all the warning messages that were produced during a program analysis.
        :return: All the errors, sorted by line number
        """
        return sorted(TypeWarning.warnings, key=lambda warning: warning.localization.line)

    def __eq__(self, other):
        """
        Type warning custom comparison
        :param other:
        :return:
        """
        if not isinstance(other, TypeWarning):
            return False

        if len(self.stack_trace_snapshot.stack) == 0 and len(other.stack_trace_snapshot.stack) == 0:
            return self.warn_msg == other.warn_msg

        if self.warn_msg == other.warn_msg:
            return True
        if self.localization.line != other.localization.line:
            return False
        if self.localization.column != other.localization.column:
            return False
        if len(self.stack_trace_snapshot.stack) != len(other.stack_trace_snapshot.stack):
            return False

        for i in xrange(len(self.stack_trace_snapshot.stack)):
            my_file_name, my_line, my_column, my_function_name, my_declared_arguments, my_arguments = \
                self.stack_trace_snapshot.stack[i]
            other_file_name, other_line, other_column, other_function_name, other_declared_arguments, other_arguments = \
                other.stack_trace_snapshot.stack[i]
            if my_file_name != other_file_name or my_line != other_line or my_column != other_column or \
                    my_function_name != other_function_name:
                return False
        return True

    @staticmethod
    def remove_warning_msg(error_obj):
        """
        Deletes an warning message from the global warning list.
        :param error_obj:
        :return:
        """
        if isinstance(error_obj, list):
            for error in error_obj:
                try:
                    TypeWarning.warnings.remove(error)
                except:
                    pass
        else:
            try:
                TypeWarning.warnings.remove(error_obj)
            except:
                pass

    @staticmethod
    def pack_warnings():
        """This method consolidates multiple warnings into one provided the following conditions are met:
        1) Belong to the same line
        2) Has a message that refer to potential undefined types
        3) Has the same call stack.

        In that case, a single warning is produced, and multiple columns are stored to indicate all the places in the
        line that may present this warning. This greatly helps to lower the amount of reported warnings produced when
        multiple arithmetic operations deal with operands with potential UndefinedType values.
        """
        type_warning_postprocessing.pack_undefined_warnings(TypeWarning)
        type_warning_postprocessing.pack_warnings_with_the_same_line_and_stack_trace(TypeWarning)

    # ######################################## PREDEFINED WARNINGS ########################################

    @staticmethod
    def enable_usage_of_dynamic_types_warning(localization, fname=""):
        if not TypeWarning.dynamic_type_warning_included:
            t = TypeWarning(localization,
                            "Usage of Python functions that dynamically evaluates Python code. This Python "
                            "feature is not yet supported. Errors reported from this line on may not be "
                            "accurate")
            TypeWarning.dynamic_type_warning_included = True
            TypeWarning.dynamic_warning = t

    @staticmethod
    def enable_usage_of_recursion_warning(fname=""):
        if not TypeWarning.recursion_warning_included:
            t = TypeWarning(None, "This program uses recursion. Type inference might not be accurate")
            TypeWarning.recursion_warning_included = True
            TypeWarning.recursion_warning = t


class UnreferencedLocalVariableTypeWarning(TypeWarning):
    """
    This special type of warning is only used if coding advices are activated. It models those cases when a global
    variable is read and later on is written to without using the global keyword. Python decides to report an error
    in this case, but in the source line that reads the value instead of the source line that write a value to the
    variable. A coding advice is generated if this kind of programming pattern is detected within the program.
    """

    def __init__(self, localization, name, context):
        TypeWarning.__init__(self, localization,
                             "Read access detected over a global name '{0}'. Any attempt of writing to this "
                             "name without using the 'global' keyword first will result into an 'UnboundLocalError: "
                             "local variable '{0}' referenced before assignment' runtime error".format(name),
                             prints_msg=True)
        self.name = name
        self.context = context

    def turn_to_advice(self):
        """
        Turns this warning to a coding advice
        :return:
        """
        return Advice(self.localization, self.message, True)


class CannotResolveTypeWarning(TypeWarning):
    """
    This warning is used when stypy has no means of resolving a call: a type inference program cannot be generated and
    there are no rules tied to a call to resolve it.
    """

    def __init__(self, localization, description):
        TypeWarning.__init__(self, localization,
                             "CANNOT RESOLVE: " + description,
                             prints_msg=True)

    def turn_to_advice(self):
        """
        Turns this warning to a coding advice
        :return:
        """
        return Advice(self.localization, self.message, True)
