#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

import type_warning
from stypy import stypy_parameters
from stypy.invokation.type_rules import type_groups
from stypy.reporting.localization import Localization
from stypy.reporting.module_line_numbering import ModuleLineNumbering


class StypyTypeError(TypeError):
    """
    A StypyTypeError represent some kind of error found when handling types in a type inference program. It can be any
    type error we found: misuse of type members, incorrect operator application, wrong types for a certain operation...
     This class is very important in stypy because it models all the program type errors that later on will be reported
      to the user.
    """

    # The class stores a list with all the produced type errors so far
    errors = []
    type_error_limit_hit = False

    def __init__(self, localization=None, msg="", prints_msg=True):
        """
        Creates a particular instance of a type error amd adds it to the error list
        :param localization: Caller information
        :param msg: Error to report to the user
        :param prints_msg: Determines if this error is silent (report its message) or not. Some error are silent
        because they are generated to generate a more accurate TypeError later once the program determines that
        TypeErrors exist on certain places of the analyzed program. This is used in certain situations to avoid
        reporting the same error multiple times
        :return:
        """

        # This happens when a declared class has its parent inferred as an error type. If that happens, the class is
        # built using this constructor as a base class initializer. We throw an error indicating the issue and do not
        # build a standard error in this case.
        if type(localization) is str:
            StypyTypeError(Localization.get_current(), "The parent class of class '" +
                           localization + "' is an error")
            return

        self.message = msg

        if localization is None:
            localization = Localization(__file__, 1, 0)

        self.localization = localization
        self.stack_trace_snapshot = localization.stack_trace.get_snapshot()

        # The error_msg is the full error to report to the user, composed by the passed msg and the stack trace.
        # We calculate it here to "capture" the precise execution point when the error is produced as stack trace is
        # dynamic and changes during the execution
        self.error_msg = self.__msg()

        if (stypy_parameters.MAX_TYPE_ERRORS > 0) and len(StypyTypeError.errors) > stypy_parameters.MAX_TYPE_ERRORS:
            StypyTypeError.type_error_limit_hit = True
            return

        # Add this error to the general error list if not already present
        if prints_msg and self not in StypyTypeError.errors:
            StypyTypeError.errors.append(self)

    def turn_to_warning(self):
        """
        Sometimes type errors have to be converted to warnings as some correct paths in the code exist although errors
        are detected. This is used, for example, when performing calls with union types. If some combinations are
        erroneous but at least one is possible, the errors for the wrong parameter type combinations are turned to
        warnings to report them precisely.
        :return:
        """
        type_warning.TypeWarning.instance(self.localization, self.message, snap=self.stack_trace_snapshot)
        StypyTypeError.remove_error_msg(self)

    def __eq__(self, other):
        """
        Type error custom comparison
        :param other:
        :return:
        """
        if not isinstance(other, StypyTypeError):
            return False

        if len(self.stack_trace_snapshot.stack) == 0 and len(other.stack_trace_snapshot.stack) == 0:
            return self.error_msg == other.error_msg

        if self.error_msg == other.error_msg:
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

    def __str__(self):
        """
        Visual representation of the error (full message: error + stack trace)
        :return:
        """
        return self.error_msg

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

    def __msg(self):
        """
        Composes the full error message, using the error message, the error localization, current file name and
        the stack trace. If available, it also displays the source code line when the error is produced and a
        ^ marker indicating the position within the error line.
        :return:
        """
        file_name = self.__format_file_name()

        source_code = ModuleLineNumbering.get_line_from_module_code(self.localization.file_name, self.localization.line)
        col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
                                                                     self.localization.line, self.localization.column)

        if source_code is not None:
            return "Compiler error in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s" % \
                   (file_name, self.localization.line, self.localization.column,
                    source_code, "" + col_offset,
                    self.message.strip(), self.stack_trace_snapshot)

        return "Compiler error in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
               (file_name, self.localization.line, self.localization.column,
                self.message, self.stack_trace_snapshot)

    def origins_in(self, localization):
        """
        Determines whether this error has its origin in the provided localization or not
        :param localization:
        :return:
        """
        stack_trace = localization.stack_trace.stack

        if len(self.stack_trace_snapshot.stack) == 0 and len(stack_trace) == 0:
            return self.localization.column == localization.column and self.localization.line == localization.line \
                   and self.localization.file_name == localization.file_name

        if len(self.stack_trace_snapshot.stack) <= len(stack_trace):
            return False

        for i in xrange(len(stack_trace)):
            my_file_name, my_line, my_column, my_function_name, my_declared_arguments, my_arguments = \
                self.stack_trace_snapshot.stack[i]
            other_file_name, other_line, other_column, other_function_name, other_declared_arguments, \
            other_arguments = stack_trace[i]

            if my_file_name != other_file_name or my_line != other_line or my_column != other_column or \
                            my_function_name != other_function_name:
                return False
        return True

    @staticmethod
    def print_error_msgs():
        """
        Prints all the error messages that were produced during a program analysis. Just for debugging
        :return:
        """
        for err in StypyTypeError.errors:
            print (err)

    @staticmethod
    def get_error_msgs():
        """
        Gets all the error messages that were produced during a program analysis.
        :return: All the errors, sorted by line number
        """
        return sorted(StypyTypeError.errors, key=lambda error: error.localization.line)

    @staticmethod
    def remove_error_msg(error_obj):
        """
        Deletes an error message from the global error list. As we said, error messages might be turn to warnings, so
        we must delete them afterwards
        :param error_obj:
        :return:
        """
        if isinstance(error_obj, list):
            for error in error_obj:
                try:
                    StypyTypeError.errors.remove(error)
                except:
                    pass
        else:
            try:
                StypyTypeError.errors.remove(error_obj)
            except:
                pass

    @staticmethod
    def remove_errors_condition(function):
        """
        Remove all errors that fulfill the passed condition
        :param function:
        :return:
        """
        unfiltered = []
        for err in StypyTypeError.errors:
            if not function(err):
                unfiltered.append(err)

        StypyTypeError.errors = unfiltered

    @staticmethod
    def reset_error_msgs():
        """
        Clears the global error message list
        :return:
        """
        StypyTypeError.errors = []

    @staticmethod
    def __format_type(type_):
        """
        Pretty-prints types
        :param type_:
        :return:
        """
        if type(type_) is str:
            return type_

        if type(type_) is type_groups.type_groups.UndefinedType or type_ is type_groups.type_groups.UndefinedType:
            return "<undefined type>"

        if isinstance(type_, StypyTypeError):
            return "TypeError"
        try:
            # Arg is already a type
            if type(type_) is types.TypeType:
                return str(type_)

            if type_ is None:
                return "None"

            basic_types = [int, bool, float, long]
            if type(type_) in basic_types:
                return str(type_)

            return str(type_)
        except:
            try:
                return type(type_).__name__
            except:
                return str(type(type_))

    # ############################## PREDEFINED ERROR TYPES ###############################

    @staticmethod
    def name_not_defined_error(localization, name, where=None):
        if where is not None:
            return StypyTypeError(localization, "The name '{0}' does not exist {1}.".format(name, where))
        return StypyTypeError(localization, "The name '{0}' does not exist in the current context".format(name))

    @staticmethod
    def member_not_defined_error(localization, obj, member_name):
        try:
            if "RecursionType" in type(obj).__name__ :
                return StypyTypeError(localization,
                                      "Cannot locate members ('{0}') in results of recursive calls".format(
                                          member_name))
        except:
            pass
        return StypyTypeError(localization,
                              "The member '{0}' of the type '{1}' does not exist in the current context".format(
                                  member_name,
                                  StypyTypeError.__format_type(obj)))

    @staticmethod
    def member_do_not_exist_error(localization, obj, member_name):
        return StypyTypeError(localization,
                              "Cannot locate a member named '{0}' on '{1}' ".format(
                                  member_name, StypyTypeError.__format_type(obj)))


    @staticmethod
    def member_cannot_be_set_error(localization, obj, member_name, type_, reason):
        return StypyTypeError(localization,
                              "The member '{0}' of object '{1}' cannot be set to the value '{2}' due to the following"
                              " reasons: '{3}'".format(
                                  member_name, StypyTypeError.__format_type(obj), StypyTypeError.__format_type(type_),
                                  str(reason)))

    @staticmethod
    def member_cannot_be_deleted_error(localization, obj, member_name, reason):
        return StypyTypeError(localization,
                              "The member '{0}' of object '{1}' cannot be deleted due to the following"
                              " reasons: '{2}'".format(
                                  member_name, StypyTypeError.__format_type(obj), str(reason)))

    @staticmethod
    def no_type_has_member_error(localization, types_, member_name):
        return StypyTypeError(localization, "None of the possible types ('{1}') has the member '{0}'".format(
            member_name, str(types_)))

    @staticmethod
    def no_type_can_set_member_error(localization, types_, member_name):
        return StypyTypeError(localization, "None of the possible types ('{1}') can set a value for the member '{0}'".
                              format(member_name, str(types_)))

    @staticmethod
    def no_type_can_delete_member_error(localization, types_, member_name):
        return StypyTypeError(localization, "None of the possible types ('{1}') can delete the member '{0}'".
                              format(member_name, str(types_)))

    @staticmethod
    def unbound_local_error(localization, var_name):
        return StypyTypeError(localization, "UnboundLocalError: local variable '{0}' referenced before assignment".
                              format(var_name))

    @staticmethod
    def unknown_python_builtin_type_error(localization, var_name):
        return StypyTypeError(localization, "There is no Python builtin type named '{0}'".format(var_name))

    @staticmethod
    def wrong_return_type_error(localization, func_name, call_result, expected_type=None):
        if expected_type is None:
            return StypyTypeError(localization,
                                  'Method {0} returned an invalid type (type {1})'.
                                  format(func_name, StypyTypeError.__format_type(call_result)))

        return StypyTypeError(localization,
                              'Conversion method {0} returned a non-{2} (type {1})'.
                              format(func_name, StypyTypeError.__format_type(call_result),
                                     StypyTypeError.__format_type(expected_type)))

    @staticmethod
    def object_must_define_member_error(localization, obj_name, member_name):
        return StypyTypeError(localization, "{0} must define a {1} member".format(obj_name, member_name))

    @staticmethod
    def invalid_callable_error(localization, callable_name, context, msg):
        return StypyTypeError(localization, "Invalid {0} for {1}: {2}".format(callable_name, context, msg))

    @staticmethod
    def wrong_parameter_type_error(localization, expected, found):
        return StypyTypeError(localization, "{0} argument expected, got {1}".format(expected, found))

    @staticmethod
    def object_must_be_type_error(localization, obj_name, expected_type, found=None):
        if found is None:
            return StypyTypeError(localization, "{0} must be of {1} type".format(obj_name, expected_type))
        return StypyTypeError(localization,
                              "{0} must be of {1} type, but {2} found".format(obj_name, expected_type, found))

    @staticmethod
    def function_cannot_be_applicable_error(localization, func_name, elements_name):
        return StypyTypeError(localization,
                              "{0} cannot be applicable over {1}".format(func_name, elements_name))

    @staticmethod
    def key_error(localization, key_type):
        return StypyTypeError(localization, "KeyError: The key of type '{0}' do not exist".format(
            StypyTypeError.__format_type(key_type)))

    @staticmethod
    def invalid_length_error(localization, context, expected_l, found_l):
        return StypyTypeError(localization,
                              "{0} has length {1}; {2} is required".format(context, expected_l, found_l))


class ConstructorParameterError(StypyTypeError):
    """
    Special kind of type error of stypy that handles errors in constructor parameters
    """

    def __init__(self, localization=None, msg="", prints_msg=True):
        StypyTypeError.__init__(self, localization, msg, prints_msg)
