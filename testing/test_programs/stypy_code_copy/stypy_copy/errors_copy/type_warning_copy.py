from ...stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering
from ...stypy_copy import errors_copy
from ...stypy_copy import stypy_parameters_copy


class TypeWarning:
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

        if prints_msg:
            TypeWarning.warnings.append(self)
            # Create the message here to capture the execution point, as stack traces are dynamic.
            self.warn_msg = self.__msg()

    @staticmethod
    def instance(localization, msg, prints_msg=True):
        """
        Creates a TypeWarning instance (or a TypeError if strict mode is used).
        :param localization: Caller information
        :param msg: Warning message
        :param prints_msg: Silent warning?
        :return:
        """
        if TypeWarning.warnings_as_errors:
            return errors_copy.type_error_copy.TypeError(localization, msg, prints_msg)
        else:
            return TypeWarning(localization, msg, prints_msg)

    def __str__(self):
        return self.warn_msg

    def __format_file_name(self):
        """
        Pretty-prints file name
        :return:
        """
        file_name = self.localization.file_name.split('/')[-1]
        file_name = file_name.split('\\')[-1]
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')

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
            return "Warning in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s" % \
                   (file_name, self.localization.line, self.localization.column,
                    source_code, "" + col_offset,
                    self.msg.strip(), self.localization.stack_trace)

        return "Warning in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
               (file_name, self.localization.line, self.localization.column,
                self.msg, self.localization.stack_trace)

    @staticmethod
    def print_warning_msgs():
        """
        Prints all the warning messages that were produced during a program analysis. Just for debugging
        :return:
        """
        for err in TypeWarning.warnings:
            print err

    @staticmethod
    def reset_warning_msgs():
        TypeWarning.warnings = []

    @staticmethod
    def get_warning_msgs():
        """
        Gets all the warning messages that were produced during a program analysis.
        :return: All the errors, sorted by line number
        """
        return sorted(TypeWarning.warnings, key=lambda warning: warning.localization.line)

    # TODO: Remove?
    # @classmethod
    # def set_warning_msgs(cls, warn_list):
    #     """
    #     Substitute the warning messages list by the provided one
    #     :param warn_list: New warning list
    #     :return:
    #     """
    #     TypeWarning.warnings = warn_list

    # @classmethod
    # def clone_existing_warnings(cls):
    #     """
    #     Clones the warning list
    #     :return:
    #     """
    #     result = list()
    #
    #     for warning in TypeWarning.warnings:
    #         result.append(warning)
    #
    #     return result


class UnreferencedLocalVariableTypeWarning(TypeWarning):
    pass
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
                             "local variable '{0}' referenced before assignment' runtime error".format(name))
        self.name = name
        self.context = context
