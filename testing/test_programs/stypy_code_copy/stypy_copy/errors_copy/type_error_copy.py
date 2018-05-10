from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
import type_warning_copy
from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization
from stypy_copy import stypy_parameters_copy


class TypeError(Type):
    """
    A TypeError represent some kind of error found when handling types in a type inference program. It can be whatever
    type error we found: misuse of type members, incorrect operator application, wrong types for a certain operation...
     This class is very important in stypy because it models all the program type errors that later on will be reported
      to the user.
    """

    # The class stores a list with all the produced type errors so far
    errors = []
    # This flag is used to indicate that a certain sentence of the program has used an unsupported feature. Therefore
    # types cannot be accurately determined on the subsequent execution of the type inference program, and further
    # TypeErrors will only report this fact to avoid reporting false errors.
    usage_of_unsupported_feature = False

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
        if TypeError.usage_of_unsupported_feature:
            self.msg = "The type of this member could not be obtained due to the previous usage of an unsupported " \
                       "stypy feature"
        else:
            self.msg = msg

        if localization is None:
            localization = Localization(__file__, 1, 0)

        self.localization = localization

        if prints_msg and not TypeError.usage_of_unsupported_feature:
            TypeError.errors.append(self)

        # The error_msg is the full error to report to the user, composed by the passed msg and the stack trace.
        # We calculate it here to "capture" the precise execution point when the error is produced as stack trace is
        # dynamic and changes during the execution
        self.error_msg = self.__msg()

    def turn_to_warning(self):
        """
        Sometimes type errors have to be converted to warnings as some correct paths in the code exist although errors
        are detected. This is used, for example, when performing calls with union types. If some combinations are
        erroneus but at least one is possible, the errors for the wrong parameter type combinations are turned to
        warnings to report them precisely.
        :return:
        """
        type_warning_copy.TypeWarning.instance(self.localization, self.msg)
        TypeError.remove_error_msg(self)

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
        file_name = self.localization.file_name.split('/')[-1]
        file_name = file_name.split('\\')[-1]
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')

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
                    self.msg.strip(), self.localization.stack_trace)

        return "Compiler error in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
               (file_name, self.localization.line, self.localization.column,
                self.msg, self.localization.stack_trace)

    @staticmethod
    def print_error_msgs():
        """
        Prints all the error messages that were produced during a program analysis. Just for debugging
        :return:
        """
        for err in TypeError.errors:
            print err

    @staticmethod
    def get_error_msgs():
        """
        Gets all the error messages that were produced during a program analysis.
        :return: All the errors, sorted by line number
        """
        return sorted(TypeError.errors, key=lambda error: error.localization.line)

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
                TypeError.errors.remove(error)
        else:
            try:
                TypeError.errors.remove(error_obj)
            except:
                pass

    @staticmethod
    def reset_error_msgs():
        """
        Clears the global error message list
        :return:
        """
        TypeError.errors = []

    # ############################## OTHER TYPE METHODS ###############################
    """
    As errors are also stypy Type objects, they must provide the rest of its interface methods in order to allow
    the analysis of the program in an orthogonal fashion. These method do nothing, as they don't make sense within
    a TypeError. If methods of this object report errors upon called, the error reporting will display repeated
    errors at the end.
    """
    def get_python_entity(self):
        return self

    def get_python_type(self):
        return self

    def get_instance(self):
        return None

    # ############################## STORED TYPE METHODS ###############################

    def can_store_elements(self):
        return False

    def can_store_keypairs(self):
        return False

    def get_elements_type(self):
        return self

    def is_empty(self):
        return self

    def set_elements_type(self, localization, elements_type, record_annotation=True):
        return self

    def add_type(self, localization, type_, record_annotation=True):
        return self

    def add_types_from_list(self, localization, type_list, record_annotation=True):
        return self

    def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
        return self

    # ############################## MEMBER TYPE GET / SET ###############################

    def get_type_of_member(self, localization, member_name):
        return self

    def set_type_of_member(self, localization, member_name, member_value):
        return self

    # ############################## MEMBER INVOKATION ###############################

    def invoke(self, localization, *args, **kwargs):
        return self

    # ############################## STRUCTURAL REFLECTION ###############################

    def delete_member(self, localization, member):
        return self

    def supports_structural_reflection(self):
        return False

    def change_type(self, localization, new_type):
        return self

    def change_base_types(self, localization, new_types):
        return self

    def add_base_types(self, localization, new_types):
        return self

    # ############################## TYPE CLONING ###############################

    def clone(self):
        return self
