from ....errors_copy.stack_trace_copy import StackTrace


class Localization:
    """
    This class is used to store caller information on function calls. It comprises the following data of the caller:
    - Line and column of the source code that performed the call
    - Python source code file name
    - Current stack trace of calls.

    Localization objects are key to generate accurate errors. Therefore most of the calls that stypy does uses
    localization instances for this matter
    """
    def __init__(self, file_name="[Not specified]", line=0, column=0):
        self.stack_trace = StackTrace.Instance()
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

    def clone(self):
        """
        Deep copy (Clone) this object
        :return:
        """
        return Localization(self.file_name, self.line, self.column)
