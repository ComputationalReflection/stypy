from singleton_copy import Singleton
from ...stypy_copy import stypy_parameters_copy
from ... import stypy_copy

@Singleton
class StackTrace:
    """
    This class allow TypeErrors to enhance the information they provide including the stack trace that lead to the
    line that produced the time error. This way we can precisely trace inside the program where is the type error in
    order to fix it. StackTrace information is built in the type inference program generated code and are accessed
    through TypeErrors, so no direct usage of this class is expected. There is a single stack trace object per execution
    flow.
    """
    def __init__(self):
        self.stack = []

    def set(self, file_name, line, column, function_name, declared_arguments, arguments):
        """
        Sets the stack trace information corresponding to a function call
        :param file_name: .py file where the function is placed
        :param line: Line when the function is declared
        :param column: Column when the function is declared
        :param function_name: Function name that is called
        :param declared_arguments: Arguments declared in the function code
        :param arguments: Passed arguments in the call
        :return:
        """
        self.stack.append((file_name, line, column, function_name, declared_arguments, arguments))

    def unset(self):
        """
        Pops the last added stack trace (at function exit)
        :return:
        """
        self.stack.pop()

    def __format_file_name(self, file_name):
        """
        Pretty-print the .py file name
        :param file_name:
        :return:
        """
        file_name = file_name.split('/')[-1]
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
        file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')

        return file_name

    def __format_type(self, type_):
        """
        Pretty-prints types
        :param type_:
        :return:
        """
        if isinstance(type_, stypy_copy.errors.type_error.TypeError):
            return "TypeError"
        return str(type_)

    def __pretty_string_params(self, declared_arguments, arguments):
        """
        Pretty-prints function parameters
        :param declared_arguments:
        :param arguments:
        :return:
        """
        zipped = zip(declared_arguments, arguments)
        ret_str = ""
        for tuple_ in zipped:
            ret_str += tuple_[0] + ": " + self.__format_type(tuple_[1]) + ", "

        return ret_str[:-2]

    def __pretty_string_vargargs(self, arguments):
        """
        Pretty-prints the variable list of arguments of a function
        :param arguments:
        :return:
        """
        if len(arguments) == 0:
            return ""

        ret_str = ", *starargs=["
        for arg in arguments:
            ret_str += self.__format_type(arg) + ", "

        return ret_str[:-2] + "]"

    def __pretty_string_kwargs(self, arguments):
        """
        Pretty-prints the keyword arguments of a function
        :param arguments:
        :return:
        """
        if len(arguments) == 0:
            return ""

        ret_str = ", **kwargs={"
        for key, arg in arguments.items():
            ret_str += str(key) + ": " + self.__format_type(arg) + ", "

        return ret_str[:-2] + "}"

    def to_pretty_string(self):
        """
        Prints each called function header and its parameters in a human-readable way, comprising the full stack
        trace information stored in this object.
        :return:
        """
        if len(self.stack) == 0:
            return ""
        s = "Call stack: [\n"

        for i in xrange(len(self.stack) - 1, -1, -1):
            file_name, line, column, function_name, declared_arguments, arguments = self.stack[i]

            file_name = self.__format_file_name(file_name)

            s += " - File '%s' (line %s, column %s)\n   Invocation to '%s(%s%s%s)'\n" % \
                 (file_name, line, column, function_name, self.__pretty_string_params(declared_arguments, arguments[0]),
                  self.__pretty_string_vargargs(arguments[1]), self.__pretty_string_kwargs(arguments[2]))
        s += "]"
        return s

    def __str__(self):
        return self.to_pretty_string()
