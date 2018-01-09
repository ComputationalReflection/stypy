from stypy_copy import stypy_parameters_copy


class ModuleLineNumbering:
    """
    This is an utility class to put line numbers to source code lines. Numbered source code lines are added to the
    beginning of generated type inference programs to improve its readability if the generated source code has to be
    reviewed. Functions of this class are also used to report better errors.
    """
    file_numbered_code_cache = dict()

    def __init__(self):
        pass

    @staticmethod
    def clear_cache():
        """
        Numbered lines of source files are cached to improve performance. This clears this cache.
        :return:
        """
        ModuleLineNumbering.file_numbered_code_cache = dict()

    @staticmethod
    def __normalize_path_name(path_name):
        """
        Convert file paths into a normalized from
        :param path_name: File path
        :return: Normalized file path
        """
        path_name = path_name.replace("\\", "/")
        return path_name

    @staticmethod
    def __calculate_line_numbers(file_name, module_code):
        """
        Utility method to put numbers to the lines of a source code file, caching it once done
        :param file_name: Name of the file
        :param module_code: Code of the file
        :return: str with the original source code, attaching line numbers to it
        """
        if file_name in ModuleLineNumbering.file_numbered_code_cache.keys():
            return ModuleLineNumbering.file_numbered_code_cache[file_name]

        numbered_original_code_lines = module_code.split('\n')

        number_line = dict()
        for i in range(len(numbered_original_code_lines)):
            number_line[i + 1] = numbered_original_code_lines[i]

        ModuleLineNumbering.file_numbered_code_cache[
            ModuleLineNumbering.__normalize_path_name(file_name)] = number_line

        return number_line

    @staticmethod
    def put_line_numbers_to_module_code(file_name, module_code):
        """
        Put numbers to the lines of a source code file, caching it once done
        :param file_name: Name of the file
        :param module_code: Code of the file
        :return: str with the original source code, attaching line numbers to it
        """
        number_line = ModuleLineNumbering.__calculate_line_numbers(file_name, module_code)
        numbered_original_code = ""
        for number, code in number_line.items():
            numbered_original_code += str(number) + ": " + code + "\n"

        return numbered_original_code

    @staticmethod
    def __get_original_source_code_file(file_name):
        """
        From a type inference code file name, obtain the original source code file name
        :param file_name: File name (of a type inference program)
        :return: File name (of a Python program)
        """
        if stypy_parameters_copy.type_inference_file_postfix in file_name:
            file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, "")

        if stypy_parameters_copy.type_inference_file_directory_name in file_name:
            file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name + "/", "")

        return file_name

    @staticmethod
    # TODO: Revise this, if the code is not cached, is this returning something?
    def get_line_numbered_module_code(file_name):
        """
        Get the numbered source code of the passed file name
        :param file_name: File name
        :return: Numbered source code (str)
        """
        normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
        normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)

        try:
            for file in ModuleLineNumbering.file_numbered_code_cache.keys():
                if file in normalized_file_name:
                    return ModuleLineNumbering.file_numbered_code_cache[file]
        except:
            return None

    @staticmethod
    def get_line_from_module_code(file_name, line_number):
        """
        Get the source code line line_number from the source code of file_name. This is used to report type errors,
        when we also include the source line.

        :param file_name: Python src file
        :param line_number: Line to get
        :return: str (line of source code)
        """
        normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
        normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)

        linenumbers = ModuleLineNumbering.get_line_numbered_module_code(normalized_file_name)
        if linenumbers is not None:
            try:
                return linenumbers[line_number]
            except:
                return None
        return None

    @staticmethod
    def get_column_from_module_code(file_name, line_number, col_offset):
        """
        Calculates the position of col_offset inside the line_number of the file file_name, so we can physically locate
         the column within the file to report meaningful errors. This is used then reporting type error, when we also
         include the error line source code and the position within the line that has the error.
        :param file_name:
        :param line_number:
        :param col_offset:
        :return:
        """
        normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
        normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)

        line = ModuleLineNumbering.get_line_from_module_code(normalized_file_name, line_number)
        if line is None:
            return None

        blank_line = " " * col_offset + "^"

        return blank_line
