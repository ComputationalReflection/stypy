import os

from stypy.type_expert_system.types.library.known_types.known_python_types_handling import get_type_name
from stypy import stypy_parameters


class TypeDataFileWriter:
    def __init__(self, file_path):
        self.already_processed_contexts = []
        self.type_file_txt = "import types\n\ntest_types = {\n"
        file_path = file_path.replace('\\', '/')
        self.file_path = file_path
        self.dest_folder = os.path.dirname(file_path)
        self.type_file = (file_path.split('/')[-1])[0:-3].split('__')[
                             0] + stypy_parameters.type_data_file_postfix + ".py"

    def add_type_dict_for_main_context(self, var_dict):
        self.__add_type_dict_for_context(var_dict)

    def add_type_dict_for_context(self, var_dict):
        import traceback

        func_name = traceback.extract_stack(None, 2)[0][2]

        self.__add_type_dict_for_context(var_dict, func_name)

    def __add_type_dict_for_context(self, var_dict, context="__main__"):
        if context in self.already_processed_contexts:
            return

        vars = filter(lambda var: not "__" in var and not var == 'stypy' and
                                  not var == 'type_test', var_dict.keys())

        self.type_file_txt += "    '" + context + "': {\n"
        for var in vars:
            self.type_file_txt += "        '" + var + "': " + get_type_name(type(var_dict[var])) + ", \n"

        self.type_file_txt += "    " + "}, \n"

        self.already_processed_contexts.append(context)

    def generate_type_data_file(self):
        print self.dest_folder
        print self.type_file
        self.type_file_txt += "}\n"
        with open(self.dest_folder + "/" + self.type_file, 'w') as outfile:
            outfile.write(self.type_file_txt)

