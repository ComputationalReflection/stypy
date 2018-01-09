import os

from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy import get_type_name
from stypy_copy import stypy_parameters_copy


class TypeDataFileWriter:
    """
    A simple writer to write type data files, that are used to unit test the code generation of stypy when applied
    to the programs included in the test battery. A type data file has a format like this, written by this code:

    import types
    from stypy import union_type
    from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType

    test_types = {
        '__init__': {
            'StringComp': int,
            'self': types.InstanceType,
            'Discr': int,
            'PtrComp': types.NoneType,
            'IntComp': int,
            'EnumComp': int,
        },
        'Proc5': {
        },
        'Proc4': {
            'BoolLoc': int #bool,
        },
        'Func1': {
            'CharLoc2': str,
            'CharLoc1': str,
            'CharPar2': str,
            'CharPar1': str,
        },
        'Func2': {
            'StrParI1': str,
            'CharLoc': union_type.UnionType.create_union_type_from_types(str, UndefinedType()),
            'StrParI2': str,
            'IntLoc': int,
        },
        'Proc7': {
            'IntParOut': int,
            'IntLoc': int,
            'IntParI1': int,
            'IntParI2': int,
        },
        'Proc8': {
            'Array1Par': list,
            'IntParI2': int,
            'IntParI1': int,
            'Array2Par': list,
            'IntLoc': int,
            'IntIndex': int,
        },
        'copy': {
            'self': types.InstanceType,
        },
        'Proc3': {
            'PtrParOut': types.InstanceType,
        },
        'Func3': {
            'EnumLoc': int,
            'EnumParIn': int,
        },
        'Proc6': {
            'EnumParIn': int,
            'EnumParOut': int,
        },
        'Proc1': {
            'NextRecord': types.InstanceType,
            'PtrParIn': types.InstanceType,
        },
        'Proc2': {
            'EnumLoc': int,
            'IntParIO': int,
            'IntLoc': int,
        },
        'Proc0': {
            'EnumLoc': int,
            'String2Loc': str,
            'IntLoc2': int,
            'IntLoc3': int,
            'String1Loc': str,
            'IntLoc1': int,
            'i': int,
            'CharIndex': str,
            'benchtime': float,
            'loopsPerBenchtime': float,
            'loops': int,
            'nulltime': float,
            'starttime': float,
        },
        'pystones': {
            'loops': int,
        },
        'main': {
            'stones': int, #should be float
            'loops': int,
            'benchtime': int, #should be float
        },
        '__main__': {
            'Array1Glob': list,
            'loops': int,
            'TRUE': int,
            'Record': types.ClassType,
            'Func3': types.LambdaType,
            'Func2': types.LambdaType,
            'Func1': types.LambdaType,
            'Array2Glob': list,
            'clock': types.BuiltinFunctionType,
            'BoolGlob': union_type.UnionType.create_union_type_from_types(int, bool),
            'LOOPS': int,
            'main': types.LambdaType,
            'Proc8': types.LambdaType,
            'Char2Glob': str,
            'pystones': types.LambdaType,
            'PtrGlbNext': union_type.UnionType.create_union_type_from_types(types.InstanceType, types.NoneType),
            'nargs': int,
            'sys': types.ModuleType,
            'TypeDataFileWriter': types.ClassType,
            'IntGlob': int,
            'Ident4': int,
            'Ident5': int,
            'FALSE': int,
            'Ident1': int,
            'Ident2': int,
            'Ident3': int,
            'Char1Glob': str,
            'PtrGlb': types.NoneType, #types.InstanceType,
            'error': types.LambdaType,
            'Proc5': types.LambdaType,
            'Proc4': types.LambdaType,
            'Proc7': types.LambdaType,
            'Proc6': types.LambdaType,
            'Proc1': types.LambdaType,
            'Proc0': types.LambdaType,
            'Proc3': types.LambdaType,
            'Proc2': types.LambdaType,
        },
    }
    As we see, there are a fixed number of imports and a dictionary called test_types with str keys and dict values.
    Each key correspond to the name of a function/method and the value is the variable table (name: type) expected in
    this context.
    """

    def __init__(self, file_path):
        """
        Creates a writer for type data files
        :param file_path: File to write to
        :return:
        """
        self.already_processed_contexts = []
        self.type_file_txt = "import types\n\ntest_types = {\n"
        file_path = file_path.replace('\\', '/')
        self.file_path = file_path
        self.dest_folder = os.path.dirname(file_path)
        self.type_file = (file_path.split('/')[-1])[0:-3].split('__')[
                             0] + stypy_parameters_copy.type_data_file_postfix + ".py"

    def add_type_dict_for_main_context(self, var_dict):
        """
        Add the dictionary of variables for the main context
        :param var_dict: dictionary of name: type
        :return:
        """
        self.__add_type_dict_for_context(var_dict)

    def add_type_dict_for_context(self, var_dict):
        """
        Add the dictionary of variables for a function context. Function name is automatically obtained by traversin
        the call stack. Please note that this function is used in type data autogenerator programs, therefore we can
        obtain this data using this technique
        :param var_dict: dictionary of name: type
        :return:
        """
        import traceback

        func_name = traceback.extract_stack(None, 2)[0][2]

        self.__add_type_dict_for_context(var_dict, func_name)

    def __add_type_dict_for_context(self, var_dict, context="__main__"):
        """
        Helper method for the previous one
        :param var_dict:
        :param context:
        :return:
        """
        if context in self.already_processed_contexts:
            return

        vars_ = filter(lambda var_: "__" not in var_ and not var_ == 'stypy' and not var_ == 'type_test',
                       var_dict.keys())

        self.type_file_txt += "    '" + context + "': {\n"
        for var in vars_:
            self.type_file_txt += "        '" + var + "': " + get_type_name(type(var_dict[var])) + ", \n"

        self.type_file_txt += "    " + "}, \n"

        self.already_processed_contexts.append(context)

    def generate_type_data_file(self):
        """
        Generates the type data file
        :return:
        """
        # print self.dest_folder
        # print self.type_file
        self.type_file_txt += "}\n"
        with open(self.dest_folder + "/" + self.type_file, 'w') as outfile:
            outfile.write(self.type_file_txt)
