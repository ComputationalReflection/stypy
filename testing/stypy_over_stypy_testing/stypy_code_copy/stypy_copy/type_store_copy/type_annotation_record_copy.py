class TypeAnnotationRecord:
    """
    Class to annotate the types of variables defined in a Python source file. This is used as a record for generating
     Python type annotated programs
    """
    annotations_per_file = dict()

    @staticmethod
    def is_type_changing_method(method_name):
        """
        Determines if this method name is bound to a Python method that we know it changes the state of the classes
        that define it, to enable type annotations within these methods.
        :param method_name: str
        :return: bool
        """
        return method_name in ["__setitem__",
                               "__add__",
                               "__setslice__",
                               "add",
                               "append"]

    # TODO: Remove?
    # def __init__(self, function_name):
    #     self.function_name = function_name
    #     self.annotation_dict = dict()

    # def annotate_type(self, line, col_offset, var_name, type_):
    #     if line not in self.annotation_dict.keys():
    #         self.annotation_dict[line] = list()
    #
    #     anottation_tuple = (var_name, type_, col_offset)
    #     if not anottation_tuple in self.annotation_dict[line]:
    #         self.annotation_dict[line].append(anottation_tuple)
    #
    # def get_annotations_for_line(self, line):
    #     try:
    #         return self.annotation_dict[line]
    #     except:
    #         return None
    #
    # def clone(self):
    #     clone = TypeAnnotationRecord(self.function_name)
    #     for (key, value) in self.annotation_dict.items():
    #         clone.annotation_dict[key] = value
    #
    #     return clone

    @staticmethod
    def get_instance_for_file(file_name):
        """
        Get an instance of this class for the specified file_name. As there can be only one type annotator per file,
        this is needed to reuse existing type annotators.
        :param file_name: str
        :return: TypeAnnotationRecord object
        """
        if file_name not in TypeAnnotationRecord.annotations_per_file.keys():
            TypeAnnotationRecord.annotations_per_file[file_name] = TypeAnnotationRecord()

        return TypeAnnotationRecord.annotations_per_file[file_name]

    def __init__(self):
        """
        Creates a TypeAnnotationRecord object
        :return:
        """
        self.annotation_dict = dict()

    def annotate_type(self, line, col_offset, var_name, type_):
        """
        Annotates a variable type information, including its position
        :param line: Source line
        :param col_offset: Column inside the source line
        :param var_name: Variable name
        :param type_: Variable type
        :return:
        """
        if line not in self.annotation_dict.keys():
            self.annotation_dict[line] = list()

        annotation_tuple = (var_name, type_, col_offset)
        if annotation_tuple not in self.annotation_dict[line]:
            self.annotation_dict[line].append(annotation_tuple)

    def get_annotations_for_line(self, line):
        """
        Get all annotations registered for a certain line
        :param line: Line number
        :return: Annotation list
        """
        try:
            return self.annotation_dict[line]
        except:
            return None

    def reset(self):
        """
        Remove all type annotations
        :return:
        """
        self.annotation_dict = dict()

    @staticmethod
    def clear_annotations():
        """
        Remove all type annotations for all files
        :return:
        """
        for a in TypeAnnotationRecord.annotations_per_file.values():
            a.reset()
