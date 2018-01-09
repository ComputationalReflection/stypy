from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.src_code_generators.modules import \
    generate_code_for_module
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.src_code_generators.classes import \
    generate_code_for_class
from stypy.python_lib.python_types.instantiation.known_python_types_management import ExtraTypeDefinitions


def generate_python_modules():
    # generate_code_for_module("__builtins__")
    # generate_code_for_module("sys", excluded_members=["exit", "setrecursionlimit"])
    generate_code_for_module("os",
                             excluded_members=['_exit', 'abort', 'access', 'altsep', 'chdir', 'chmod', 'close', 'closerange',
                                               'defpath', 'devnull', 'dup', 'dup2', 'environ', 'errno',
                                               'error'
                                 , 'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe'
                                 , 'extsep', 'fdopen', 'fstat', 'fsync', 'getcwd', 'getcwdu', 'getenv', 'getpid',
                                               'isatty', 'kill', 'linesep', 'listdir', 'lseek', 'lstat', 'makedirs',
                                               'mkdir',
                                               'open', 'pardir', 'path', 'pathsep', 'pipe', 'popen', 'popen2',
                                               'popen3'
                                 , 'popen4', 'read', 'remove', 'removedirs', 'rename', 'renames', 'rmdir',
                                               'sep', 'spawnl', 'spawnle', 'spawnv', 'spawnve', 'startfile', 'stat',
                                               'stat_float_times', 'stat_result', 'statvfs_result', 'strerror', 'sys',
                                               'system', 'tempnam', 'tmpfile', 'tmpnam', 'umask', 'unlink',
                                               'unsetenv', 'utime', 'waitpid', 'walk', 'write'])
    # generate_code_for_module("operator", maximum_arity=4, excluded_members=["__setslice__", "setslice"])

    # The result of this generation have to be added manually to the type rule files because of performance reasons
    # generate_code_for_module("operator", maximum_arity=4, only_for_members=["__setslice__"])

    # generate_code_for_module("math")
    # generate_code_for_module("ast")


    def generate_python_builtin_iterators():
        generate_code_for_class(ExtraTypeDefinitions.listiterator, "modules/__builtins__", "iterators", "listiterator",
                                type_to_mask_name="ExtraTypeDefinitions.listiterator")

        generate_code_for_class(ExtraTypeDefinitions.tupleiterator, "modules/__builtins__", "iterators",
                                "tupleiterator",
                                type_to_mask_name="ExtraTypeDefinitions.tupleiterator")
        generate_code_for_class(ExtraTypeDefinitions.rangeiterator, "modules/__builtins__", "iterators",
                                "rangeiterator",
                                type_to_mask_name="ExtraTypeDefinitions.rangeiterator")
        generate_code_for_class(ExtraTypeDefinitions.callable_iterator, "modules/__builtins__", "iterators",
                                "callable_iterator",
                                type_to_mask_name="ExtraTypeDefinitions.callable_iterator")
        generate_code_for_class(ExtraTypeDefinitions.listreverseiterator, "modules/__builtins__", "iterators",
                                "listreverseiterator",
                                type_to_mask_name="ExtraTypeDefinitions.listreverseiterator")

        generate_code_for_class(ExtraTypeDefinitions.dict_items, "modules/__builtins__", "iterators", "dict_items",
                                type_to_mask_name="ExtraTypeDefinitions.dict_items")
        generate_code_for_class(ExtraTypeDefinitions.dict_keys, "modules/__builtins__", "iterators", "dict_keys",
                                type_to_mask_name="ExtraTypeDefinitions.dict_keys")
        generate_code_for_class(ExtraTypeDefinitions.dict_values, "modules/__builtins__", "iterators", "dict_values",
                                type_to_mask_name="ExtraTypeDefinitions.dict_values")

        generate_code_for_class(ExtraTypeDefinitions.dictionary_keyiterator, "modules/__builtins__", "iterators",
                                "dictionary_keyiterator",
                                type_to_mask_name="ExtraTypeDefinitions.dictionary_keyiterator")
        generate_code_for_class(ExtraTypeDefinitions.dictionary_itemiterator, "modules/__builtins__", "iterators",
                                "dictionary_itemiterator",
                                type_to_mask_name="ExtraTypeDefinitions.dictionary_itemiterator")
        generate_code_for_class(ExtraTypeDefinitions.dictionary_valueiterator, "modules/__builtins__", "iterators",
                                "dictionary_valueiterator",
                                type_to_mask_name="ExtraTypeDefinitions.dictionary_valueiterator")

        generate_code_for_class(ExtraTypeDefinitions.bytearray_iterator, "modules/__builtins__", "iterators",
                                "bytearray_iterator",
                                type_to_mask_name="ExtraTypeDefinitions.bytearray_iterator")
        generate_code_for_class(ExtraTypeDefinitions.getset_descriptor, "modules/__builtins__", "iterators",
                                "getset_descriptor",
                                type_to_mask_name="ExtraTypeDefinitions.getset_descriptor")
        generate_code_for_class(ExtraTypeDefinitions.member_descriptor, "modules/__builtins__", "iterators",
                                "member_descriptor",
                                type_to_mask_name="ExtraTypeDefinitions.member_descriptor")
        generate_code_for_class(ExtraTypeDefinitions.formatteriterator, "modules/__builtins__", "iterators",
                                "formatteriterator",
                                type_to_mask_name="ExtraTypeDefinitions.formatteriterator")

if __name__ == "__main__":
    generate_python_modules()
    # generate_python_builtin_iterators()
