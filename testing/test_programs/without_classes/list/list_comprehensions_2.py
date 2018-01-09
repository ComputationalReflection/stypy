modules = [("module1", "desc1"),
          ("module2", "desc2")]

normalized_path = "foo"

user_defined_modules = dict((module_name, module_desc) for (module_name, module_desc) in modules
                            if (normalized_path not in str(module_desc) and "built-in" not in
                                str(module_desc)
                                and module_desc is not None))

