#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

from stypy import stypy_parameters
from stypy.errors.type_error import StypyTypeError


def normalize(path_):
    """
    Normalizes a file path
    :param path_:
    :return:
    """
    return path_.replace("\\", "/")


"""
Do not check file update dates to determine if a type inference program will be regenerated: Regenerate it always.
"""
always_regenerate_files = False


class SGMC(object):
    """
    The Stypy Global Module Cache (SGMC) is the stypy module responsible of handling type inference program creation
    and update when the original source changes. By creating type inference programs and storing them in this cache we
    save a lot of time when analyzing programs, as only those sources that have been changed will be processed again.
    SGMC also compiles generated files, so Python will load .pyc files when available. File creation follows the
    module path of the original files, so import sentences have to be just redirected to its corresponding type
    inference file. For example, if we generate a type inference file for the module test.program, the type inference
    program can be loaded by importing sgmc.sgmc_cache.test.program.
    """
    normalized_path = map(lambda elem: normalize(elem), sys.path)
    sgmc_cache_absolute_path = stypy_parameters.ROOT_PATH + "/sgmc/sgmc_cache"

    # Avoids generating the same file over and over in the same run
    already_generated_files_in_this_run = []

    @staticmethod
    def init():
        """
        Initializes de cache. Just configures de path to allow importing modules from the SGMC
        :return:
        """
        if SGMC.sgmc_cache_absolute_path not in sys.path:
            sys.path = [SGMC.sgmc_cache_absolute_path] + sys.path
        if SGMC.sgmc_cache_absolute_path + "/site-packages" not in sys.path:
            sys.path = [SGMC.sgmc_cache_absolute_path + "/site-packages"] + sys.path

    @staticmethod
    def get_sgmc_route(file_name):
        """
        Obtains the corresponding route in the SGMC of this file name
        :param file_name:
        :return:
        """
        SGMC.init()
        file_name = normalize(file_name)
        file_name = file_name.replace("-", "_")
        for path_ in SGMC.normalized_path:
            if path_ in file_name:
                route = file_name.replace(path_, "")
                return route
        return None

    @staticmethod
    def __get_sgmc_routes_for_module(localization, module_path):
        """
        When we import a module from a destination directory, this module may have submodules that may be also loaded
        when loading the parent one. As the SGMC do not parse Python files, it has no knowledge about what files are
        going to be needed when loading the parent one. We adopted the policy to generate all the submodules of a
        certain one when the parent is loaded to avoid trying to load a type inference file whose generation has not
        been triggered. This also avoids problems when certain import scenarios where circular imports may be present.
        The method therefore return all the routes (paths) and files that may need to be generated once a module is
        loaded.
        :param localization:
        :param module_path:
        :return:
        """
        module_path = normalize(module_path)
        dirtree = os.walk(module_path)

        only_py_files = map(lambda tuple_: (
            normalize(tuple_[0]),
            filter(lambda file_name: file_name.endswith(".py"), tuple_[2])), dirtree)

        if len(only_py_files) == 0:
            return StypyTypeError(localization,
                                  "The route {0} is not a Python module: No Python source code files are present".
                                  format(module_path))

        only_modules = filter(lambda tuple_: "__init__.py" in tuple_[1] and not (
            stypy_parameters.type_inference_file_directory_name in tuple_[0]),
                              only_py_files)
        if len(only_modules) == 0:
            return StypyTypeError(localization,
                                  "The route {0} do not contain Python modules: missing __init__.py files".format(
                                      module_path
                                  ))

        is_a_module = len(filter(lambda tuple_: tuple_[0] == module_path, only_modules)) > 0
        if not is_a_module:
            return StypyTypeError(localization, "The route {0} is not a Python module: missing __init__.py file".format(
                module_path
            ))

        total_files = reduce(lambda list_, tuple_:
                             list_ + map(lambda file_name: tuple_[0] + "/" + file_name, tuple_[1]),
                             only_modules, [])

        # Also load parent modules __init__ files
        parent_module_path = os.path.dirname(module_path)
        while not (parent_module_path == ""):
            module_init = parent_module_path + "/__init__.py"
            if os.path.isfile(module_init):
                total_files = [module_init] + total_files
            else:
                break
            parent_module_path = os.path.dirname(parent_module_path)

        sgmc_routes = map(lambda file_name: SGMC.sgmc_cache_absolute_path + SGMC.get_sgmc_route(file_name), total_files)

        return total_files, sgmc_routes

    @staticmethod
    def import_module(localization, module_path):
        """
        Obtains the files to be generated when a module is imported from the provided path
        :param localization:
        :param module_path:
        :return:
        """
        original_files, sgmc_routes = SGMC.__get_sgmc_routes_for_module(localization, module_path)
        files_to_generate = map(lambda element: (element[0], element[1]),
                                filter(lambda tuple_: SGMC.__file_needs_to_be_updated(tuple_[0], tuple_[1]),
                                       zip(original_files, sgmc_routes)))

        return files_to_generate

    @staticmethod
    def __file_needs_to_be_updated(original_file, stypy_file):
        """
        Determines if a file needs to be updated checking its creation time.
        :param original_file:
        :param stypy_file:
        :return:
        """
        if always_regenerate_files:
            # If we already generated a file there is no point in doing it again.
            if stypy_file not in SGMC.already_generated_files_in_this_run:
                SGMC.already_generated_files_in_this_run.append(stypy_file)
                return True
            else:
                return False
        try:
            original_stats = os.stat(original_file)
            if not os.path.isfile(stypy_file):
                return True

            stypy_file_stats = os.stat(stypy_file)

            return not str(original_stats.st_mtime) == str(stypy_file_stats.st_mtime)  # Same modification time
        except:
            # If an error occurs, regenerate file.
            return True

    @staticmethod
    def change_file_modification_time(original_file, stypy_file):
        """
        Alters a file update time. This is used to put a type inference file to the same value as the source code one.
        :param original_file:
        :param stypy_file:
        :return:
        """
        original_stats = os.stat(original_file)
        os.utime(stypy_file, (original_stats.st_atime, original_stats.st_mtime))

    @staticmethod
    def is_sgmc_path(module_name):
        """
        Determines if a module name belongs to a SGMC-stored module
        :param module_name:
        :return:
        """
        return SGMC.sgmc_cache_absolute_path in sys.modules[module_name].__file__.replace("\\", "/")

    @staticmethod
    def get_sgmc_full_module_name(file_name):
        """
        Gets the SGMC module name corresponding to the passed file name
        :param file_name:
        :return:
        """
        temp_route = SGMC.get_sgmc_route(file_name)

        if temp_route.endswith("/__init__.pyc"):
            temp_route = temp_route[:-len("/__init__.pyc")]
        if temp_route.endswith("/__init__.py"):
            temp_route = temp_route[:-len("/__init__.py")]
        if temp_route.endswith(".pyc"):
            temp_route = temp_route[:-len(".pyc")]
        if temp_route.endswith(".py"):
            temp_route = temp_route[:-len(".py")]
        return temp_route.replace("/", ".")

    @staticmethod
    def get_parent_module_name(module_name, level):
        """
        Gets a parent module of a certain one, level modules below.
        :param module_name:
        :param level:
        :return:
        """
        splitter = module_name.split('.')
        parent_name = ""
        for i in range(len(splitter) - level):
            parent_name += splitter[i] + "."

        return parent_name[:-1]

    @staticmethod
    def get_original_module_name(module_name):
        """
        Gets the source code module corresponding to a SGMC module name
        :param module_name:
        :return:
        """
        return module_name.replace("stypy.sgmc.sgmc_cache.", "").replace("site_packages.", "")
