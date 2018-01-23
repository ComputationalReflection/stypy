#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import unittest

from stypy import stypy_main
from stypy.contexts import context
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.reporting.module_line_numbering import ModuleLineNumbering
from stypy.sgmc.sgmc_main import SGMC
from stypy.type_inference_programs.checking.type_inference_checking import check_type_store
from stypy.types import union_type, undefined_type
from testing.testing_parameters import *


def compare_types(class_, class_name):
    comp_result = type(class_).__name__ == class_name
    if comp_result:
        return comp_result

    if class_name == "module":
        return isinstance(class_, context.Context)

    return False


def _compare_types(class_, class_name, allow_undefined):
    if isinstance(class_, union_type.UnionType):
        temp = None
        for t in class_.types:
            if allow_undefined and (isinstance(t, undefined_type.UndefinedType) or t == undefined_type.UndefinedType):
                pass
            else:
                temp = union_type.UnionType.add(temp, t)
    else:
        temp = class_

    return compare_types(temp, class_name)


def instance_of_class_name(class_name, allow_undefined=False):
    return lambda class_: _compare_types(class_, class_name, allow_undefined)


class TestCommon(unittest.TestCase):
    def setUp(self):
        if ROOT_PATH not in sys.path:
            sys.path.append(ROOT_PATH)

        self.file_path = CODE_GENERATION_TESTING_PROGRAMS_PATH + "/test_programs"
        StypyTypeError.reset_error_msgs()
        TypeWarning.reset_warning_msgs()
        ModuleLineNumbering.clear_cache()
        # TypeAnnotationRecord.clear_annotations()
        context.Context.module_parent_contexts = dict()

    def print_output(self, txt, file_=None):
        print (txt)
        if file_ is not None:
            file_.write(str(txt))

    def run_stypy_with_program(self, program_file,
                               verbose=False,
                               generate_type_data_file=False,
                               output_results=False,
                               output_file=None,
                               time_stypy=False,
                               force_type_data_file=True):
        route = SGMC.get_sgmc_route(program_file)
        destination_file = SGMC.sgmc_cache_absolute_path + route
        init = 0
        end = 0
        try:
            if time_stypy:
                init = time.time()

            stypy = stypy_main.Stypy(program_file, PYTHON_EXE, verbose, generate_type_data_file,
                                     type_inference_program_file=destination_file, is_main=True)
            result = stypy.analyze()
            if time_stypy:
                end = time.time()
        except SystemExit as se:
            return -2

        file_ = None
        if output_file is not None:
            file_ = open(output_file, "w")

        if type(result) is StypyTypeError:
            self.print_output("AN ERROR OCURRED WHILE ANALYZING TYPE INFERENCE FILES:\n", file_)
            self.print_output(result, file_)
            if output_file is not None:
                file_.close()

            return -3  # Analysis error

        ti_type_store = stypy.get_analyzed_program_type_store()

        if output_results or verbose:
            self.print_output("\n*************** Type store *************** ", file_)
            self.print_output(ti_type_store, file_)

            self.print_output("\n*************** Errors *************** ", file_)
            errors = stypy.get_analyzed_program_errors()

            if len(errors) > 0:
                self.print_output("{0} errors detected:\n".format(len(errors)), file_)
                err_count = 1
                for error in errors:
                    self.print_output(str(err_count) + ": " + str(error) + "\n", file_)
                    err_count += 1
            else:
                self.print_output("No errors detected.", file_)

            self.print_output("\n*************** Warnings *************** ", file_)
            warnings = stypy.get_analyzed_program_warnings()
            if len(warnings) > 0:
                self.print_output("{0} warnings detected:\n".format(len(warnings)), file_)
                warn_count = 1
                for warning in warnings:
                    self.print_output(str(warn_count) + ": " + str(warning) + "\n", file_)
                    warn_count += 1
            else:
                self.print_output("No warnings detected.", file_)

                self.print_output("\n", file_)

        if output_file is not None:
            file_.close()

        if time_stypy:
            print ("Stypy analyzed the program in " + str(end - init) + " seconds.")
        if ti_type_store is None:
            return None
        else:
            ret = check_type_store(ti_type_store, program_file, verbose, force_type_data_file)
            if not force_type_data_file:
                if len(stypy.get_analyzed_program_errors()) > 0:
                    return -2 # Errors exist in the analysis
                else:
                    return 0  # No error

            return ret
