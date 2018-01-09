#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def write_python_source_code(source_file_path, src):
    """
    Writes Python source code to the provided file
    :param source_file_path: Destination .py file
    :param src: Source code
    :return:
    """
    dirname = os.path.dirname(source_file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(source_file_path, 'w') as outfile:
        outfile.write(src)
        outfile.close()
