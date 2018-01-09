import subprocess
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk("."):
    f.extend(filenames)

pys = filter(lambda fil: fil.endswith(".py") and fil.startswith("numpy"), f)
skel = """
import types
from stypy.errors.type_error import StypyTypeError
from stypy.types.undefined_type import UndefinedType
from stypy.types import union_type
from testing.code_generation_testing.codegen_testing_common import instance_of_class_name
import numpy

test_types = {{
    '__main__': {{
{0}
    }},
}}
"""

for py_file in pys:
    cmd = ['c:/Python27/python.exe', py_file]
    print py_file
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    out, err = p.communicate()
    if out != "":
        out_file = py_file.replace(".py", "") + "__type_data.py"
        out_f = open("./stypy_test_files/" + out_file, "w")
        out_lines = out.split("\n")
        out_txt = ""
        for line in out_lines:
            out_txt += "\t\t" + line + "\n"

        txt = skel.format(out_txt)
        out_f.write(txt)
        out_f.close()
