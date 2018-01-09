import os

filenames = [f for f in os.listdir('.') if os.path.isfile(f)]

pys = filter(lambda fil: fil.endswith(".py") and fil.startswith("numpy"), filenames)

other_skel = """
    def test_{0}(self):
        file_path = self.file_path + "/numpy/basic_numpy/functions/{1}"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)
"""

for py_file in pys:
    print (other_skel.format(py_file.replace(".py", ""), py_file))
