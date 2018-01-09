from codegen_testing_common import TestCommon


# from stypy.errors.advice import Advice


class TestRealPrograms(TestCommon):
    def test_pystone(self):
        file_path = self.file_path + "/real_programs/pystone.py"
        result = self.run_stypy_with_program(file_path)

        # print "\n\n*************** Coding Advices ***************\n"
        # print "Found {0} advice(s)\n\n".format(len(Advice.get_advice_msgs()))
        # advices = Advice.get_advice_msgs()
        # for advice in advices:
        #      print advice

        self.assertEqual(result, 0)
