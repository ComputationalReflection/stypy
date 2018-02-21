#!/usr/bin/env python
# -*- coding: utf-8 -*-
from testing.code_generation_testing.codegen_testing_common import TestCommon


class TestShedSkin(TestCommon):
    def test_adatron(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/adatron.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_ac_encode(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/ac_encode.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_amaze(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/amaze.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_ant(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/ant.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_bh(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/bh.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_block(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/block.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_brainfuck(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/brainfuck.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_chaos(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/chaos.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_chess(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/chess.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_dijkstra(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/dijkstra.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_dijkstra2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/dijkstra2.py"
        result = self.run_stypy_with_program(file_path, output_results=True)

        self.assertEqual(result, 0)

    def test_genetic(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/genetic.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_genetic2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/genetic2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_go(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/go.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_kanoodle(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/kanoodle.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_kmeanspp(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/kmeanspp.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_life(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/life.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_linalg(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/linalg.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_loop(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/loop.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_LZ2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/LZ2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mandelbrot(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mandelbrot.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mandelbrot2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mandelbrot2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mao(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mao.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mastermind_main(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mastermind_main.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_neural2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/neural2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_oliva2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/oliva2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_othello(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/othello.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_pisang(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/pisang.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_pygmy(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/pygmy.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_rgbConverter(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/rgbConverter.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_richards(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/richards.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_rubik2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/rubik2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sokoban(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sokoban.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sudoku1(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sudoku1.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sudoku4(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sudoku4.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_voronoi(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/voronoi.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mwmatching(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mwmatching.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False, expected_errors=2)

        self.assertEqual(result, 0)

    def test_rsync(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/rsync.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_score4(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/score4.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_shaImplementation(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/shaImplementation.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sieve(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sieve.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_solitaire(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/solitaire.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sudoku2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sudoku2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sudoku3(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sudoku3.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_sudoku5(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/sudoku5.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_tictactoe(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/tictactoe.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_TonyJpegDecoder(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/TonyJPegDecoder.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_yopyra(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/yopyra.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)


    # Programs to look closely
    def test_path_tracing(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/path_tracing.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_plcfrs(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/plcfrs.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)



    #Hangs
    def test_minilight_main(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/minilight_main.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_mastermind2(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/mastermind2.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    def test_rubik(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/rubik.py"
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)

    # Recursion problem when generating code
    def test_hq2x(self):
        file_path = self.file_path + "/benchmark_suite/shedskin/hq2x.py"
        import sys
        sys.setrecursionlimit(1500)
        result = self.run_stypy_with_program(file_path, output_results=True, force_type_data_file=False)

        self.assertEqual(result, 0)