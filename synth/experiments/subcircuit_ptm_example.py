import numpy as np
from sim.PTM import *
from sim.PTV import *
from sim.circs.circs import *
from sim.sim import *
from synth.experiments.example_circuits_for_proposal import *

def sub_circuit_PTM_example(): #TODO: Make this a unit test
    #1st case
    c = Example_Circ_MAC()
    input_data = np.random.uniform(size=(1000, 8))
    Cin = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ])

    result = sim_circ_PTM(c, Dataset(input_data), Cin, validate=True, lsb='left')

    #2nd case
    c2 = TWO_ANDs()
    Cin = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    result_top = sim_circ_PTM(c2, Dataset(input_data[:, :4]), Cin, validate=True, lsb='left')
    result_bot = sim_circ_PTM(c2, Dataset(input_data[:, 4:]), Cin, validate=True, lsb='left')

    new_dataset = np.hstack((result_top.out, result_bot.out))

    Cin = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    c3 = TWO_MUXs()
    result_overall = sim_circ_PTM(c3, Dataset(new_dataset), Cin, validate=True, lsb='left')

    err_total = 0
    for i, out in enumerate(result_overall.out):
        err_total += MSE(out, result.correct[i])
    print(np.sqrt(err_total / len(result_overall.correct)))