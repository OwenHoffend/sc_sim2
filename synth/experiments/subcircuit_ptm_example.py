import numpy as np
from sim.PTM import *
from sim.PTV import *
from sim.circs.circs import *
from sim.sim import *
from synth.experiments.example_circuits_for_proposal import *

def sub_circuit_PTM_example():
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

    result = sim_circ_PTM(c, Dataset(input_data), Cin, validate=True)
    scc = np.mean(result.scc_array(0, 1))
    print("First case RMSE: ", result.RMSE())
    print("First case scc: ", scc)

    #2nd case
    c2 = TWO_ANDs()
    Cin = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    result_top = sim_circ_PTM(c2, Dataset(input_data[:, :4]), Cin, validate=True)
    result_bot = sim_circ_PTM(c2, Dataset(input_data[:, 4:]), Cin, validate=True)

    scc_top = np.mean(result_top.scc_array(0, 1))
    scc_bot = np.mean(result_bot.scc_array(0, 1))
    print("Second case intermediate scc: ", scc_top, scc_bot)

    new_dataset = np.hstack((result_top.out, result_bot.out))

    Cin = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    c3 = TWO_MUXs()
    result_overall = sim_circ_PTM(c3, Dataset(new_dataset), Cin, validate=True)
    scc_overall = np.mean(result_overall.scc_array(0, 1))
    print("Second case RMSE: ", result_overall.RMSE(other_correct=result.correct))
    print("Second case overall scc: ", scc_overall)

    #Third case
    c_and = C_AND_N(2)
    c_or = C_OR_N(2)
    c_and_not = AND_WITH_NOT_CONST()
    c_and_const = AND_WITH_CONST()

    #Layer 1 AND gates
    Cin_l1 = np.identity(2)
    a1 = sim_circ_PTM(c_and, Dataset(input_data[:, :2]), Cin_l1, validate=True)
    a2 = sim_circ_PTM(c_and, Dataset(input_data[:, 2:4]), Cin_l1, validate=True)
    a3 = sim_circ_PTM(c_and, Dataset(input_data[:, 4:6]), Cin_l1, validate=True)
    a4 = sim_circ_PTM(c_and, Dataset(input_data[:, 6:8]), Cin_l1, validate=True)

    #Layer 2 AND gates
    Cin_l2 = np.identity(1)
    a5 = sim_circ_PTM(c_and_not, Dataset(a1.out), Cin_l2, validate=True)
    a6 = sim_circ_PTM(c_and_const, Dataset(a2.out), Cin_l2, validate=True)
    a7 = sim_circ_PTM(c_and_not, Dataset(a3.out), Cin_l2, validate=True)
    a8 = sim_circ_PTM(c_and_const, Dataset(a4.out), Cin_l2, validate=True)

    #Layer 3 OR gates
    Cin_l3 = np.identity(2)
    z1 = sim_circ_PTM(c_or, Dataset(np.hstack((a5.out, a6.out))), Cin_l3, validate=True)
    z2 = sim_circ_PTM(c_or, Dataset(np.hstack((a7.out, a8.out))), Cin_l3, validate=True)
    print("Third case Z1 RMSE: ", z1.RMSE(other_correct=result.correct[:, 0]))
    print("Third case Z2 RMSE: ", z2.RMSE(other_correct=result.correct[:, 1]))

    #Fourth case
    c2 = TWO_ANDs()
    Cin = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    result_top = sim_circ_PTM(c2, 
        Dataset(np.vstack((input_data[:, 0], input_data[:, 1], input_data[:, 4], input_data[:, 5])).T), Cin, validate=True)
    result_bot = sim_circ_PTM(c2, 
        Dataset(np.vstack((input_data[:, 2], input_data[:, 3], input_data[:, 6], input_data[:, 7])).T), Cin, validate=True)

    scc_top = np.mean(result_top.scc_array(0, 1))
    scc_bot = np.mean(result_bot.scc_array(0, 1))
    print("Fourth case intermediate scc: ", scc_top, scc_bot)

    new_dataset = np.vstack((result_top.out[:, 0], result_bot.out[:, 0], result_top.out[:, 1], result_bot.out[:, 1])).T

    Cin = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    c3 = TWO_MUXs()
    result_overall = sim_circ_PTM(c3, Dataset(new_dataset), Cin, validate=True)
    scc_overall = np.mean(result_overall.scc_array(0, 1))
    print("Fourth case RMSE: ", result_overall.RMSE(other_correct=result.correct))
    print("Fourth case overall scc: ", scc_overall)