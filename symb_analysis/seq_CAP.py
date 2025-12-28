from typing import Any
import sympy as sp
import numpy as np
import itertools
from sim.PTV import get_Q
from sim.PTM import TT_to_ptm

def get_steady_state_nullspace(T):
    ns = (T.T - sp.Matrix.eye(T.rows)).nullspace()
    return sp.simplify(ns[0] / sum(ns[0]))

def get_steady_state_linear_system(T):
    #Version of get_steady_state that uses a linear system of equations to solve for the steady state
    A = (T.T - sp.Matrix.eye(T.rows))
    pi = sp.symbols([f"pi{idx}" for idx in range(T.rows)])
    eqs = list(A * sp.Matrix(pi)) + [sum(pi) - 1]
    sol = sp.solve(eqs, pi)
    return sp.Matrix(list(sol.values()))

def get_steady_state(T, vars=None):
    eigs = (T.T).eigenvects()

    #Find the eigenvector with eigenvalue closest to 1
    eigvals = np.array([x[0] for x in eigs])
    if vars is not None:
        eigval_idx = None
        for i in range(len(eigs)):
            if eigvals[i].subs(sum(vars), 1) == 1:
                eigval_idx = i
                break
        if eigval_idx is None:
            raise ValueError("No eigenvector with eigenvalue closest to 1 found")
        eigvec_indx = eigval_idx
    else:
        eigvec_indx = np.argmin(np.abs(eigvals - 1))
    eigvec = eigs[eigvec_indx][2][0]

    #Normalize the eigenvector
    eigvec = eigvec / np.sum(eigvec)
    return sp.simplify(eigvec)

def get_dv_from_rho_single(rho, use_new_symbol_for_max=False, symbolic_pos=False, symbol=sp.symbols("x")):
    #symbolic solution for the DV for a single variable in terms of the autocorrelation
    x = symbol

    if symbolic_pos:
        pxx = x ** 2 * (1 - rho) + rho * x
    else:
        if rho >= 0:
            pxx = x ** 2 * (1 - rho) + rho * x 
        else:
            if use_new_symbol_for_max:
                max_symbol = sp.symbols("m")
                pxx = x ** 2 * (1 + rho) - rho * max_symbol
            else:
                pxx = x ** 2 * (1 + rho) - rho * sp.Max(2*x-1, 0)
    p = sp.Matrix([
        1,
        x, 
        x,
        pxx
    ])
    Q = get_Q(2, lsb='right')
    Qinv = sp.Matrix(np.linalg.inv(Q))
    return sp.nsimplify(Qinv @ p)

def lfsr_dv_model(n, symbol=sp.symbols("x")):
    if n != 1:
        raise ValueError("LFSR DV model only supports n=1 for now")

    x = symbol
    return sp.Matrix([
        sp.Max(1-1.5*x, 0.5-0.5*x),
        sp.Min(0.5*x, 0.5-0.5*x),
        sp.Min(0.5*x, 0.5-0.5*x),
        sp.Max(1.5*x - 0.5, 0.5*x)
    ])

def transition_matrix_to_FSM(num_states, T):
    transitions = []
    for row in range(num_states):
        for col in range(num_states):
            if T[row, col] != 0:
                transitions.append((row, col, T[row, col]))
    return transitions

def FSM_to_transition_matrix(num_states, transitions, vars=None, time_steps=0):
    #Given a specification of an FSM, return its transition matrix
    #(0, -1, x(1-y))
    T = sp.zeros(num_states, num_states)
    for transition in transitions:
        T[transition[0], transition[1]] = transition[2]

    #assert that the rows of T sum to 1
    if vars is not None:
        symbols = get_DV_symbols(vars, time_steps)
        #vs = sp.symbols([f"v{idx}" for idx in range(len(vars))])            
        for row in range(num_states):
            #for idx, v in enumerate(symbols):
                #print(v, vs[idx])
                #T[row, :] = T[row, :].subs(v, vs[idx])
            assert sp.simplify(sum(T[row, :]).subs(sum(symbols), 1)) == 1
    return T

class JointProb(sp.Expr):
    """Custom operator that distributes joint probability over addition"""
    
    def __new__(cls, left, right):
        # If both operands are Add expressions, distribute
        if isinstance(left, sp.Add) and isinstance(right, sp.Add):
            terms = []
            for term_left in left.args:
                for term_right in right.args:
                    terms.append(JointProb(term_left, term_right))
            return sp.Add(*terms)
        # If only left is Add, distribute over left
        elif isinstance(left, sp.Add):
            terms = []
            for term in left.args:
                terms.append(JointProb(term, right))
            return sp.Add(*terms)
        # If only right is Add, distribute over right
        elif isinstance(right, sp.Add):
            terms = []
            for term in right.args:
                terms.append(JointProb(left, term))
            return sp.Add(*terms)
        # If neither is Add, create the operator
        else:
            #obj = sp.Expr.__new__(cls)
            #obj._args = (left, right)
            return sp.symbols(f"{left}^{right}")
    
    @property
    def left(self):
        return self._args[0]
    
    @property
    def right(self):
        return self._args[1]
    
    def _latex(self, printer):
        return f"({printer._print(self.left)})^{{{printer._print(self.right)}}}"
    
    def _sympystr(self, printer):
        return f"{self.left}^{self.right}"

def get_DV_symbols(vars, time_steps):
    #Return a list of the symbols for the DV probabilities for a given number of time steps

    if len(vars) == 1:
        undelayed_vars = sp.symbols(f"{vars[0]}b {vars[0]}")
    else:
        undelayed_vars = []
        for var in vars:
            if len(undelayed_vars) == 0:
                undelayed_vars = [f"{var}b",var]
            else:
                undelayed_vars = list(itertools.product(undelayed_vars, [f"{var}b",var]))
        undelayed_vars = [sp.symbols(f"{var[0]}&{var[1]}") for var in undelayed_vars]

    if time_steps == 0:
        return undelayed_vars
    elif time_steps == 1:
        delayed_vars = list(itertools.product(undelayed_vars, undelayed_vars))
        return [sp.symbols(f"{var[0]}^{var[1]}") for var in delayed_vars]
    else:
        raise ValueError("Only time steps of 0 and 1 are supported")

def extend_markov_chain_t1(transitions, vars, dv=None, for_printing=False, return_state_mapping=False):
    #Extend a markov chain representing 0 time steps of history to one representing 1 time step of history

    symbols = get_DV_symbols(vars, 0)
    varsum = sum(symbols) #varsum should always equal 1

    new_transitions = []
    state_mapping = {}
    for idx_a, transition_a in enumerate(transitions):
        for idx_b, transition_b in enumerate(transitions):
            if transition_a[1] == transition_b[0]:
                #print(idx_b, transition_b[1], transition_b[2])
                state_mapping[idx_b] = transition_b[1]
                if for_printing:
                    new_transitions.append((idx_a, idx_b, f"{transition_b[2]}|{transition_a[2]}"))
                else:
                    #Apply law of conditional probability to the transition probability
                    denom = JointProb(varsum, transition_a[2])
                    trans_prob = JointProb(transition_b[2], transition_a[2]) / denom
                    new_transitions.append((idx_a, idx_b, sp.simplify(trans_prob)))

    if for_printing:
        print(state_mapping)
        return new_transitions

    #return results in terms of DV
    dv_symbols = get_DV_symbols(vars, 1)
    new_transitions_dv = []
    for transition in new_transitions:
        expr = transition[2]
        for idx, v in enumerate(dv_symbols):
            expr = expr.subs(v, sp.symbols(f"v{idx}"))
            #substitute probability expressions for the DV symbols
            #This is necessary to reduce the number of symbols in the transition matrix
            if dv is not None:
                expr = expr.subs(sp.symbols(f"v{idx}"), dv[idx])
        new_transitions_dv.append((transition[0], transition[1], expr))
    
    #State mapping is which states in the extended FSM correspond to which states in the original FSM
    if return_state_mapping:
        return new_transitions_dv, state_mapping

    return new_transitions_dv

def symbolic_minterm_eval(expr, pattern):
    #Assuming expression is encoded with xb&y&zb etc..., evaluate its truth value for a given pattern
    result = 1
    for idx, var in enumerate(str(expr).split('&')):
        if var[-1] == 'b':
            result = result * (1 - pattern[idx])
        else:
            result = result * pattern[idx]
    return result

def symbolic_SOP_eval(expr, pattern):
    for subex in sp.Add.make_args(expr):
        if symbolic_minterm_eval(subex, pattern):
            return 1
    return 0

def get_extended_mealy_ptm(pi, original_transitions, extended_transitions, state_mapping, vars, mealy_TTs):
    n = len(vars)
    k = mealy_TTs.shape[1]

    #For each state in the extended FSM, get the Mealy transition table
    overall_PTM = sp.zeros(2 ** (2*n), 2**(2*k))
    for dest in state_mapping.keys():
        #filter the transitions to only include those that correspond to the current state
        filtered_transitions = [transition for transition in extended_transitions if transition[1] == dest]

        #get the nodes that point to this dest
        #Get the representative non-extended node for these. They should be the same
        prior_states = []
        rep_states = []
        for node in filtered_transitions:
            prior_states.append(node[0])
            rep_states.append(state_mapping[node[0]])

        assert all(item == rep_states[0] for item in rep_states)
        src_rep_state = rep_states[0]

        #Build the Mealy TT for this state
        #For the current cycle, it's the same as the non-extended Mealy TT
        #For the prior cycle, it uses the Mealy output of the representative state
        #Note that some of these transitions will be invalid
        dest_rep_state = state_mapping[dest]
        current_TT = mealy_TTs[:, :, dest_rep_state]
        prior_TT = mealy_TTs[:, :, src_rep_state].copy()

        #We need to evaluate the prior_TT using the input pattern that led to the dest state

        #first get the relevant state transition function (the one for the arrows pointing at the dest state)
        trans_func = [transition for transition in original_transitions \
            if transition[0] == src_rep_state and transition[1] == dest_rep_state][0][2]
        
        for idx, pattern in enumerate(itertools.product([0, 1], repeat=len(vars))):
            if symbolic_SOP_eval(trans_func, pattern) == 0:
                prior_TT[idx] = -1 #invalid (this means that this particular pattern could not have led to dest) 

        full_TT = np.zeros((2**(2*n), 2*k))
        for current_idx in range(2**n):
            for prior_idx in range(2**n):
                full_TT[current_idx * 2 ** n + prior_idx, :k] = current_TT[current_idx, :]
                full_TT[current_idx * 2 ** n + prior_idx, k:] = prior_TT[prior_idx, :]

        #full_TT = np.array(list(itertools.product(current_TT, prior_TT)))[:, :, 0]
        state_PTM = TT_to_ptm(full_TT, 2*n, 2*k, allow_invalids=True)
        state_PTM = sp.Matrix(state_PTM * 1) * pi[dest]
        overall_PTM += state_PTM

    #divide each row by its row sum
    for i in range(overall_PTM.rows):
        overall_PTM[i, :] = overall_PTM[i, :] / sum(overall_PTM[i, :])
    return overall_PTM

def numeric_seq_CAP(circ, dv_numeric):
    transitions = circ.get_transition_list()
    vars = circ.get_vars()
    extended_transitions, state_mapping = extend_markov_chain_t1(transitions, vars, dv=dv_numeric, return_state_mapping=True)
    T = FSM_to_transition_matrix(max(state_mapping.keys()) + 1, extended_transitions, vars=vars, time_steps=1)
    pi = get_steady_state_nullspace(T)
    mealy_TTs = circ.get_mealy_TTs()
    extended_ptm = get_extended_mealy_ptm(pi, transitions, extended_transitions, state_mapping, vars, mealy_TTs)
    return extended_ptm.T @ dv_numeric