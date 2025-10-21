import sympy as sp
import numpy as np
import itertools

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

def transition_matrix_to_FSM(num_states, T):
    transitions = []
    for row in range(num_states):
        for col in range(num_states):
            if T[row, col] != 0:
                transitions.append((row, col, T[row, col]))
    return transitions

def FSM_to_transition_matrix(num_states, transitions, vars=None):
    #Given a specification of an FSM, return its transition matrix
    #(0, -1, x(1-y))
    T = sp.zeros(num_states, num_states)
    for transition in transitions:
        T[transition[0], transition[1]] = transition[2]

    #assert that the rows of T sum to 1
    if vars is not None:
        #vs = sp.symbols([f"v{idx}" for idx in range(len(vars))])            
        for row in range(num_states):
            #for idx, v in enumerate(vars):
            #    #print(v, vs[idx])
            #    T[row, :] = T[row, :].subs(v, vs[idx])
            assert sp.simplify(sum(T[row, :]).subs(sum(vars), 1)) == 1
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
    undelayed_vars = []
    for var in vars:
        if len(undelayed_vars) == 0:
            undelayed_vars = [var, f"{var}b"]
        else:
            undelayed_vars = list(itertools.product(undelayed_vars, [var, f"{var}b"]))
    undelayed_vars = [sp.symbols(f"{var[0]}{var[1]}") for var in undelayed_vars]

    if time_steps == 0:
        return undelayed_vars
    elif time_steps == 1:
        delayed_vars = list(itertools.product(undelayed_vars, undelayed_vars))
        return [sp.symbols(f"{var[0]}^{var[1]}") for var in delayed_vars]
    else:
        raise ValueError("Only time steps of 0 and 1 are supported")

def extend_markov_chain_t1(transitions, vars, for_printing=False):
    #Extend a markov chain representing 0 time steps of history to one representing 1 time step of history

    varsum = sum(vars) #varsum should always equal 1

    new_transitions = []
    for idx_a, transition_a in enumerate(transitions):
        for idx_b, transition_b in enumerate(transitions):
            if transition_a[1] == transition_b[0]:
                if for_printing:
                    new_transitions.append((idx_a, idx_b, f"{transition_b[2]}|{transition_a[2]}"))
                else:
                    #Apply law of conditional probability to the transition probability
                    denom = JointProb(varsum, transition_a[2])
                    trans_prob = JointProb(transition_b[2], transition_a[2]) / denom
                    new_transitions.append((idx_a, idx_b, sp.simplify(trans_prob)))

    return new_transitions