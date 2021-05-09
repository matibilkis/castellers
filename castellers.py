import numpy as np

class QuantumCircuit:
    def __init__(self, n_qubits=1):
        self.n_qubits = n_qubits

    def rz(self,th):
        """
        Comment: we add some phase here (equivalent to yet a second rotation around z-axis of value -th, i.e. Exp[- \ii \theta \Sigma_z /2]), such that Rz(phi)|0> = |0>
        """
        return np.array([[1,0],[0,np.exp(-th*1j)]])
        ##without the phase substraction, MatrixExp(pauliZ) is.
        #return np.array([[np.exp(th*0.5j),0],[0,np.exp(-th*0.5j)]])


    def ry(self,th):
        return np.array([[np.cos(th/2),-np.sin(th/2)],[np.sin(th/2),np.cos(th/2)]])

    @property
    def H(self):
        return np.array([[1,1],[1,-1]])/np.sqrt(2)

    @property
    def S(self):
        return np.array([[1,0],[0,1j]])

    def unitary(self,list_of_gates):
        """
        input: list of gates as ussually drawn in the quantum circuit
        output: unitary representing the circuit (to be applied to fiducial state, say |0>^N)
        """
        unitary = list_of_gates[-1]
        #gather the gates into single matrix
        for it, k in enumerate(list_of_gates[::-1][1:]):
            unitary = np.matmul(unitary,k)
        return unitary

    def output_state(self,unitary):
        """
        Given a unitary returns its action on a fiducial state (assumed to be |0>^n)
        returns
        """
        return unitary[:,0]


    def get_probability(self,projector,output_state,shots=np.inf):
        """
        very trivial variation, depending on the number of shots (measurements) that you perform.

        Warning: this only works for 1 qubit, since otherwise you should get a multinomial distribution and compute all the probabilities so to simulate, and we will do that using already-written libraries.
        """
        if shots == np.inf:
            return np.abs(np.matmul(projector,output_state))**2
        else:
            #just check things are ok
            assert shots>0
            assert isinstance(shots,int)
            assert self.n_qubits == 1

            #compute the probability by repeating many times the experiment, with the very same state, and counting how many times the prepared state is measured on |pr>. For 1-qubit this is equivalent to sample from a bernoulli distribution with mean value = "probability to find the state in |pr>" (we assume that if you get a 1 in the Bernoulli experiment, the state was fonud in |pr> and if you get a 0, the state was fonud in id - |pr><pr|).

            pr_binary = np.abs(np.matmul(projector,output_state))**2
            estimated_probability = np.random.binomial(shots, pr_binary)/shots
            return estimated_probability

    def observable_mean(self, output_state,  operator="x", shots=np.inf):
        """
        this returns the mean value of a pauli observable
        computed by measuring on the computational basis the output_state.

        Warning: it quickly becomes complicated to put all the transformations needed (see https://docs.microsoft.com/en-us/azure/quantum/concepts-pauli-measurements), so i'll do only 1 qubit.
        """
        projectors = np.eye(2)
        if operator.lower() == "x":
            output_state = np.matmul(self.H, output_state)
        elif operator.lower() == "y":
            output_state = np.matmul(np.conjugate(self.S), output_state)
            output_state = np.matmul(self.H, output_state)
        avg=0
        for pr, eigen in zip(projectors, [1,-1]):
            avg+= eigen*self.get_probability(pr, output_state, shots=shots)
        return avg
