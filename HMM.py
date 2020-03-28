
########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

# import nltk
# nltk.download('cmudict')
# from nltk.corpus import cmudict 


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            probs[1][i] = self.O[i][x[0]] * self.A_start[i]
            seqs[1][i] = str(i)

        for j in range(1, M):
            for state in range(self.L):
                possible_prob = [probs[j][prev_state] \
                    * self.A[prev_state][state] * self.O[state][x[j]] \
                    for prev_state in range(self.L)]
                prefix = seqs[j][np.argmax(possible_prob)]
                probs[j + 1][state] = max(possible_prob)
                seqs[j + 1][state] = prefix + str(state)

        max_index = np.argmax(probs[M]) 
        return seqs[M][max_index]

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        alphas[1]= [self.O[i][x[0]] * self.A_start[i] for i in range(self.L)]

        for i in range(1, M):
            for state in range(self.L):
                
                alphas[i + 1][state] = self.O[state][x[i]] * \
                sum([alphas[i][j] * self.A[j][state] for j in range(self.L)])

        if normalize:
            for i in range(0, M):
                alphas[i + 1] = np.array(alphas[i + 1]) / sum(alphas[i + 1])

        return alphas



    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

       

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        betas[M] = [1 for _ in range(self.L)]

        for i in range(M - 1, 0, -1):
            for state in range(self.L):
                betas[i][state] = sum([betas[i + 1][j] * self.A[state][j] \
                    * self.O[j][x[i]] for j in range(self.L)])

        if normalize:
            for i in range(M, 0, -1):
                betas[i] = np.array(betas[i]) / sum(betas[i])

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for a in range(self.L):
            for b in range(self.L):
                denominator = 0
                numerator = 0
                for j in range(len(X)):
                    for i in range(len(Y[j]) - 1):
                        denominator += (Y[j][i] == b)
                        numerator += (Y[j][i + 1] == a and Y[j][i] == b)
                self.A[b][a] = numerator / denominator


        # Calculate each element of O using the M-step formulas.
        for w in range(self.L):
            for z in range(self.D):
                denominator = 0
                numerator = 0
                for i in range(len(X)):
                    for j in range(len(X[i])):
                        denominator += (Y[i][j] == w)
                        numerator += (X[i][j] == z and Y[i][j] == w)
                self.O[w][z] = numerator / denominator




    def unsupervised_learning(self, X, N_iters, reverse=False):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''


        for iterations in range(N_iters):
                    
            # Initialize zero matrixes A_numerator, A_denominator, O_numerator, O_denominator
            A_numerator = [[0 for i in range(self.L)] for j in range(self.L)]
            O_numerator = [[0 for i in range(self.D)] for j in range(self.L)]
            A_denominator = [[0.] for i in range(self.L)]
            O_denominator = [[0.] for i in range(self.L)]
            
            for x in X:
                if reverse:
                  x.reverse()

                M_j = len(x)

                # Calculate alphas and betas using forward and backward methods
                alphas = self.forward(x, normalize = True)
                betas = self.backward(x, normalize = True)
        

                for i in range(M_j):

                    # Calculate and normalize marginal probabilities
                    state_marginal = [alphas[i + 1][z] * betas[i + 1][z] \
                                        for z in range(self.L)]
                    state_marginal = np.array(state_marginal) / sum(state_marginal)

                    if i > 0:
                        
                        # Calculate and normalize marginal probabilities
                        change_state_marginal = np.array([[alphas[i][a] \
                            * self.A[a][b] * self.O[b][x[i]] * betas[i + 1][b] \
                            for b in range(self.L)] for a in range(self.L) ])
                        change_state_marginal = np.divide(change_state_marginal, \
                            sum(sum(change_state_marginal)))
                        A_numerator += change_state_marginal

                    for z in range(self.L):
                        O_numerator[z][x[i]] += state_marginal[z]
                        O_denominator[z] += state_marginal[z]
                        
                    for a in range(self.L):
                        if (i + 1 >= M_j):
                            break
                        A_denominator[a] += state_marginal[a]


            self.A = np.array(A_numerator) / np.array(A_denominator)
            self.O = np.array(O_numerator) / np.array(O_denominator)



    def generate_emission(self, M, obs_map_r, syllable_dictionary, first=None):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:                      Length of the emission to generate; number of 
                                    syllables in the stanza.

            obs_map_r:              Maps to emission number to word

            syllable_dictionary:    Dictionary that maps all words to the 
                                    number of syllables that they have. The 
                                    dictionary isn't directly used by the 
                                    function, but is passed to add_syllables()

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        syllables = 0
          
        states.append(np.random.choice(range(self.L)))
          
        if first is None:
            first_emission = np.random.choice(range(self.D), p = self.O[states[0]])
            while first_emission in [0, 1, 2, 3, 4, 5]:
                first_emission = np.random.choice(range(self.D), p = self.O[states[0]])
        else:
            first_emission = first

        # emission.append(first_emission)
        syllables = self.add_syllables(obs_map_r[first_emission], syllable_dictionary, M)
        count = syllables



        while True:
            poss_state = np.random.choice(range(self.L), p = self.A[states[0]])
            poss_emission = np.random.choice(range(self.D), p = self.O[poss_state])
            if poss_emission in [0, 1, 2, 3, 4, 5]:
                break


        first_state = states[0]
        states = [poss_state]
        states.append(first_state)
        emission = [poss_emission]
        emission.append(first_emission)

        i = 1
          
          # Loop until we get M syllables in the stanza
        while syllables < M:
  
            poss_state = np.random.choice(range(self.L), p = self.A[states[i - 1]])
            poss_emission = np.random.choice(range(self.D), p = self.O[poss_state])
            
            if poss_emission in [0, 1, 2, 3, 4, 5]:
                continue
            else:
                num_added = self.add_syllables(obs_map_r[poss_emission], syllable_dictionary, M - syllables)
            # If poss_emission doesn't have the desired number of syllables,
            # continue the while loop to find another possible emission.
            if num_added == None:
                continue
            
            # Otherwise, append state and emission 
            elif num_added + count <= M:
                  states.append(poss_state)
                  emission.append(poss_emission)
                  syllables += num_added
                  count += num_added
                  i += 1

        return emission, states

    def add_syllables(self, word, syllable_dictionary, max_syllables):

        '''
          Finds the number of syllables that a word will add to the sentence. 

          Arguments:
            word:                   Word that the HMM wants to add to the sentence.

            syllable_dictionary:    Dictionary that maps all words to the 
                                    number of syllables that they have

            max_syllables:          Max possible number of syllables that one 
                                    can add to the sentence.

          Returns:
               Number of syllables of the word.

            
          '''

        # List of possible number of syllables of the word
        possible_word_syllables = syllable_dictionary[word][0]

        # List of possible number of syllables of the word if word is end word
        possible_end_cases = syllable_dictionary[word][1]

        choices = []

        # Add possible number of syllables to a choiceslist
        for i in range(len(possible_word_syllables)):
            if possible_word_syllables[i] <= max_syllables:
                choices.append(possible_word_syllables[i])

        # If choices is empty and there is a "syllable exception" when the 
        # word is the last one in a line, check if the number of syllables in the
        # "syllable exception" is valid
        if len(choices) == 0 and len(possible_end_cases) != 0:
            for j in range(len(possible_end_cases)):
                if possible_end_cases[j] == max_syllables:
                    return possible_end_cases[j]

        # If choices list is nonempty, pick a random number of syllables 
        # from the list
        elif len(choices) != 0:
            return random.choice(choices)

        else:
            return None


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters, reverse=False):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]
 
    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
 
    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters, reverse=False)

    return HMM
