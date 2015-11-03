#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cython.parallel import prange

cdef extern from "math.h" nogil:
    double fabs(double)


cdef double dot(double[::1] x,
                double[::1] y,
                int dim) nogil:

    cdef int i
    cdef double result = 0.0

    for i in range(dim):
        result += x[i] * y[i]

    return result


def compute_rank_violations(double[:, ::1] wordvec,
                            double[::1] wordvec_norm,
                            double[:, ::1] input,
                            int[:] expected,
                            int[:, ::1] inputs,
                            int[::1] rank_violations,
                            int no_threads):
    """
    Compute the rank violations
    of the expected words in the word analogy task.
    """

    cdef int i, j, k, no_input_vectors, no_wordvec, skip_word
    cdef int no_components, violations

    cdef double score_of_expected, score

    no_input_vectors = input.shape[0]
    no_wordvec = wordvec.shape[0]
    no_components = wordvec.shape[1]

    with nogil:
        for i in prange(no_input_vectors, num_threads=no_threads,
                        schedule='dynamic'):

            # Compute the score of the expected word.
            score_of_expected = (dot(input[i],
                                     wordvec[expected[i]],
                                     no_components)
                                     / wordvec_norm[expected[i]])

            # Compute all other scores and count
            # rank violations.
            violations = 0

            for j in range(no_wordvec):

                # Words from the input do not
                # count as violations.
                skip_word = 0
                for k in range(4):
                    if inputs[i, k] == j:
                        skip_word = 1
                        break

                if skip_word == 1:
                    continue

                score = (dot(input[i],
                            wordvec[j],
                            no_components)
                         / wordvec_norm[j])

                if score >= score_of_expected:
                    violations = violations + 1

            # Update the average rank with the rank
            # of this example.
            rank_violations[i] = violations


def get_closest(double[:, ::1] wordvec,
                            double[::1] wordvec_norm,
                            double[:, ::1] input,
                            int[:] expected,
                            int[:, ::1] inputs,
                            int[::1] rank_violations,
                            int no_threads):
    """
    Return the index of the word that is the best answer for the analogy task
    """

    cdef int i, j, k, no_input_vectors, no_wordvec, skip_word
    cdef int no_components, largest_index

    cdef double score, largest_value

    no_input_vectors = input.shape[0]
    no_wordvec = wordvec.shape[0]
    no_components = wordvec.shape[1]

    with nogil:
        for i in prange(no_input_vectors, num_threads=no_threads,
                        schedule='dynamic'):
                            
            #reset these values
            largest_index = -1
            largest_value = -1                    


            for j in range(no_wordvec):

                # Words from the input do not
                # count as potential matches.
                skip_word = 0
                for k in range(4):
                    if inputs[i, k] == j:
                        skip_word = 1
                        break

                if skip_word == 1:
                    continue

                score = (dot(input[i],
                            wordvec[j],
                            no_components)
                         / wordvec_norm[j])

                if (largest_index == -1) or (score > largest_value):
                    largest_value = score
                    largest_index = j
            
            rank_violations[i] = largest_index


