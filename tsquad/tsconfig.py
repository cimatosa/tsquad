"""
These parameters control the pre-calculation of the nodes and weights use for the
numeric tanh-sinh integration scheme.
"""

# this file holds the pre-calculated nodes and weights, it will be generated on the fly and is controlled by
# the parameters given below
_f_name = "nodes_weights.py"

# various limit for t_max,
#     t_max=3      -> x_low = 4.3e-14
#     t_max=6.1124 -> x_low = 2.2e-308
t_max_list = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.1124]

# whereas t_min is fixed because x(t < -3.2) = 2. (using double precision)
t_min = -3.2

N_t_max = len(t_max_list)

# 2*N_0+1  sets the initial number of equally distributed nodes on the t-axis
N_0 = 4

# the maximum number of doubling the nodes to achieve a finer grid before raising
# an Exception that the integration has not converged
num_sub_grids = 8
