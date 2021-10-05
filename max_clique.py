import cplex
import numpy as np
import os
import time
from argparse import ArgumentParser
# from igraph import Graph
import networkx as nx


def read_graph(path):
    with open(path, "r") as fp:
        for line in fp:
            if line.startswith('p'):
                _, name, vertices_num, edges_num = line.split()
                adjacency_matrix = np.zeros((int(vertices_num), (int(vertices_num))), dtype=np.bool)
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                adjacency_matrix[int(v1) - 1, int(v2) - 1] = 1
            else:
                continue
    np.fill_diagonal(adjacency_matrix, 1)
    graph = nx.convert_matrix.from_numpy_matrix(adjacency_matrix)
    return graph


def parse_args():
    parser = ArgumentParser(
        description="Finds max clique for DIMACS graphs"
    )
    parser.add_argument("-p",
                        "--path",
                        help="Path to source graph file in .clq format",
                        default="DIMACS_all_ascii/brock200_1.clq",
                        type=str)
    parser.add_argument('-m',
                        "--method",
                        type=str, choices=["LP", "ILP"],
                        default="LP",
                        help='Treat the problem as LP or ILP')

    return parser.parse_args()

def get_ind_sets(graph):
    ind_sets = []
    strategies = [nx.coloring.strategy_largest_first,
                  nx.coloring.strategy_random_sequential,
                  nx.coloring.strategy_independent_set,
                  nx.coloring.strategy_connected_sequential_bfs,
                  nx.coloring.strategy_connected_sequential_dfs,
                  nx.coloring.strategy_saturation_largest_first]

    for strategy in strategies:
        d = nx.coloring.greedy_color(graph, strategy=strategy)
        for color in set(color for node, color in d.items()):
            ind_sets.append(
                [key for key, value in d.items() if value == color])
    return ind_sets

def construct_problem(graph, solve_type):
    problem = cplex.Cplex()
    problem.set_log_stream(None)
    problem.set_results_stream(None)
    problem.set_warning_stream(None)
    problem.set_error_stream(None)
    problem.objective.set_sense(problem.objective.sense.maximize)
    if solve_type == "LP":
        type_one = 1.0
        type_zero = 0.0
        var_type = problem.variables.type.continuous
    elif solve_type == "ILP":
        type_one = 1
        type_zero = 0
        var_type = problem.variables.type.binary
    else:
        raise NotImplementedError("Solve type should be in ['LP', 'ILP']")
    # num_nodes = graph.vcount()
    num_nodes = graph.number_of_nodes()
    obj = [type_one] * num_nodes
    upper_bounds = [type_one] * num_nodes
    lower_bounds = [type_zero] * num_nodes
    types = zip(range(num_nodes), [var_type] * num_nodes)
    # lower bounds are all 0.0 (the default)
    columns_names = [f'x{x}' for x in range(num_nodes)]
    # not_connected = graph.complementer().get_edgelist()
    not_connected = nx.complement(graph).edges
    # independent_sets = graph.independent_vertex_sets(min=3, max=6)
    independent_sets = get_ind_sets(graph)

    problem.variables.add(obj=obj, ub=upper_bounds, lb=lower_bounds,
                          names=columns_names)
    problem.variables.set_types(types)

    constraints = []

    for xi, xj in not_connected:
        constraints.append(
            [['x{0}'.format(xi), 'x{0}'.format(xj)], [type_one, type_one]])
    for ind_set in independent_sets:
        if len(ind_set) > 2:
            constraints.append([['x{0}'.format(x) for x in ind_set], [type_one] * len(ind_set)])

    contstraints_length = len(constraints)
    right_hand_side = [type_one] * contstraints_length
    constraint_names = [f'c{x}' for x in range(contstraints_length)]
    constraint_senses = ['L'] * contstraints_length

    problem.linear_constraints.add(lin_expr=constraints,
                                   senses=constraint_senses,
                                   rhs=right_hand_side,
                                   names=constraint_names)
    return problem

def main():
    args = parse_args()
    graph = read_graph(args.path)
    problem_max_clique = construct_problem(graph, args.method)
    start = time.time()
    problem_max_clique.solve()
    print("\n", time.time() - start)
    values = problem_max_clique.solution.get_values()
    objective_value = problem_max_clique.solution.get_objective_value()

    print(f"objective value: {objective_value} ")
    print("values:")
    for idx in range(len(values)):
        if values[idx] != 0:
            print(f"\tx_{idx} = {values[idx]}", end='')

if __name__ == "__main__":
    start = time.time()
    main()
    print("\nTotal:", time.time() - start)
