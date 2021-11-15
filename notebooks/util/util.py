#!/usr/bin/env python
# -*- coding: utf-8 -*-

import igraph as ig
from scipy import stats
from scipy import sparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from ortools.linear_solver import pywraplp
import osqp
import copy

def build_website_graph(
        nnodes: int= 10,
        rate: float= 2,
        extra_arc_fraction : float = 0.0,
        seed : int = None):
    """
    """
    if seed is not None:
        np.random.seed(seed)
    graph = ig.Graph(directed=True)
    dist = stats.distributions.poisson(rate)
    # Build the root node
    root = graph.add_vertex()
    graph.add_edge(root, root)
    # Add chldren in a bradth-first fashion
    leaves = [root.index]
    while len(graph.vs) < nnodes:
        # Pick a random leave
        parent = np.random.choice(leaves, 1)[0]
        leaves.remove(parent)
        # Pick a number of children
        nchildren = dist.rvs()
        # Create all children
        for _ in range(nchildren):
            if len(graph.vs) < nnodes:
                child = graph.add_vertex()
                graph.add_edge(parent, child)
                graph.add_edge(child, child)
                leaves.append(child.index)
    # Add extra arcs (as a fraction of the number of nodes)
    if extra_arc_fraction > 0:
        # Build all possible arcs
        missing_arcs = [(i, j)
                for i in range(nnodes) for j in range(nnodes)
                if not graph[i,j]]
        # Determine number of extra arcs
        n_extra = int(extra_arc_fraction * len(missing_arcs))
        # Pick a random subsetA
        aidx = np.random.choice(len(missing_arcs), n_extra, replace=False)
        for i in aidx:
            graph.add_edge(missing_arcs[i][0], missing_arcs[i][1])
        # for _ in range(n_extra_arcs):
        #     src = np.random.choice(graph.vs, 1)[0]
        #     snks = [v for v in graph.vs if v.index not in (src.index, root.index)]
        #     snk = np.random.choice(snks, 1)[0]
        #     graph.add_edge(src, snk)
    return graph


def route_random_flow(
        start_node : ig.Vertex,
        min_units : float,
        max_units : float,
        eoh : int,
        max_len : int = None,
        seed : int = None):
    # Seed the RNG
    if seed is not None:
        np.random.seed(seed)
    # Pick a starting point within the horizon
    tme = np.random.randint(0, eoh)
    # Pick a length
    n_steps = eoh - tme  # Cap at EOH
    if max_len is not None:
        n_steps = min(n_steps, np.random.randint(1, max_len + 1))
    # Pick a number of units
    n_units = min_units + np.random.random() * (max_units - min_units)
    # Build the path
    path = []
    next_nodes = [start_node]
    for _ in range(n_steps):
        # Pick a random node
        node = np.random.choice(next_nodes, 1)[0]
        path.append((tme, node.index))
        # Determine the next nodes
        next_nodes = node.successors()
        # Advance time
        tme += 1
    return n_units, path


def build_random_paths(
        graph : ig.Graph,
        min_paths : int,
        max_paths : int,
        min_units : float,
        max_units : float,
        eoh : int,
        max_len : int = None,
        seed : int = None):
    """
    """
    # Seed the RNG
    if seed is not None:
        np.random.seed(seed)
    # Pick a number of paths
    n_paths = np.random.randint(min_paths, max_paths + 1)
    # Prepare a data structure for the results
    paths = []
    # Route flow units along the paths
    for _ in range(n_paths):
        # Route a random path
        n_units, path = route_random_flow(
            start_node=graph.vs[0],
            min_units=min_units,
            max_units=max_units,
            eoh=eoh,
            max_len=max_len,
            seed=None)
        paths.append((n_units, path))
    return paths


def get_dynamic_counts(
        graph : ig.Graph,
        eoh : int,
        paths):
    """
    """
    # Prepare a data structure for the results
    node_counts = {(t, i):0
            for t in range(eoh)
            for i in range(len(graph.vs))}
    arc_counts = {(t, e.source, e.target):0
            for t in range(1, eoh)
            for e in graph.es}
    # Route flow units along the paths
    for n_units, path in paths:
        # Update the counts
        past = None
        for t, i in path:
            node_counts[(t, i)] += n_units
            if past is not None:
                arc_counts[(t, past, i)] += n_units
            past = i
    return node_counts, arc_counts


def build_time_unfolded_graph(
        graph : ig.Graph,
        eoh : int):
    """
    """
    # Start with an empty graph
    res = ig.Graph(directed=True)
    res['unfolded'] = True
    # Add the nodes
    nodes = {(t,i): res.add_vertex(time=t, index_o=i)
            for t in range(eoh)
            for i in range(len(graph.vs))}
    # Add the edges
    for t in range(1, eoh):
        for e in graph.es:
            i, j = e.source, e.target
            res.add_edge(nodes[(t-1, i)], nodes[(t, j)],
                    time=t, source_o=i, target_o=j)
    # Return result
    return res


def apply_weights(
        tug : ig.Graph,
        node_weights : dict,
        arc_weights : dict):
    """
    """
    # Makde a copy of the graph
    res = copy.deepcopy(tug)
    # Store node counts as vertex weights
    for v in res.vs:
        v['weight'] = node_weights[(v['time'], v['index_o'])]
    # Store arc counts as edge weights
    for e in res.es:
        e['weight'] = arc_weights[(e['time'], e['source_o'], e['target_o'])]
    return res


def get_visual_style(graph, weight_scale=5):
    """
    """
    res = {}
    # Define color map
    cmap = cm.get_cmap('Set3')
    # Handle unfolded graphs
    if 'unfolded' in graph.attributes() and graph['unfolded']:
        # Disable automatic curving
        res['autocurve'] = False
        # Define the labels
        res['vertex_label'] = [f'{n["time"], n["index_o"]}'
                for n in graph.vs]
        # Define the layout
        coords = [(n['time'],
                   n['index_o'] if n['index_o'] >= 0 else 0.5)
                   for n in graph.vs]
        res['layout'] = ig.Layout(coords=coords)
        # Define the vertex colors
        res['vertex_color'] = [(cmap(n['index_o'])
            if n['index_o'] >= 0 else '#BBB')
            for n in graph.vs]
    else:
        # Define the labels
        res['vertex_label'] = [f'{n.index}' for n in graph.vs]
        # Define the layout
        res['layout'] = graph.layout('kk')
        # Define the vertex colors
        res['vertex_color'] = [cmap(n.index) for n in graph.vs]
    # Choose the vertex label size
    res['vertex_label_size'] = 10
    res['vertex_size'] = 30
    # Define the edge width
    if 'weight' in graph.es.attributes():
        max_wgt = max(e['weight'] for e in graph.es)
        min_wgt = min(e['weight'] for e in graph.es)
        scale = weight_scale / max_wgt
        res['edge_width'] = [max(1, (e['weight']-min_wgt) * scale)
                for e in graph.es]
        res['edge_color'] = ['#000' if e['weight'] > 0 else '#BBB'
                for e in graph.es]
    else:
        res['edge_width'] = 1
        res['edge_arrow_size'] = 1
    # Define the node size
    # res['vertex_size'] = 1.0
    # Assign the visual style
    return res


def _add_source_to_tug(time_unfolded_graph : ig.Graph):
    """
    """
    assert(time_unfolded_graph['unfolded'])
    # Make a copy of the graph
    g = copy.deepcopy(time_unfolded_graph)
    # Build a fake source node
    source = g.add_vertex(time=-1, index_o=-1)
    # Add edges from the source to all copies of the root node
    root_copies = time_unfolded_graph.vs.select(index_o=0)
    for root_copy in root_copies:
        t = root_copy['time']
        j = root_copy['index_o']
        if 'weight' in g.es.attributes():
            g.add_edge(source, root_copy, weight=0,
                    time=t, source_o=-1, target_o=j)
        else:
            g.add_edge(source, root_copy,
                    time=t, source_o=-1, target_o=j)
    # Return the paths
    return g, source


def find_all_paths(graph : ig.Graph,
                   source : ig.Vertex,
                   exclude_source : bool = False):
    """
    """
    # Prepare a data structure to store the paths
    res = []
    def _find_paths(v, path):
        # Make a copy of the current path
        lpath = [v for v in path]
        # Add the current node to the path
        if not exclude_source or v.index != source.index:
            lpath.append(v.index)
        # Get the successors
        successors = graph.successors(v)
        # If there are not successor, store the path
        if len(successors) == 0:
            res.append(lpath)
        # Otherwise, recurse on each successor
        else:
            for n in successors:
                _find_paths(graph.vs[n], lpath)
    # Find the paths
    _find_paths(source, [])
    # Return the paths
    return res


def tug_paths_to_original(time_unfolded_graph : ig.Graph,
        paths : list):
    """
    """
    assert(time_unfolded_graph['unfolded'])
    # Make a copy of all paths
    vtx = time_unfolded_graph.vs
    res = [[(vtx[v]['time'], vtx[v]['index_o']) for v in p] for p in paths]
    return res


# def original_paths_to_tug(time_unfolded_graph : ig.Graph,
#         paths : list):
#     """
#     """
#     assert(time_unfolded_graph['unfolded'])
#     # Make a copy of all paths
#     vtx = time_unfolded_graph.vs
#     res = [[(vtx[v]['time'], vtx[v]['index_o']) for v in p] for p in paths]
#     return res


def _counts_to_target_vectors(
        time_unfolded_graph : ig.Graph,
        node_counts,
        arc_counts):
    """
    """
    assert(time_unfolded_graph['unfolded'])
    # Get counts
    nnodes = len(time_unfolded_graph.vs)
    nedges = len(time_unfolded_graph.es)
    # Define the vertex target vector
    v = np.array([node_counts[(n['time'], n['index_o'])]
        for n in time_unfolded_graph.vs])
    # Define the edge target vector
    e = np.array([arc_counts[(e['time'], e['source_o'], e['target_o'])]
        for e in time_unfolded_graph.es])
    return v, e


def _paths_to_coefficient_matrices(
        time_unfolded_graph : ig.Graph,
        tug_paths : list):
    """
    """
    assert(time_unfolded_graph['unfolded'])
    # Get counts
    nnodes = len(time_unfolded_graph.vs)
    nedges = len(time_unfolded_graph.es)
    npaths = len(tug_paths)
    # Prepare the matrices
    V = sparse.csc_matrix((nnodes, npaths))
    E = sparse.csc_matrix((nedges, npaths))
    for j, p in enumerate(tug_paths):
        last = None
        for v in p:
            # Set a coefficient in the vertex matrix
            V[v, j] = 1
            # Set a coefficient in the edge matrix
            if last is not None:
                eid = time_unfolded_graph.get_eid(last, v)
                E[eid, j] = 1
            # Store the last visited vertex
            last = v
    # Return the matrices
    return V, E


def _non_negativity_constraints(
        npaths : int):
    """
    """
    # Build a sparse eye matrix
    A = sparse.eye(npaths).tocsc()
    # Build a zero lower-bound vector
    l = np.zeros(npaths)
    # Build an infinite upper-bound vector
    u = np.inf * np.ones(npaths)
    # Return the matrices
    return A, l, u


def plot_matrix(A, b=None, figsize=None, title=None, title_b=None):
    """
    """
    if b is not None:
        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=figsize,
                gridspec_kw={'width_ratios': [A.shape[1], 1]})
    else:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(A, cmap='Greys_r', interpolation='nearest')
    ax.set_title(title)
    if b is not None:
        ax2.imshow(b.reshape((-1, 1)),
            cmap='Greys_r', interpolation='nearest')
        ax2.set_title(title_b)
    # plt.tight_layout()
    plt.show()


class PathSelectionSolver(object):
    """Docstring for MyClass. """

    def __init__(self,
            time_unfolded_graph : ig.Graph,
            node_counts,
            arc_counts):
        """TODO: to be defined. """
        # Initialize the graph
        self.tug = time_unfolded_graph
        # Initialize the counts
        self.node_counts = node_counts
        self.arc_counts = arc_counts
        # Store L1 regularization weight
        self.alpha = None
        # Build the solver
        self.mdl = osqp.OSQP()
        # Define a variable for the paths
        self.paths = None
        # Define variables to store the matrices
        self.V = None
        self.E = None
        self.v = None
        self.e = None
        self.A = None
        self.l = None
        self.u = None
        # Define a variable to store the solution
        self.sol = None

    def _recompute_matrices(self, paths):
        # Store the paths (for solution reconstruction)
        self.paths = copy.deepcopy(paths)
        # Compute the coefficient matrices
        V, E = _paths_to_coefficient_matrices(self.tug, paths)
        self.V, self.E = V, E
        # Compute the target vectors
        v, e = _counts_to_target_vectors(self.tug,
                self.node_counts, self.arc_counts)
        self.v, self.e = v, e
        # Compute the semi-definite matrix
        P = V.T.dot(V) + E.T.dot(E)
        self.P = P
        # Compute the linear coefficient vector
        q = -V.T.dot(v).T - E.T.dot(e).T + self.alpha
        self.q = q
        # Compute the constraint matrix and the bound vectors
        A, l, u = _non_negativity_constraints(len(paths))
        self.A, self.l, self.u = A, l, u
        return P, q, A, l, u

    def setup(self, paths, alpha=0, **settings):
        # Store L1 regularization weight
        self.alpha = alpha
        # Recompute the problem matrices
        P, q, A, l, u = self._recompute_matrices(paths)
        # Setup the solver
        self.mdl.setup(P=P, q=q, A=A, l=l, u=u, **settings)

    def solve(self, updated_paths=None):
        # Update the matrices, if needed
        if updated_paths is not None:
            # Recompute matrices
            P, q, A, l, u = self._recompute_matrices(updated_paths)
            # Update the solver (warm starting)
            self.mdl.update(q=q, l=l, u=u)
            self.mdl.update(Px=P, Ax=A)
        # Solve the problem
        sol = self.mdl.solve()
        self.sol = sol
        # Return the solution
        return sol

    def sol_to_paths(self, eps=1e-2):
        # Quick access to the current solution
        sol = self.sol
        # Extract the paths
        res = []
        for j, p in enumerate(self.paths):
            if sol.x[j] > eps:
                path_o = [(self.tug.vs[n]['time'],
                           self.tug.vs[n]['index_o']) for n in p]
                res.append((sol.x[j], path_o))
            # else:
            #     print(f'WARNING: {sol.x[j]}')
        # Return the solution
        return res

    # def get_standard_duals(self):
    #     # Quick access to the needed data structures
    #     x = self.sol.x
    #     P = self.P
    #     q = self.q
    #     # Extract the duals
    #     # duals = 0.5 * x.T.dot((P + P.T).toarray()) + q.T
    #     duals = self.sol.y
    #     print(duals)
    #     print(duals * x)
    #     print([v for vx,v in zip(x, duals) if vx >= 1e-3])
    #     print([v for vx,v in zip(x, duals) if vx < 1e-3])

    # def _check_solution(self, sol):
    #     """
    #     """
    #     # Compute the coefficient matrices
    #     V, E = _paths_to_coefficient_matrices(self.tug, self.paths)
    #     # Compute the target vectors
    #     v, e = _counts_to_target_vectors(self.tug,
    #             self.node_counts, self.arc_counts)
    #     # Compute vertex residuals
    #     v_res = V.dot(sol.x) - v
    #     # Compute edge residualsA
    #     e_res = E.dot(sol.x) - e
    #     print(v_res)
    #     print(e_res)


def consolidate_paths(
        paths : list,
        node_counts : dict,
        arc_counts : dict,
        tlim : int = None):
    """
    """
    # Build the solver
    slv = pywraplp.Solver.CreateSolver('CBC')
    # Prepar some useful constants
    npaths = len(paths)
    inf = slv.infinity()
    # Define the flow variables
    x = [slv.NumVar(0, inf, f'x_{j}') for j in range(npaths)]
    # Define the path activation variables
    z = [slv.IntVar(0, 1, f'z_{j}') for j in range(npaths)]
    # Define the path activation variables
    M = max(v for v in node_counts.values())
    for j in range(npaths):
        slv.Add(x[j] <= M * z[j])
    # Collect paths by node and arc
    paths_by_node = {}
    paths_by_arc = {}
    for j, (_, p) in enumerate(paths):
        last = None
        for n in p:
            # Store the path idx for the node key
            if n not in paths_by_node:
                paths_by_node[n] = [j]
            else:
                paths_by_node[n].append(j)
            # Store the path idx for the arc key
            if last is not None:
                key = (n[0], last[1], n[1])
                if key not in paths_by_arc:
                    paths_by_arc[key] = [j]
                else:
                    paths_by_arc[key].append(j)
            # Update the last node
            last = n
    # Define the reconstruction error constraints for all nodes
    for n in paths_by_node:
        # Get the paths for the node
        p = paths_by_node[n]
        # Define the constraint
        slv.Add(sum(x[j] for j in p) == node_counts[n])
    # Define the reconstruction error constraints for all arcs
    for a in paths_by_arc:
        # Get the paths for the arc
        p = paths_by_arc[a]
        # Define the constraint
        slv.Add(sum(x[j] for j in p) == arc_counts[a])
    # Define the objective
    slv.Minimize(sum(z[j] for j in range(npaths)))
    # Set a time limit, if requested
    if tlim is not None:
        slv.SetTimeLimit(tlim)
    # Solve the problem
    status = slv.Solve()
    # Return the solution
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        # Extract the paths in the solution
        sol_paths = None
        sol_paths = [(x[j].solution_value(),
            paths[j][1]) for j in range(npaths)
                if z[j].solution_value() > 0.5]
        # Return the solution
        if status == slv.OPTIMAL:
            return True, sol_paths
        else:
            return False, sol_paths
    else:
        return False, None


def get_estimation_residuals(
        graph : ig.Graph,
        eoh : int,
        paths : list,
        node_counts : dict,
        arc_counts : dict):
    # Compute the counts for the set of paths
    p_node_counts, p_arc_counts = get_dynamic_counts(graph, eoh, paths)
    # Compute the differences
    node_res = {n: p_node_counts[n] - node_counts[n] for n in node_counts}
    arc_res = {a: p_arc_counts[a] - arc_counts[a] for a in arc_counts}
    # Return the residuals
    return node_res, arc_res


def plot_dict(data, title=None, figsize=None,
        data2=None, label=None, label2=None):
    """
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the data
    if data2 is None:
        x = np.arange(len(data))
    else:
        x = np.arange(0, 2*len(data), 2)
        x2 = np.arange(1, 2*len(data), 2)
    keys = sorted(data.keys())
    lbl = [f'{k}' for k in keys]
    y = np.array([data[k] for k in keys])
    plt.bar(x, y, label=label)
    if data2 is not None:
        y2 = np.array([data2[k] for k in keys])
        plt.bar(x2, y2, label=label2)
        plt.legend(loc='best')
    if data2 is None:
        xticks = x
    else:
        xticks = x + 0.5
    plt.xticks(xticks, lbl, rotation=90)
    plt.tight_layout()
    plt.show()



