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
from ortools.sat.python import cp_model

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
        seed : int = None,
        max_wait : int = None):
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
    nwaits = 0
    for _ in range(n_steps):
        # Pick a random node
        node = np.random.choice(next_nodes, 1)[0]
        path.append((tme, node.index))
        # Count waits
        if node.index == path[-1][1]:
            nwaits += 1
        else:
            nwaits = 0
        # Determine the next nodes
        next_nodes = node.successors()
        if max_wait is not None and nwaits >= max_wait:
            next_nodes = [n for n in next_nodes if n.index != node.index]
        # Stop if there are no viable successors
        if len(next_nodes) == 0:
            break
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
        seed : int = None,
        max_wait : int = None):
    """
    """
    # Seed the RNG
    if seed is not None:
        np.random.seed(seed)
    # Pick a number of paths
    n_paths = np.random.randint(min_paths, max_paths + 1)
    # Prepare a data structure for the results
    flows, paths = [], []
    # Route flow units along the paths
    for i in range(n_paths):
        # Pick a random start node
        start_node = np.random.choice(graph.vs, 1)[0]
        # Route a random path
        ls = None if seed is None else seed + i
        n_units, path = route_random_flow(
            start_node=start_node,
            min_units=min_units,
            max_units=max_units,
            eoh=eoh,
            max_len=max_len,
            seed=ls,
            max_wait=max_wait)
        flows.append(n_units)
        paths.append(path)
    return flows, paths


def get_counts(
        tug : ig.Graph,
        flows,
        paths):
    """
    """
    assert(tug['unfolded'])
    # Prepare a data structure for the results
    eoh = tug['eoh']
    node_counts = {(t, v['index_o']):0
            for t in range(eoh)
            for v in tug.vs}
    arc_counts = {(t, e['source_o'], e['target_o']):0
            for t in range(1, eoh)
            for e in tug.es}
    # Route flow units along the paths
    for flow, path in zip(flows, paths):
        # Update the counts
        past = None
        for k in path:
            try:
                t, i = k
            except TypeError:
                t, i = tug.vs[k]['time'], tug.vs[k]['index_o']
            node_counts[(t, i)] += flow
            if past is not None:
                arc_counts[(t, past, i)] += flow
            past = i
    return node_counts, arc_counts


def add_proportional_noise(node_counts_in, arc_counts_in, sigma, seed=None):
    """
    """
    # Add noise to the node counts
    node_counts = {}
    for k, v in node_counts_in.items():
        node_counts[k] = max(0, v * (1 + np.random.normal(0, sigma)))
    # Add noise to the arc counts
    arc_counts = {}
    np.random.seed(seed)
    for k, v in arc_counts_in.items():
        arc_counts[k] = max(0, v * (1 + np.random.normal(0, sigma)))
    return node_counts, arc_counts


def build_time_unfolded_graph(
        graph : ig.Graph,
        eoh : int):
    """
    """
    # Start with an empty graph
    res = ig.Graph(directed=True)
    res['unfolded'] = True
    res['eoh'] = eoh
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


# def apply_weights(
#         tug : ig.Graph,
#         node_weights : dict,
#         arc_weights : dict):
#     """
#     """
#     # Makde a copy of the graph
#     res = copy.deepcopy(tug)
#     # Store node counts as vertex weights
#     for v in res.vs:
#         v['weight'] = node_weights[(v['time'], v['index_o'])]
#     # Store arc counts as edge weights
#     for e in res.es:
#         e['weight'] = arc_weights[(e['time'], e['source_o'], e['target_o'])]
#     return res


def get_visual_style(
        graph : ig.Graph,
        vertex_weights = None,
        edge_weights = None):
    """
    """
    res = {}
    cl, cu = 0.1, 0.9
    # Rescale weights
    if vertex_weights is not None:
        u = np.max(np.abs(list(vertex_weights.values())))
        vw = {k: np.clip((v+u) / (2*u), cl, cu)
                for k, v in vertex_weights.items()}
    if edge_weights is not None:
        u = np.max(np.abs(list(edge_weights.values())))
        ew = {k: np.clip((v+u) / (2*u), cl, cu)
                for k, v in edge_weights.items()}
    # Define color map
    if vertex_weights is None:
        cmap = cm.get_cmap('Set3')
    else:
        cmap = cm.get_cmap('coolwarm')
    # Handle unfolded graphs
    if 'unfolded' in graph.attributes() and graph['unfolded']:
        # Disable automatic curving
        res['autocurve'] = False
        # Layout height
        height = max(graph.vs['index_o'])
        # Build vertex attributes
        res['vertex_label'] = []
        res['vertex_color'] = []
        coords = []
        for n in graph.vs:
            t, i = n['time'], n['index_o']
            # Vertex labels
            res['vertex_label'].append(f'{t},{i}')
            # Coordinates
            if i >= 0: coords.append((t, i))
            else: coords.append((-1, 0.5 * height))
            # Color
            if vertex_weights is not None:
                if (t, i) in vw:
                    res['vertex_color'].append(cmap(vw[(t, i)]))
                else:
                    res['vertex_color'].append(cmap(0.5))
            else:
                if i >= 0:
                    res['vertex_color'].append(cmap(i))
                else:
                    res['vertex_color'].append('#BBB')
        # Setup the layout
        res['layout'] = ig.Layout(coords=coords)
        # Define the edge colors
        if edge_weights is not None:
            res['edge_color'] = []
            for e in graph.es:
                t, i, j = e['time'], e['source_o'], e['target_o']
                if (t, i, j) in ew:
                    res['edge_color'].append(cmap(ew[(t, i, j)]))
                else:
                    res['edge_color'].append(cmap(0.5))
    else:
        # Define the labels
        res['vertex_label'] = [f'{n.index}' for n in graph.vs]
        # Define the layout
        res['layout'] = graph.layout('kk')
        # Define the vertex colors
        if vertex_weights is None:
            res['vertex_color'] = [cmap(n.index) for n in graph.vs]
        else:
            res['vertex_color'] = [cmap(ew[n.index]) for n in graph.vs]
        # Define the edge colors
        if edge_weights is not None:
            res['edge_color'] = []
            for e in graph.es:
                i, j = e['source'], e['target']
                res['edge_color'].append(cmap(ew[(t, i, j)]))
    # Choose the vertex label size
    res['vertex_label_size'] = 10
    res['vertex_size'] = 30




    # # Define the edge width
    # if 'weight' in graph.es.attributes():
    #     max_wgt = max(e['weight'] for e in graph.es)
    #     min_wgt = min(e['weight'] for e in graph.es)
    #     scale = weight_scale / max_wgt
    #     res['edge_width'] = [max(1, (e['weight']-min_wgt) * scale)
    #             for e in graph.es]
    #     res['edge_color'] = ['#000' if e['weight'] > 0 else '#BBB'
    #             for e in graph.es]
    # else:
    #     res['edge_width'] = 1
    #     res['edge_arrow_size'] = 1
    # Define the node size
    # res['vertex_size'] = 1.0
    # Assign the visual style
    return res


def _add_source_to_tug(tug : ig.Graph):
    """
    """
    assert(tug['unfolded'])
    # Make a copy of the graph
    g = copy.deepcopy(tug)
    # Build a fake source node
    source = g.add_vertex(time=-1, index_o=-1)
    # Add edges from the source to all nodes
    for node in tug.vs:
        t = node['time']
        j = node['index_o']
        if 'weight' in g.es.attributes():
            g.add_edge(source, node, weight=0,
                    time=t, source_o=-1, target_o=j)
        else:
            g.add_edge(source, node,
                    time=t, source_o=-1, target_o=j)
    # Return the paths
    return g, source


def enumerate_paths(graph : ig.Graph,
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
            # Store the path
            res.append(lpath)
        # Get the successors
        successors = graph.successors(v)
        # Otherwise, recurse on each successor
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


def _min_vertex_cover_constraints(
        tug : ig.Graph,
        paths : list,
        node_counts : dict,
        min_vertex_cover : float):
    """
    """
    assert(tug['unfolded'])
    # Build the coefficient matrix
    nnodes = len(tug.vs)
    npaths = len(paths)
    A = sparse.csc_matrix((nnodes, npaths))
    for j, p in enumerate(paths):
        for v in p:
            A[v, j] = 1
    # Build the lower-bound vector
    l = []
    for n in tug.vs:
        nc = node_counts[(n['time'], n['index_o'])]
        l.append(min_vertex_cover * nc)
    l = np.array(l)
    # Build the upper-bound vector
    u = np.inf * np.ones(nnodes)
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
            arc_counts,
            alpha : float = 0,
            min_vertex_cover=None):
        """TODO: to be defined. """
        # Initialize the graph
        self.tug = time_unfolded_graph
        # Initialize the counts
        self.node_counts = node_counts
        self.arc_counts = arc_counts
        # Initialize the L1 regularization term
        self.alpha = alpha
        # Initialize the minimum cover fractionA
        self.mvc = min_vertex_cover
        # Store L1 regularization weight
        self.alpha = alpha
        # Build the solver
        self.mdl = None
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
        # Build the matrices for the min vertex cover constraint
        if self.mvc is not None:
            A, l, u = _min_vertex_cover_constraints(self.tug, paths,
                    self.node_counts, self.mvc)
            self.A = sparse.vstack([self.A, A])
            self.l = np.concatenate([self.l, l])
            self.u = np.concatenate([self.u, u])
        return self.P, self.q, self.A, self.l, self.u

    # def setup(self, paths, alpha=0, **settings):
    #     # Store L1 regularization weight
    #     self.alpha = alpha
    #     # Recompute the problem matrices
    #     P, q, A, l, u = self._recompute_matrices(paths)
    #     # Setup the solver
    #     self.mdl.setup(P=P, q=q, A=A, l=l, u=u, **settings)

    def solve(self, paths, **settings):
        # Build the solver
        self.mdl = osqp.OSQP()
        # Recompute the problem matrices
        P, q, A, l, u = self._recompute_matrices(paths)
        # Setup the solver
        self.mdl.setup(P=P, q=q, A=A, l=l, u=u, **settings)
        # if updated_paths is not None:
        #     # Recompute matrices
        #     P, q, A, l, u = self._recompute_matrices(updated_paths)
        #     # Update the solver (warm starting)
        #     self.mdl.update(q=q, l=l, u=u)
        #     self.mdl.update(Px=P, Ax=A)
        # Solve the problem
        sol = self.mdl.solve()
        self.sol = sol
        # Return the solution
        return self.sol

    def sol_to_paths(self, eps=1e-2):
        # Quick access to the current solution
        sol = self.sol
        # Extract the paths
        flows, paths = [], []
        for j, p in enumerate(self.paths):
            if sol.x[j] > eps:
                flows.append(sol.x[j])
                paths.append(p.copy())
            # else:
            #     print(f'WARNING: {sol.x[j]}')
        # Return the solution
        return flows, paths

    def get_cover_duals(self):
        if self.mvc is not None:
            return self.sol.y[-len(self.tug.vs):]
        else:
            return None

    def get_non_negativity_duals(self):
        return self.sol.y[:len(self.paths)]

    # def sol_to_paths(self, eps=1e-2):
    #     # Quick access to the current solution
    #     sol = self.sol
    #     # Extract the paths
    #     res = []
    #     for j, p in enumerate(self.paths):
    #         if sol.x[j] > eps:
    #             path = [(self.tug.vs[n]['time'],
    #                        self.tug.vs[n]['index_o']) for n in p]
    #             res.append((sol.x[j], path))
    #         # else:
    #         #     print(f'WARNING: {sol.x[j]}')
    #     # Return the solution
    #     return res

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


def _paths_by_node_and_arc(tug, paths):
    """
    """
    paths_by_node = {}
    paths_by_arc = {}
    for j, p in enumerate(paths):
        last = None
        for n in p:
            nk = tug.vs[n]['time'], tug.vs[n]['index_o']
            # Store the path idx for the node key
            if nk not in paths_by_node:
                paths_by_node[nk] = [j]
            else:
                paths_by_node[nk].append(j)
            # Store the path idx for the arc key
            if last is not None:
                ak = (nk[0], last[1], nk[1])
                if ak not in paths_by_arc:
                    paths_by_arc[ak] = [j]
                else:
                    paths_by_arc[ak].append(j)
            # Update the last node
            last = nk
    return paths_by_node, paths_by_arc


def consolidate_paths(
        tug : ig.Graph,
        paths : list,
        node_counts : dict,
        arc_counts : dict,
        tlim : int = None):
    """
    """
    assert(tug['unfolded'])
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
    paths_by_node, paths_by_arc = _paths_by_node_and_arc(tug, paths)
    # paths_by_node = {}
    # paths_by_arc = {}
    # for j, p in enumerate(paths):
    #     last = None
    #     for n in p:
    #         nk = tug.vs[n]['time'], tug.vs[n]['index_o']
    #         # Store the path idx for the node key
    #         if nk not in paths_by_node:
    #             paths_by_node[nk] = [j]
    #         else:
    #             paths_by_node[nk].append(j)
    #         # Store the path idx for the arc key
    #         if last is not None:
    #             ak = (nk[0], last[1], nk[1])
    #             if ak not in paths_by_arc:
    #                 paths_by_arc[ak] = [j]
    #             else:
    #                 paths_by_arc[ak].append(j)
    #         # Update the last node
    #         last = nk
    # Define the reconstruction error constraints for all nodes
    for n, p in paths_by_node.items():
        # Define the constraint
        slv.Add(sum(x[j] for j in p) == node_counts[n])
    # Define the reconstruction error constraints for all arcs
    for a, p in paths_by_arc.items():
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
        sol_flows, sol_paths = [], []
        for j in range(npaths):
            if z[j].solution_value() > 0.5:
                sol_flows.append(x[j].solution_value())
                sol_paths.append(paths[j])
        # sol_flows = [x[j].solution_value() for ]
        # sol_paths = [(,
        #     paths[j]) for j in range(npaths)
        #         if z[j].solution_value() > 0.5]
        # Return the solution
        if status == slv.OPTIMAL:
            return sol_flows, sol_paths, True
        else:
            return sol_flows, sol_paths, False
    else:
        return None, None, False


def _get_residuals(
        tug : ig.Graph,
        flows : list,
        paths : list,
        node_counts : dict,
        arc_counts : dict):
    assert(tug['unfolded'])
    # Compute the counts for the set of paths
    p_node_counts, p_arc_counts = get_counts(tug, flows, paths)
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



def tug_shortest_paths(tug : ig.Graph,
                   node_weights : dict,
                   arc_weights : dict,
                   exclude_source : bool = False):
    assert(tug['unfolded'])
    # Add a fake source to the graph
    tugs, tugs_source = _add_source_to_tug(tug)
    # Prepare a data structure for the shortest paths
    nnodes = len(tugs.vs)
    sp = {i: [] for i in range(nnodes)}
    spw = {i: 0 for i in range(nnodes)}
    # Visit all nodes in topological order
    tpnodes = tugs.topological_sorting()
    for i in tpnodes:
        # Quick access to the node
        node = tugs.vs[i]
        nk = node['time'], node['index_o']
        # Update the path weight
        spw[i] += node_weights[nk] if nk in node_weights else 0
        # Add the current node to the shortest path
        sp[i].append(i)
        # Process all outgoing arcs
        for e in node.out_edges():
            # Compute the path weight
            ak = e['time'], e['source_o'], e['target_o']
            pw = spw[i] + (arc_weights[ak] if ak in arc_weights else 0)
            # Get arc target
            j = e.target
            # Check whether the path is shorter
            if len(sp[j]) == 0 or pw < spw[j]:
                # Update the shortest path
                sp[j] = sp[i].copy()
                spw[j] = pw
    # Return the shortest paths as lists
    spw = [spw[i] for i in range(nnodes) if len(sp[i]) > 0]
    sp = [sp[i] for i in range(nnodes) if len(sp[i]) > 0]
    # Exclude source, if requested
    if exclude_source:
        # NOTE all arcs from the source have 0 weight
        spw = [v for p, v in zip(sp, spw) if len(p) > 1]
        sp = [p[1:] for p in sp if len(p) > 1]
    return sp, spw


def solve_path_selection_full(
        tug : ig.Graph,
        node_counts : dict,
        arc_counts : dict,
        initial_paths : list = None,
        verbose=0,
        alpha=0,
        min_vertex_cover=None,
        return_duals=False,
        **settings):
    assert(tug['unfolded'])
    # If no intial paths are provided, enumerate
    if initial_paths is None:
        # Add a fake source to the graph
        tugs, tugs_source = _add_source_to_tug(tug)
        # Enumerate all paths
        paths = enumerate_paths(tugs, tugs_source, exclude_source=True)
    else:
        paths = initial_paths
    # Build the path selection solver
    prb = PathSelectionSolver(tug, node_counts, arc_counts,
                alpha=alpha, min_vertex_cover=min_vertex_cover)
    # Solve the path selection problem
    sol = prb.solve(paths, verbose=verbose, polish=True, **settings)
    # # DEBUG
    # cst = prb.A.dot(sol.x) - prb.l
    # cst = cst[-len(tug.vs):]
    # cvduals = prb.get_cover_duals()
    # for i, v in enumerate(cst):
    #     print(i, v, cvduals[i])
    # Convert the solution to paths on the original graph
    flows, rpaths = prb.sol_to_paths()
    # Return the solution
    res = [flows, rpaths]
    if return_duals:
        res.append(prb.get_non_negativity_duals())
        if min_vertex_cover is not None:
            res.append(prb.get_cover_duals())
    return res


def print_solution(tug, flows, paths,
        use_unfolded_indexes : bool = False,
        sort=None,
        max_paths : int = None):
    """
    """
    assert(tug['unfolded'])
    # Sort by decreasing flow
    if sort == 'ascending':
        sidx = np.argsort(flows)
    elif sort == 'descending':
        sidx = np.argsort(flows)[::-1]
    elif sort is None:
        sidx = np.arange(len(flows))
    else:
        raise ValueError(f'Unknown sort order: {sort}')
    flows = [flows[i] for i in sidx]
    paths = [paths[i] for i in sidx]
    # Print the solution
    for i, (f, p) in enumerate(zip(flows, paths)):
        if use_unfolded_indexes:
            ps = [f'{v}' for v in p]
        else:
            ps = [f'{tug.vs[v]["time"]},{tug.vs[v]["index_o"]}' for v in p]
        print(f'{f:.2f}: {" > ".join(ps)}')
        if max_paths is not None and i+1 >= max_paths:
            print('...')
            break


def print_ground_truth(flows, paths, sort=None):
    """
    """
    # Sort by decreasing flow
    if sort == 'ascending':
        sidx = np.argsort(flows)
    elif sort == 'descending':
        sidx = np.argsort(flows)[::-1]
    elif sort is None:
        sidx = np.arange(len(flows))
    else:
        raise ValueError(f'Unknown sort order: {sort}')
    flows = [flows[i] for i in sidx]
    paths = [paths[i] for i in sidx]
    # Print the paths
    for f, p in zip(flows, paths):
        ps = [f'{t},{v}' for t, v in p]
        print(f'{f:.2f}: {" > ".join(ps)}')



def solve_pricing_problem(
        tug : ig.Graph,
        flows : list,
        paths : list,
        node_counts,
        arc_counts,
        filter_paths : bool = True,
        tol = 1e-3,
        alpha = 0,
        cover_duals = None):
    assert(tug['unfolded'])
    # Obtain the residuals
    nres, ares = _get_residuals(tug, flows, paths, node_counts, arc_counts)
    # In case cover duals are passed, update the node weights
    if cover_duals is not None:
        for i, v in enumerate(tug.vs):
            nk = v['time'], v['index_o']
            nres[nk] += cover_duals[i]
    # Find all shortest paths (sequences of arcs)
    sp, spw = tug_shortest_paths(tug, nres, ares, exclude_source=True)
    # Add the alpha term
    spw = [v + alpha for v in spw]
    # Filter out paths with non-negative reduced costs
    if filter_paths:
        sp = [p for i, p in enumerate(sp) if spw[i] <= -tol]
        spw = [pw for pw in spw if pw <= -tol]
    # Sort paths by increasing reduced cost
    sidx = np.argsort(spw)
    res = [sp[i] for i in sidx]
    res_c = [spw[i] for i in sidx]
    # Return the results
    return res_c, res


def solve_pricing_problem_maxwaits(
        tug : ig.Graph,
        flows : list,
        paths : list,
        node_counts,
        arc_counts,
        max_waits : int = None,
        filter_paths : bool = True,
        prec : float = 1e-3,
        alpha : float = 0,
        cover_duals : list = None,
        time_limit : float = None,
        max_paths : int = None,
        optimize : bool = False):
    assert(tug['unfolded'])
    assert(tug['eoh'] > 1)
    assert(max_waits is None or max_waits >= 1)

    # Obtain the residuals
    nres, ares = _get_residuals(tug, flows, paths, node_counts, arc_counts)
    # Build an index map
    tug_idx = {}
    for i, v in enumerate(tug.vs):
        # Store the tug node index in the map
        nk = v['time'], v['index_o']
        tug_idx[nk] = i
        # In case cover duals are passed, also update the node weights
        if cover_duals is not None:
            nres[nk] += cover_duals[i]

    # Round all node and arc weights
    nres = {k: round(v / prec) for k, v in nres.items()}
    ares = {k: round(v / prec) for k, v in ares.items()}

    # Compute an bounds on the maximum weight per time step
    minwgt = min(nres.values()) + min(ares.values())
    maxwgt = max(nres.values()) + max(ares.values())

    # Quick access to some useful parameters
    eoh = tug['eoh'] # end of horizon
    mni = max(tug.vs['index_o']) # max node index in the original graph

    # Build a new model
    mdl = cp_model.CpModel()

    # Add the variables (two per time step)
    # sequence, the value -2 is for "blanks" on the right side
    x = {i: mdl.NewIntVar(-2, mni, f'x_{i}') for i in range(eoh)}
    c = {i: mdl.NewIntVar(minwgt, maxwgt, f'c_{i}') for i in range(1, eoh)}
    z = mdl.NewIntVar(minwgt * eoh, maxwgt * eoh, 'z')

    # Add the transition constraints
    for t in range(1, eoh):
        # Select all the arcs that start at time t
        edges = tug.es.select(time=t)
        # Build allowed tuples
        alw = []
        for e in edges:
            i, j = e['source_o'], e['target_o']
            # Compute the transition weight
            w = ares[t, i, j] + nres[t, j]
            if t == 1: w = nres[t-1, i]
            # Add the tuple
            alw.append((i, j, w))
        # Add the transitions to and from a blank
        for j in range(mni+1):
            # Transition from a blank
            alw.append((-1, j, nres[t, j]))
            # Transition to a blank
            if t == 1:
                alw.append((j, -2, nres[t-1, j]))
            else:
                alw.append((j, -2, 0))
        # Add the transitions to extend a blank
        alw.append((-1, -1, 0))
        alw.append((-2, -2, 0))
        # Build a table constraint
        mdl.AddAllowedAssignments([x[t-1], x[t], c[t]], alw)
    # Add the forbidden transitions
    if max_waits is not None:
        for t in range(max_waits, eoh):
            # Buid the forbidden tuples
            frb = [tuple([i] * (max_waits+1)) for i in range(mni+1)]
            # Build the scope
            scope = [x[h] for h in range(t-max_waits, t+1)]
            # Add the constraint
            mdl.AddForbiddenAssignments(scope, frb)
    # Add the objective
    mdl.Add(z == sum(c[i] for i in range(1, eoh)))
    if optimize:
        mdl.Minimize(z)
    # Add a bound on the objective
    if filter_paths:
        mdl.Add(z < -round(alpha / prec))

    # Build a solver
    slv = cp_model.CpSolver()
    # Set a time limit
    if time_limit is not None:
        slv.parameters.max_time_in_seconds = time_limit

    # Build a solution collector (and handle the path limit)
    slv.parameters.enumerate_all_solutions = True

    class Collector(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.paths = []
            self.costs = []
            # self.solutions = [] # for debugging

        def OnSolutionCallback(self):
            # Store the path
            path = [tug_idx[t, self.Value(x[t])] for t in range(eoh)
                    if self.Value(x[t]) >= 0]
            if len(path) > 0:
                self.paths.append(path)
                # Store the cost
                self.costs.append(self.Value(z) * prec + alpha)
            # # Store the solution
            # sol = [self.Value(x[i]) for i in range(eoh)]
            # self.solutions.append(sol)
            # Handle the solution limit
            if max_paths is not None and len(self.paths) > max_paths:
                self.StopSearch()

    collector = Collector()
    # Solve the model
    status = slv.SolveWithSolutionCallback(mdl, collector)
    closed = status in (cp_model.OPTIMAL, cp_model.INFEASIBLE)

    # Sort paths by increasing reduced cost
    sidx = np.argsort(collector.costs)
    res = [collector.paths[i] for i in sidx]
    res_c = [collector.costs[i] for i in sidx]
    # Return the results
    return res_c, res


def trajectory_extraction_cg(
        tug : ig.Graph,
        node_counts : dict,
        arc_counts : dict,
        max_iter : int,
        max_paths_per_iter : int = None,
        alpha : float = 0,
        min_vertex_cover : float = None,
        max_waits : int = None,
        pricing_time_limit : float = None,
        force_lcg_pricing : bool = False,
        verbose : int = 0,
        return_history : bool = False):
    """
    """
    assert(tug['unfolded'])
    assert(max_waits is None or max_waits >= 1)
    # Quick access to some useful parameters
    eoh = tug['eoh'] # end of horizon
    # Build an initial solution (one path per node)
    paths = [[v.index] for v in tug.vs]
    # Start the main loop
    sol = None # Incumbent solution
    sse_history = [] # Sum of squared errors
    for it in range(max_iter):
        # Build the master problem solver
        master = PathSelectionSolver(tug, node_counts, arc_counts,
                alpha=alpha, min_vertex_cover=min_vertex_cover)
        # Solve the master problem
        sol = master.solve(paths, verbose=0, polish=True)
        # Convert the solution to paths
        sol_flows, sol_paths = master.sol_to_paths()
        # Compute the error (just for tracking)
        sse = get_reconstruction_error(tug, sol_flows, sol_paths,
                node_counts, arc_counts)
        sse_history.append(sse)
        # Obtain Lagrangian multipliers
        if min_vertex_cover is not None:
            cover_duals = master.get_cover_duals()
        else:
            cover_duals = None
        # Solve the pricing problem
        if (max_waits is None or max_waits >= eoh) and not force_lcg_pricing:
            npc, np = solve_pricing_problem(tug, sol_flows, sol_paths,
                    node_counts, arc_counts,
                    alpha=alpha, cover_duals=cover_duals)
            # Limit the number of new paths, if needed
            if max_paths_per_iter is not None:
                np = np[:max_paths_per_iter]
        else:
            npc, np = solve_pricing_problem_maxwaits(tug, sol_flows, sol_paths,
                    node_counts, arc_counts,
                    alpha=alpha, cover_duals=cover_duals,
                    max_waits=max_waits, max_paths=max_paths_per_iter,
                    time_limit=pricing_time_limit)
        # Update the path pool (discarding duplicates)
        # NOTE this is necessary since sometimes the polishing step fails
        # and therefore we need to deal with an approximate solution
        nold = len(paths)
        old_as_set = set([tuple(p) for p in paths])
        found_as_set = set([tuple(p) for p in np])
        new_as_set = old_as_set.union(found_as_set)
        paths = [list(p) for p in new_as_set]
        nnew = len(paths) - nold
        # Print some informtion
        if verbose > 0:
            print(f'It.{it}, sse: {sse:.2f}, #paths: {len(paths)}, new: {nnew}')
        # Check termination condition
        if nnew == 0:
            break
        # print('PATHS', sorted(paths))
    # Return the solution
    res = [sol_flows, sol_paths]
    if return_history:
        res.append(sse_history)
    return res


def get_reconstruction_error(
        tug : ig.Graph,
        flows : list,
        paths : list,
        node_counts : dict,
        arc_counts : dict):
    """
    """
    assert(tug['unfolded'])
    # Compute the residuals
    nres, ares = _get_residuals(tug, flows, paths, node_counts, arc_counts)
    # Compute the reconstruction error
    return sum(nres[n]**2 for n in nres) + sum(ares[a]**2 for a in ares)


def get_default_benchmark_graph(
        nnodes : int,
        eoh : int,
        seed : int = None,
        sigma : float = None):
    g = build_website_graph(nnodes=nnodes, rate=3, extra_arc_fraction=0.25, seed=seed)
    flows, paths = build_random_paths(g, min_paths=3, max_paths=5,
                                           min_units=1, max_units=10, eoh=eoh, seed=seed)
    tug = build_time_unfolded_graph(g, eoh=eoh)
    nc, ac = get_counts(tug, flows, paths)
    if sigma is not None:
        nc, ac = add_proportional_noise(nc, ac, sigma=sigma)
    return g, tug, flows, paths, nc, ac
