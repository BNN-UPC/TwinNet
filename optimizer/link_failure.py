"""
   Copyright 2021 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import networkx as nx
import configparser
import numpy as np
import random
import tensorflow as tf
from statistics import mean
import os
import pickle
import sys

sys.path.insert(1, "../GNN/")
from model import model_fn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONFIG = configparser.ConfigParser()
CONFIG._interpolation = configparser.ExtendedInterpolation()
CONFIG.read('../GNN/config.ini')
SLA = [0.6, 1, float('inf')]
POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])
MAX_NUM_QUEUES = 5

def transformation(x, y):
    traffic_mean = 661.045
    traffic_sdv = 419.19
    packets_mean = 0.661
    packets_sdv = 0.419
    capacity_mean = 25495.603
    capacity_sdv = 16228.992

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_sdv

    x["packets"] = (x["packets"] - packets_mean) / packets_sdv

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_sdv

    y = tf.math.log(y)

    return x, y


def sample_to_dependency_graph(sample, intensity, R=None):
    G = nx.DiGraph(sample.get_topology_object())
    if R is None:
        R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                D_G.add_node('p_{}_{}'.format(src, dst),
                             traffic=(T[src, dst]['Flows'][0]['AvgBw'] / sample.maxAvgLambda) * intensity,
                             packets=(T[src, dst]['Flows'][0]['PktsGen'] / sample.maxAvgLambda) * intensity,
                             tos=int(T[src, dst]['Flows'][0]['ToS']),
                             source=src,
                             destination=dst,
                             delay=float(P[src, dst]['AggInfo']['AvgDelay']))

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 capacity=G.edges[src, dst]['bandwidth'],
                                 policy=np.where(G.nodes[src]['schedulingPolicy'] == POLICIES)[0][0])
                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                    q_s = str(G.nodes[h_1]['queueSizes']).split(',')
                    # policy = G.nodes[h_1]['schedulingPolicy']
                    if 'schedulingWeights' in G.nodes[h_1]:
                        q_w = str(G.nodes[h_1]['schedulingWeights']).split(',')
                    else:
                        q_w = ['-']
                    if 'tosToQoSqueue' in G.nodes[h_1]:
                        map = [m.split(',') for m in str(G.nodes[h_1]['tosToQoSqueue']).split(';')]
                    else:
                        map = [[1], [2], [3]]
                    q_n = 0
                    for q in range(G.nodes[h_1]['levelsQoS']):
                        D_G.add_node('q_{}_{}_{}'.format(h_1, h_2, q),
                                     size=int(q_s[q]),
                                     priority=q_n,
                                     weight=float(q_w[q]) if q_w[0] != '-' else 0)
                        D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'q_{}_{}_{}'.format(h_1, h_2, q))
                        if str(int(T[src, dst]['Flows'][0]['ToS'])) in map[q]:
                            D_G.add_edge('p_{}_{}'.format(src, dst), 'q_{}_{}_{}'.format(h_1, h_2, q))
                            D_G.add_edge('q_{}_{}_{}'.format(h_1, h_2, q), 'p_{}_{}'.format(src, dst))
                        q_n += 1

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    n_q = 0
    n_p = 0
    n_l = 0
    mapping = {}
    for entity in list(D_G.nodes()):
        if entity.startswith('q'):
            mapping[entity] = ('q_{}'.format(n_q))
            n_q += 1
        elif entity.startswith('p'):
            mapping[entity] = ('p_{}'.format(n_p))
            n_p += 1
        elif entity.startswith('l'):
            mapping[entity] = ('l_{}'.format(n_l))
            n_l += 1

    D_G = nx.relabel_nodes(D_G, mapping)
    return D_G, n_q, n_p, n_l


def generator(sample, intenisty, comb_routing, comb_scheduling):
    it = 0
    for routing in comb_routing:
        D_G, n_q, n_p, n_l = sample_to_dependency_graph(sample, intenisty, routing)

        link_to_path = np.array([], dtype='int32')
        queue_to_path = np.array([], dtype='int32')
        l_p_s = np.array([], dtype='int32')
        l_q_p = np.array([], dtype='int32')
        path_ids = np.array([], dtype='int32')
        for i in range(n_p):
            l_s_l = 0
            q_s_l = 0
            for elem in D_G['p_{}'.format(i)]:
                if elem.startswith('l_'):
                    link_to_path = np.append(link_to_path, int(elem.replace('l_', '')))
                    l_s_l += 1
                elif elem.startswith('q_'):
                    queue_to_path = np.append(queue_to_path, int(elem.replace('q_', '')))
                    q_s_l += 1
            path_ids = np.append(path_ids, [i] * q_s_l)
            l_p_s = np.append(l_p_s, range(l_s_l))
            l_q_p = np.append(l_q_p, range(q_s_l))

        path_to_queue = np.array([], dtype='int32')
        sequence_queues = np.array([], dtype='int32')
        for i in range(n_q):
            seq_len = 0
            for elem in D_G['q_{}'.format(i)]:
                path_to_queue = np.append(path_to_queue, int(elem.replace('p_', '')))
                seq_len += 1
            sequence_queues = np.append(sequence_queues, [i] * seq_len)

        queue_to_link = np.array([], dtype='int32')
        sequence_links = np.array([], dtype='int32')
        l_q_l = np.array([], dtype='int32')
        for i in range(n_l):
            seq_len = 0
            for elem in D_G['l_{}'.format(i)]:
                queue_to_link = np.append(queue_to_link, int(elem.replace('q_', '')))
                seq_len += 1
            sequence_links = np.append(sequence_links, [i] * seq_len)
            l_q_l = np.append(l_q_l, range(seq_len))

        if -1 in list(nx.get_node_attributes(D_G, 'delay').values()):
            continue

        if it % 500 == 0:
            print("GENERATED SAMPLE: {}".format(it))
        it += 1

        yield {"traffic": list(nx.get_node_attributes(D_G, 'traffic').values()),
               "packets": list(nx.get_node_attributes(D_G, 'packets').values()),
               "capacity": list(nx.get_node_attributes(D_G, 'capacity').values()),
               "size": list(nx.get_node_attributes(D_G, 'size').values()),
               "policy": list(nx.get_node_attributes(D_G, 'policy').values()),
               "priority": list(nx.get_node_attributes(D_G, 'priority').values()),
               "weight": list(nx.get_node_attributes(D_G, 'weight').values()),
               "link_to_path": link_to_path,
               "queue_to_path": queue_to_path,
               "path_to_queue": path_to_queue,
               "queue_to_link": queue_to_link,
               "sequence_queues": sequence_queues,
               "sequence_links": sequence_links,
               "path_ids": path_ids,
               "l_p_s": l_p_s,
               "l_q_p": l_q_p,
               "l_q_l": l_q_l,
               "n_queues": n_q,
               "n_links": n_l,
               "n_paths": n_p,
               }, list(nx.get_node_attributes(D_G, 'delay').values())


def input_fn(sample, intenisty, comb_routing, comb_scheduling, transform=True, repeat=True, take=None):
    ds = tf.data.Dataset.from_generator(
        lambda: generator(sample=sample, intenisty=intenisty, comb_routing=comb_routing,
                          comb_scheduling=comb_scheduling),
        ({"traffic": tf.float32, "packets": tf.float32,
          "capacity": tf.float32,
          "size": tf.float32, "policy": tf.int32,
          "priority": tf.int32,
          "weight": tf.float32, "link_to_path": tf.int32,
          "queue_to_path": tf.int32, "path_to_queue": tf.int32,
          "queue_to_link": tf.int32, "sequence_queues": tf.int32,
          "sequence_links": tf.int32, "path_ids": tf.int32,
          "l_p_s": tf.int32, "l_q_p": tf.int32,
          "l_q_l": tf.int32,
          "n_queues": tf.int32, "n_links": tf.int32,
          "n_paths": tf.int32},
         tf.float32),
        ({"traffic": tf.TensorShape([None]), "packets": tf.TensorShape([None]),
          "capacity": tf.TensorShape([None]),
          "size": tf.TensorShape([None]), "policy": tf.TensorShape([None]),
          "priority": tf.TensorShape([None]),
          "weight": tf.TensorShape([None]), "link_to_path": tf.TensorShape([None]),
          "queue_to_path": tf.TensorShape([None]), "path_to_queue": tf.TensorShape([None]),
          "queue_to_link": tf.TensorShape([None]), "sequence_queues": tf.TensorShape([None]),
          "sequence_links": tf.TensorShape([None]), "path_ids": tf.TensorShape([None]),
          "l_p_s": tf.TensorShape([None]), "l_q_p": tf.TensorShape([None]),
          "l_q_l": tf.TensorShape([None]),
          "n_queues": tf.TensorShape([]), "n_links": tf.TensorShape([]),
          "n_paths": tf.TensorShape([])},
         tf.TensorShape([None])))
    if transform:
        ds = ds.map(lambda x, y: transformation(x, y))

    if repeat:
        ds = ds.repeat()

    if take:
        ds = ds.take(take)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def evaluate_solution(D_G, pred_delays):
    sat_tos = [[] for _ in range(int(CONFIG['DATASET']['num_tos']))]
    no_sat_tos = [[] for _ in range(int(CONFIG['DATASET']['num_tos']))]
    best_effort = []
    delays = [[] for _ in range(int(CONFIG['DATASET']['num_tos']))]
    id = 0
    for node, data in D_G.nodes(data=True):
        if node.startswith('p_'):
            if data['tos'] < len(sat_tos) - 1:
                # D_G.nodes[node]['predicted_delay'] = delays[id]
                if pred_delays[id] <= SLA[data['tos']]:
                    # D_G.nodes[node]['sla'] = True
                    sat_tos[data['tos']].append(node)
                else:
                    # D_G.nodes[node]['sla'] = False
                    no_sat_tos[data['tos']].append(node)
                delays[data['tos']].append(pred_delays[id])
            else:
                delays[2].append(pred_delays[id])
                best_effort.append(node)
            id += 1

    for it in range(len(best_effort)):
        if mean(delays[2]) < delays[2][it]:
            no_sat_tos[2].append(best_effort[it])
        else:
            sat_tos[2].append(best_effort[it])

    return sat_tos, no_sat_tos, delays


def compute_mean(a):
    means = []
    for elem in a:
        means.append(mean(elem))
    return mean(means)


def k_shortest_paths(G, source, target, k, weight=None):
    paths = []
    leng = -1
    for path in nx.shortest_simple_paths(G, source, target, weight=weight):
        if leng == -1:
            leng = len(path)
        if len(path) == leng or len(path) <= leng + k:
            paths.append(path)
        elif len(path) > leng + k:
            return paths
    return paths


MODEL_DIR = './logs/all_queues'
estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    params=CONFIG
)


def delete_one_link(G):
    remove = []
    for e in G.edges(data=True):
        G_aux = G.copy()
        G_aux.remove_edge(e[0], e[1])
        G_aux.remove_edge(e[1], e[0])
        if nx.is_strongly_connected(G_aux):
            remove.append(e)


    failure = random.choice(remove)
    G.remove_edge(failure[0], failure[1])
    G.remove_edge(failure[1], failure[0])

    """nx.draw_networkx(G)
    plt.show()"""

    return G


NO_SOL_IT = 2
COMB_PER_ITERATION = 1000
MAX_SAMPLES = 5000
df_to_concat = []
failures = 1
inten = 1900
NUM_SCENARIOS = 5
for scenario in range(1,NUM_SCENARIOS):
    with open('./scheduling/scheduling_1600.pkl', 'rb') as f:
        G = pickle.load(f)

    with open('./SAMPLE_FIFO_TOS_ROUTING.pkl', 'rb') as f:
        sample = pickle.load(f)
        sample._set_topology_object(G.copy())

    stop = False
    failures=1
    while not stop:

        print("NUM OF FAILURES {}".format(failures))
        K = 1
        G = delete_one_link(nx.DiGraph(sample.get_topology_object()))
        if G is None:
            break

        P = np.zeros((len(G), len(G)), dtype='object')
        routing = np.zeros((len(G), len(G)), dtype='object')
        for src in G.nodes():
            for dst in G.nodes():
                P[src][dst] = k_shortest_paths(G, src, dst, K)

        for src in G.nodes():
            for dst in G.nodes():
                routing[src][dst] = random.choice(P[src, dst])

        sample._set_topology_object(G.copy())
        D_G, n_q, n_p, n_l = sample_to_dependency_graph(sample, inten)

        comb_routing = [routing]
        pred_results = estimator.predict(input_fn=lambda: input_fn(
            sample,
            inten,
            comb_routing,
            None,
            transform=True,
            repeat=False))

        pred_delay = np.exp([pred['predictions'] for pred in pred_results])

        sat_tos, no_sat_tos, delays = evaluate_solution(D_G, pred_delay)


        for intensity in [inten]:

            it = 0
            it_no_sol = 0
            while True:
                if it_no_sol == NO_SOL_IT:
                    break
                print("ITERATION {}".format(it))
                print("{} NOT SATISFYING SLA 0: {}".format(len(no_sat_tos[0]), no_sat_tos[0]))
                print("{} NOT SATISFYING SLA 1: {}".format(len(no_sat_tos[1]), no_sat_tos[1]))

                comb_routing = []
                change = None
                if len(no_sat_tos[0]) != 0:
                    print("ITERATING OVER TOS = 0")
                    reward = no_sat_tos[0] + random.sample(no_sat_tos[1], int(len(no_sat_tos[1]) / 5))
                    penalize = random.sample(sat_tos[0], int(len(sat_tos[0]) / 5)) + random.sample(sat_tos[1], int(
                        len(sat_tos[1]) / 5)) + random.sample(sat_tos[2],
                                                              int(len(sat_tos[1]) / 10))
                elif len(no_sat_tos[1]) != 0:
                    print("ITERATING OVER TOS = 1")
                    reward = no_sat_tos[1]
                    penalize = random.sample(sat_tos[2], int(len(sat_tos[1]) / 5))
                else:
                    print("ITERATING OVER BEST EFFORT")
                    reward = no_sat_tos[2]
                    penalize = sat_tos[0] + sat_tos[1]

                if it_no_sol >= 2:
                    print("NO SOLUTION FOUND DURING {} ITERATIONS. STARTING PENALIZING...".format(it_no_sol))

                for _ in range(COMB_PER_ITERATION):
                    R_aux = np.copy(routing)

                    for path in reward:
                        src = D_G.nodes[path]['source']
                        dst = D_G.nodes[path]['destination']
                        # print("CHANGING SRC: {} DST: {}".format(src,dst))
                        R_aux[src, dst] = random.choice(P[src, dst])

                    if it_no_sol >= 2:
                        for path in penalize:
                            src = D_G.nodes[path]['source']
                            dst = D_G.nodes[path]['destination']
                            # print("CHANGING SRC: {} DST: {}".format(src,dst))
                            R_aux[src, dst] = random.choice(P[src, dst])

                    comb_routing.append(R_aux)

                pred_results = estimator.predict(input_fn=lambda: input_fn(
                    sample,
                    intensity,
                    comb_routing,
                    None,
                    transform=True,
                    repeat=False))

                pred_delay = np.exp([pred['predictions'] for pred in pred_results])

                splited_delay = np.array_split(pred_delay, COMB_PER_ITERATION)

                it_no_sol += 1
                for i in range(len(splited_delay)):
                    s_sat_tos, s_no_sat_tos, s_delays = evaluate_solution(D_G, splited_delay[i])

                    if len(s_no_sat_tos[0]) < len(no_sat_tos[0]):
                        print("FOUND BETTER SOLUTION 1: BEFORE {} AFTER {}".format(len(sat_tos[0]), len(s_sat_tos[0])))

                        sat_tos = s_sat_tos
                        no_sat_tos = s_no_sat_tos
                        delays = s_delays
                        routing = np.copy(comb_routing[i])
                        it_no_sol = 0
                    elif (len(s_no_sat_tos[0]) == len(no_sat_tos[0])) and (len(s_no_sat_tos[1]) < len(no_sat_tos[1])):
                        print("FOUND BETTER SOLUTION 2: BEFORE {} AFTER {}".format(len(sat_tos[1]), len(s_sat_tos[1])))

                        sat_tos = s_sat_tos
                        no_sat_tos = s_no_sat_tos
                        delays = s_delays
                        routing = np.copy(comb_routing[i])
                        it_no_sol = 0
                    elif (len(s_no_sat_tos[0]) == len(no_sat_tos[0])) and (len(s_no_sat_tos[1]) == len(no_sat_tos[1])) and (
                            mean(s_delays[2]) < mean(delays[2])):
                        print("FOUND BETTER SOLUTION 3: BEFORE {} AFTER {}".format(mean(delays[2]), mean(s_delays[2])))

                        sat_tos = s_sat_tos
                        no_sat_tos = s_no_sat_tos
                        delays = s_delays
                        routing = np.copy(comb_routing[i])
                        it_no_sol = 0

                print("CURRENT SOLUTION")
                print("SATISFIED SLA 0: {}".format(len(sat_tos[0])))
                print("SATISFIED SLA 1: {}".format(len(sat_tos[1])))
                print("MEAN DELAY BEST EFFORT: {}".format(mean(delays[2])))
                it += 1

            print("BEST SOLUTION FOUND FOR INTENSITY {}".format(intensity))
            print("SATISFIED SLA 0: {}".format(len(sat_tos[0])))
            print("SATISFIED SLA 1: {}".format(len(sat_tos[1])))
            print("MEAN DELAY BEST EFFORT: {}".format(mean(delays[2])))

            with open('./optimizer/link_failure/routing_{}_{}.pkl'.format(scenario, failures), 'wb') as f:
                pickle.dump(routing, f, pickle.HIGHEST_PROTOCOL)
            with open('./optimizer/link_failure/topology_{}_{}.pkl'.format(scenario, failures), 'wb') as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
            with open('./optimizer/link_failure/delays_{}_{}.pkl'.format(scenario, failures), 'wb') as f:
                pickle.dump(delays, f, pickle.HIGHEST_PROTOCOL)


        stop = not (len(no_sat_tos[0]) == 0 and len(no_sat_tos[1]) == 0)
        failures += 1
