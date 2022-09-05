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
import numpy as np
import tensorflow as tf
from datanetAPI import DatanetAPI  # This API may be different for different versions of the dataset

POLICIES = np.array(['WFQ', 'SP', 'DRR', 'FIFO'])


def sample_to_dependency_graph(sample):
    G = nx.DiGraph(sample.get_topology_object())
    R = sample.get_routing_matrix()
    T = sample.get_traffic_matrix()
    P = sample.get_performance_matrix()

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                D_G.add_node('p_{}_{}'.format(src, dst),
                             traffic=T[src, dst]['Flows'][0]['AvgBw'],
                             packets=T[src, dst]['Flows'][0]['PktsGen'],
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
                        map = [['0'], ['1'], ['2']]
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


def generator(data_dir, shuffle=False, topology_size=None):
    tool = DatanetAPI(data_dir, [], shuffle)
    it = iter(tool)
    for sample in it:

        D_G, n_q, n_p, n_l = sample_to_dependency_graph(sample)

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


def transformation(x, y):
    """Apply a transformation over all the samples included in the dataset.

        Args:
            x (dict): predictor variable.
            y (array): target variable.

        Returns:
            x,y: The modified predictor/target variables.
        """

    traffic_mean = 661.4106739200497
    traffic_sdv = 422.016325882212
    packets_mean = 0.6614085793154538
    packets_sdv = 0.42200425613185977
    capacity_mean = 25315.25903714809
    capacity_sdv = 16190.490914340508

    """traffic_mean = 601.1497144409911
    traffic_sdv = 386.07829997027466
    packets_mean = 0.6011599099332077
    packets_sdv = 0.38607350642252247
    capacity_mean = 47318.97140832356
    capacity_sdv = 41888.99727626712"""

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_sdv

    x["packets"] = (x["packets"] - packets_mean) / packets_sdv

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_sdv

    y = tf.math.log(y)

    return x, y


def input_fn(data_dir, transform=True, repeat=True, shuffle=False, take=None, topology_size=None):
    """This function uses the generator function in order to create a Tensorflow dataset

        Args:
            data_dir (string): Path of the data directory.
            transform (bool): If true, the data is transformed using the transformation function.
            repeat (bool): If true, the data is repeated. This means that, when all the data has been read,
                            the generator starts again.
            shuffle (bool): If true, the data is shuffled before being processed.
            take (integer): Number of elements to take of the dataset (If none, all the dataset is took).

        Returns:
            tf.data.Dataset: Containing a tuple where the first value are the predictor variables and
                             the second one is the target variable.
        """

    ds = tf.data.Dataset.from_generator(
        lambda: generator(data_dir=data_dir, shuffle=shuffle, topology_size=topology_size),
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

    if topology_size:
        def filter_fn(x, y):
            return tf.math.equal(x['n_paths'], topology_size * (topology_size - 1))

        ds = ds.filter(filter_fn)

    if transform:
        ds = ds.map(lambda x, y: transformation(x, y))

    if repeat:
        ds = ds.repeat()

    if take:
        ds = ds.take(take)

    # ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds
