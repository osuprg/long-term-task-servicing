import networkx as nx    

def generate_graph(graph_generator_type, filepath, filename, max_rooms, max_traversal_cost):
    if graph_generator_type == 'read':
        g = read_graph_from_file(filepath+filename)
        return g
    elif graph_generator_type == 'simple_hallway':
        g = generate_simple_hallway(max_rooms, max_traversal_cost)
    else:
        raise ValueError(graph_generator_type)

def read_graph_from_file(input_filename):
    g = nx.Graph()

    # read known connections
    lines = [line.rstrip('\n') for line in open(input_filename)]
    for line in lines:
        line = line.split()
        node_a = line[0]
        node_b = line[1]
        dist = float(line[2])
        dist = max(int(round(float(dist)/60)), 1)           # convert seconds to minutes

        # add nodes
        if not(g.has_node(node_a)):
            g.add_node(node_a)
        if not(g.has_node(node_b)):
            g.add_node(node_b)

        # add edges
        g.add_edge(node_a, node_b)
        g.add_edge(node_b, node_a)
        g[node_a][node_b]['weight'] = dist
        g[node_b][node_a]['weight'] = dist

    return g


def generate_simple_hallway(num_nodes, traversal_cost):
    g = nx.Graph()
    old_node = str(0)
    g.add_node(old_node)
    for i in range(num_nodes)[1:]:
        new_node = "R" + str(i+1)
        g.add_node(new_node)
        g.add_edge(old_node, new_node)
        g.add_edge(new_node, old_node)
        g[old_node][new_node]['weight'] = traversal_cost
        g[new_node][old_node]['weight'] = traversal_cost
        old_node = new_node
    return g