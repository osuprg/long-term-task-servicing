import networkx as nx    

def generate_graph(graph_generator_type, filepath, filename, max_rooms, rooms, max_traversal_cost, distance_scaling):
    if graph_generator_type == 'read':
        g = read_graph_from_file(filepath+filename, distance_scaling)
        return g, rooms
    elif graph_generator_type == 'simple_hallway':
        g, rooms = generate_simple_hallway(max_rooms, max_traversal_cost)
        return g, rooms
    elif graph_generator_type == 'simple_floors':
        g, rooms = generate_simple_floors(max_rooms, max_traversal_cost)
        return g, rooms
    else:
        raise ValueError(graph_generator_type)

def read_graph_from_file(input_filename, distance_scaling):
    g = nx.Graph()

    # read known connections
    lines = [line.rstrip('\n') for line in open(input_filename)]
    for line in lines:
        line = line.split()
        node_a = line[0]
        node_b = line[1]
        dist = float(line[2])
        dist = max(int(round(float(dist)/60)), 1)           # convert seconds to minutes
        dist = distance_scaling*dist

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


def generate_simple_hallway(max_rooms, traversal_cost):
    g = nx.Graph()
    old_node = "R" + str(1)
    rooms = []
    g.add_node(old_node)
    for i in range(max_rooms):
        new_node = "R" + str(i+2)
        g.add_node(new_node)
        g.add_edge(old_node, new_node)
        g.add_edge(new_node, old_node)
        g[old_node][new_node]['weight'] = traversal_cost
        g[new_node][old_node]['weight'] = traversal_cost
        rooms.append(new_node)
        old_node = new_node

    return g, rooms

def generate_simple_floors(max_rooms, traversal_cost):
    g = nx.Graph()
    num_rooms_per_floor = 4

    num_floors = int(max_rooms/num_rooms_per_floor)
    node_index = 1
    old_node = "R" + str(node_index)
    node_index += 1
    g.add_node(old_node)

    rooms = []
    # floor landing
    for i in range(num_floors):
        new_node = "R" + str(node_index)
        node_index += 1
        g.add_node(new_node)

        g.add_edge(old_node, new_node)
        g.add_edge(new_node, old_node)
        g[old_node][new_node]['weight'] = traversal_cost
        g[new_node][old_node]['weight'] = traversal_cost
        old_node = new_node

        # rooms adjacent to floor landing
        for j in range(num_rooms_per_floor):
            new_node = "R" + str(node_index)
            node_index += 1
            g.add_node(new_node)

            g.add_edge(old_node, new_node)
            g.add_edge(new_node, old_node)
            g[old_node][new_node]['weight'] = 1
            g[new_node][old_node]['weight'] = 1
            rooms.append(new_node)

    return g, rooms