import os
import time

import numpy as np
import pandas as pd

PATH = "../benchmarks"
ANS_PATH = "../solutions"


def get_number_after(string, key_word, delimiter):
    num_digits = 1
    if string.find(key_word) != -1:
        starting_ind = index = string.find(key_word) + len(key_word)
        index = starting_ind
        while string[index + 1] != delimiter:
            num_digits += 1
            index += 1
        return int(string[starting_ind:starting_ind + num_digits])
    else:
        return None


class AntVrp:
    def __init__(self, file_path, num_ants=10, max_iter_num=10, alpha=1.0, beta=1, phi=0.05):
        self.paths = None
        fin = open(file_path)
        while 1:
            line = fin.readline()
            if "COMMENT" in line:
                self.num_trucks = get_number_after(line, "No of trucks: ", ',')
                self.optimal_value = get_number_after(line, "Optimal value: ", ')')
            if "DIMENSION" in line:
                self.dimension = get_number_after(line, "DIMENSION : ", '\n')
            if "CAPACITY" in line:
                self.truck_capacity = get_number_after(line, "CAPACITY : ", '\n')
            if "NODE_COORD_SECTION" in line:
                break

        node_coords = []
        for i in range(self.dimension):
            data = list(map(int, fin.readline().split()))
            node_coords.append((data[1], data[2]))

        self.adj_mat = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                self.adj_mat[i][j] = ((node_coords[i][0] - node_coords[j][0]) ** 2 + (
                        node_coords[i][1] - node_coords[j][1]) ** 2) ** (1 / 2)
                if self.adj_mat[i][j] == 0:
                    self.adj_mat[i][j] = 0.001

                self.adj_mat[j][i] = self.adj_mat[i][j]

        fin.readline()
        self.demands = []
        for i in range(self.dimension):
            data = list(map(int, fin.readline().split()))
            self.demands.append(data[1])
            if data[1] == 0:
                self.depot_position = i
        fin.close()

        self.pheromones = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                self.pheromones[i][j] = 0.2
                self.pheromones[j][i] = 0.2

        self.num_ants = num_ants
        self.max_iter_num = max_iter_num
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.visited_nodes = []
        self.best_cost = float('INF')
        self.best_cost_in_iteration = float('INF')
        self.best_paths = []
        self.best_paths_in_iteration = []

    def choose_next_vert(self, current_vert, cargo_left):
        denominator = 0
        candidates = []
        candidates_prob = []

        for i in range(self.dimension):
            if i != current_vert and i not in self.visited_nodes and cargo_left - self.demands[
                i] >= 0 and i != self.depot_position:
                denominator += (self.pheromones[current_vert][i] ** self.alpha) * (
                        (1 / self.adj_mat[current_vert][i]) ** self.beta)

        if denominator > 0:
            for i in range(self.dimension):
                if i != current_vert and i not in self.visited_nodes and cargo_left - self.demands[
                    i] >= 0 and i != self.depot_position:
                    probability = ((self.pheromones[current_vert][i] ** self.alpha) * (
                            (1 / self.adj_mat[current_vert][i]) ** self.beta)) / denominator
                    candidates.append(i)
                    candidates_prob.append(probability)

            return np.random.choice(candidates, p=candidates_prob)
        else:
            return -1

    def step_pheromone_update(self, prev_vert, curr_vert):
        del_pher = 0.2 / self.adj_mat[prev_vert][curr_vert]

        self.pheromones[prev_vert][curr_vert] = (1 - self.phi) * self.pheromones[prev_vert][
            curr_vert] + self.phi * del_pher
        self.pheromones[curr_vert][prev_vert]

    def global_pheromone_update(self):
        for truck in range(len(self.best_paths_in_iteration)):
            for vert in range(len(self.best_paths_in_iteration[truck]) - 1):
                del_pher = self.truck_capacity / self.best_cost_in_iteration
                i = self.best_paths_in_iteration[truck][vert]
                j = self.best_paths_in_iteration[truck][vert + 1]

                self.pheromones[i][j] = (1 - self.phi) * self.pheromones[i][j] + self.phi * del_pher
                self.pheromones[j][i]

    def calc_cost(self):
        total_cost = 0
        for truck in range(len(self.paths)):
            for vert in range(len(self.paths[truck]) - 1):
                total_cost += self.adj_mat[self.paths[truck][vert]][self.paths[truck][vert + 1]]

        return total_cost

    def two_opt(self, route):
        best_route = route
        improved = True
        while improved:
            improved = False
            best_distance = self.route_distance(best_route)
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_distance = self.route_distance(new_route)
                    if new_distance < best_distance:
                        best_route = new_route
                        improved = True
                        best_distance = new_distance
        return best_route

    def route_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.adj_mat[route[i]][route[i + 1]]
        return distance

    def solve(self):
        start_time = time.time()
        for iter_num in range(self.max_iter_num):
            self.best_cost_in_iteration = float('INF')
            for ant in range(self.num_ants):
                self.visited_nodes = []
                self.paths = []

                for truck in range(self.num_trucks):
                    current_position = self.depot_position
                    cargo_left = self.truck_capacity
                    self.paths.append([])
                    self.paths[truck].append(current_position)

                    while True:
                        old_position = current_position
                        current_position = self.choose_next_vert(current_position, cargo_left)
                        if current_position == -1:
                            break
                        self.paths[truck].append(current_position)
                        self.visited_nodes.append(current_position)
                        cargo_left -= self.demands[current_position]
                        self.step_pheromone_update(old_position, current_position)

                    self.paths[truck].append(self.depot_position)
                    self.step_pheromone_update(old_position, self.depot_position)

                for truck in range(self.num_trucks):
                    self.paths[truck] = self.two_opt(self.paths[truck])

                node_count = sum(len(truck_path) for truck_path in self.paths)
                if node_count - 2 * len(self.paths) + 1 == self.dimension:
                    curr_ant_cost = self.calc_cost()
                    if curr_ant_cost < self.best_cost_in_iteration:
                        self.best_paths_in_iteration = self.paths
                        self.best_cost_in_iteration = curr_ant_cost

            self.global_pheromone_update()
            if self.best_cost_in_iteration < self.best_cost:
                self.best_cost = self.best_cost_in_iteration
                self.best_paths = self.best_paths_in_iteration

        return self.optimal_value, self.best_cost, self.best_paths, time.time() - start_time


class Benchmarks:

    def __init__(self, num_ants=1, max_iter_num=1, alpha=1.0, beta=1.0, phi=0.5, times_repeat=10, verbose=False):
        self.verbose = verbose
        self.num_ants = num_ants
        self.max_iter_num = max_iter_num
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.times_repeat = times_repeat

    def check_single_benchmark(self, file_path, times_repeat=10):
        best_cost = float('INF')
        best_paths = None
        total_time = 0

        for i in range(times_repeat):
            algorithm = AntVrp(file_path, num_ants=self.num_ants, max_iter_num=self.max_iter_num, alpha=self.alpha,
                               beta=self.beta, phi=self.phi)
            solution, cost, paths, time = algorithm.solve()

            if cost < best_cost:
                best_cost = cost
                best_paths = paths
            total_time += time

        return solution, best_cost, best_paths, total_time / times_repeat

    def report(self):
        results = []
        folders = [f.path for f in os.scandir(PATH) if f.is_dir()]

        for folder in folders:
            files = [f.path for f in os.scandir(folder) if f.is_file()]
            for file in files:
                if self.verbose:
                    print("Working on", file)

                solution, best_cost, best_paths, time = self.check_single_benchmark(file, self.times_repeat)

                if self.verbose:
                    print(solution, best_cost, time)

                benchmark = os.path.splitext(os.path.basename(file))[0]
                result = {
                    'benchmark': benchmark,
                    'epochs': self.max_iter_num,
                    'n_ants': self.num_ants,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'rho': self.phi,
                    'init_pher': 1000,
                    'mean_time': time,
                    'found_cost': best_cost,
                    'opt_cost': solution
                }
                results.append(result)

                folder_name = os.path.basename(os.path.dirname(file))
                file_name = os.path.splitext(os.path.basename(file))[0]

                with open(ANS_PATH + '/' + folder_name + '/' + file_name + ".sol", "w") as ans_file:
                    for truck in range(len(best_paths)):
                        route_without_depot = [node + 1 for node in best_paths[truck] if node != 0]
                        ans_file.write("Route #%d: " % (truck + 1) + " ".join(map(str, route_without_depot)))
                        ans_file.write("\n")
                    ans_file.write("cost %d" % best_cost)
                    ans_file.write("\n")

        return pd.DataFrame(results)


bench = Benchmarks(num_ants=8, max_iter_num=100, alpha=0.75, beta=5, phi=0.1, times_repeat=10, verbose=True)
results_df = bench.report()

results_df.to_csv("Results.csv")
