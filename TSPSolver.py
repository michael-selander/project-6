#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))



import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
try:
	import Queue as Q  # ver. < 3.0
except ImportError:
	import queue as Q

TOURNAMENT_SIZE  =   40
POPULATION_SIZE  =   300
MATING_POOL_SIZE =   40
MUTATION_RATE    =   0.1

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	# The greedy solution runs in O(n^2) time and
	# O(n) space. This is because it iterates through every city
	# (n) as a starting point and then iterates through every other city to find the
	# least cost node
	# it is only O(n) space, because it only ever saves the best path of n nodes
	# Time:  O(n^2)
	# Space: O(n)
	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		count = 0
		start_time = time.time()
		bestGreedySolution = TSPSolution()

		for i in range(len(cities)):
			partialPath =[]
			partialPath.append(cities[i])
			while len(partialPath) < len(cities):
				# create a random permutation
				last = partialPath[len(partialPath)-1]

				lowestDist = math.inf
				lowestNode = None
				for c in cities:
					if c in partialPath:
						continue
					if last.costTo(c) < lowestDist:
						lowestNode = c
						lowestDist = last.costTo(c)
				if lowestNode is None:
					break
				partialPath.append(lowestNode)

			if len(partialPath) < len(cities):
				bssf = TSPSolution()
			else:
				bssf = TSPSolution(partialPath)

			if (bssf.cost < bestGreedySolution.cost):
				bestGreedySolution = bssf

		end_time = time.time()
		results['cost'] = bestGreedySolution.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bestGreedySolution
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
		pass
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	# Time:  O(n^2)
	# Space: O(n^2)
	# As you can see, a 2d array of size n x n i initialized. This takes up
	# n x n space and takes n x n time, therefore space and time are both
	# n squared
	def generateInitialCostMatrix(self, cities):
		n = len(cities)
		costMatrix = np.array([[math.inf] * n for _ in range(n)])
		for i in range(n):
			for j in range(n):
				costMatrix[i][j] = cities[i].costTo(cities[j])

		return costMatrix


	# Because the matrix is passed in, we aren't actually creating any new space,
	# therefore the Space complexity is constant. As you can see, we have two for
	# loops that each iterate through the entire n x n matrix, meaning that the
	# time complexity is 2 * (n*n) which reduces to just big-O of n squared
	# Time:  O(n^2)
	# Space: O(1)
	def reduceCostMatrix(self, matrix):
		reduceCost = 0

		# row reduction
		for i in range(len(matrix)):
			minimum = min(matrix[i])
			if minimum == 0 or minimum == math.inf: continue
			reduceCost += minimum
			for j in range(len(matrix)):
				if matrix[i][j] == math.inf:
					continue
				matrix[i][j] -= minimum

		# column reduction
		for j in range(len(matrix)):
			column = matrix[:, j]
			minimum = min(column)
			if minimum == 0 or minimum == math.inf: continue
			reduceCost += minimum
			for i in range(len(matrix)):
				if matrix[i][j] == math.inf:
					continue
				matrix[i][j] -= minimum

		return matrix, reduceCost

	# This funtion expands a state into in the worst case scenario n-1 states
	# for each new state that is created, we must create the n by n cost matrix
	# and reduce the cost matrix. Therefore, both time and space for each matrix is
	# n squared. This is done n-1 times. a discussion of the overall complexity in relation
	# to this function is disussed in the 'branchAndBound' function
	# Time:  O((n-1)*n^2)
	# Space: O((n-1)*n^2)
	def expandState(self, state, cities):
		new_states = []
		originCity = state.partialPath[len(state.partialPath)-1]
		originIndex = cities.index(originCity)
		for j in range(len(state.costMatrix)):
			if j == originIndex:
				continue
			if state.costMatrix[originIndex][j] == math.inf:
				continue
			if cities[j] in state.partialPath:
				continue

			new_state = SearchState(np.copy(state.costMatrix), state.partialPath[:], state.bound)

			new_state.bound += new_state.costMatrix[originIndex, j]

			new_state.costMatrix[originIndex, :] = math.inf
			new_state.costMatrix[:, j] = math.inf
			new_state.costMatrix[originIndex, j] = math.inf
			new_state.costMatrix[j, originIndex] = math.inf

			new_state.costMatrix, added_bound = self.reduceCostMatrix(new_state.costMatrix)

			new_state.bound += added_bound

			new_state.partialPath.append(cities[j])

			new_states.append(new_state)

		return new_states




	def branchAndBound( self, time_allowance=60.0 ):
		# Initialize variables (Constant time and space)
		max_queue_size = 0
		total_states = 0
		pruned_states = 0
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = TSPSolution()

		# Generating the initial cost matrix, as well as reducing the cost matrix,
		# are both O(n^2)
		# Time:  O(n^2)
		# Space: O(n^2)
		initialCostMatrix = self.generateInitialCostMatrix(cities)
		reducedMatrix, bound = self.reduceCostMatrix(initialCostMatrix)
		initialState = SearchState(reducedMatrix, [cities[0]], bound)

		# My Priority queue class that I uses utilizes heapq, which is a binary queue
		# that takes O(log n) time for put and pop.
		# See source code for details
		# https://github.com/python/cpython/blob/3.8/Lib/heapq.py

		# Time:  O(log n)
		# Space: O(1)
		q = Q.PriorityQueue()
		q.put(initialState)

		# The greedy solution runs in O(n^2) time and
		# O(n) space. See the self.greedy() function for details
		# Time:  O(n^2)
		# Space: O(n)
		greedySolution = self.greedy(time_allowance)['soln']
		bssf = greedySolution

		start_time = time.time()

		# This is where the actual branching and bounding takes place.
		# If we assume worst case scenario for big-O, then we assume that no
		# pruning takes place. This means that the tree will be expanded n! times!
		# If it is expanded n! times, and each state takes n^2, then the overall
		# space and time complexity is
		# Time:  O(n! * n^2)
		# Space: O(n! * n^2)
		# This however assumes there will be no pruning, which will rarely be the case
		# If we account for pruning, and assume that 'b' is equal to the average number
		# of states that each node is expanded to, the run time becomes
		# Time:  O(b^n * n^2)
		# Space: O(b^n * n^2)
		while not q.empty() and time.time()-start_time < time_allowance:
			if max_queue_size < q.qsize():
				max_queue_size = q.qsize()
			iState = q.get()
			if iState.bound > bssf.cost:
				pruned_states += 1
				continue
			new_states = self.expandState(iState, cities)
			total_states += len(new_states)
			for state in new_states:
				if len(state.partialPath) == len(cities):
					solution = TSPSolution(state.partialPath)
					if solution.cost < bssf.cost:
						bssf = solution
						count += 1
				elif state.bound < bssf.cost:
					q.put(state)
				else:
					pruned_states += 1

		#This just empties the queue if there is still something there
		# (i.e. if the time allowance is up)
		while not q.empty():
			s = q.get()
			pruned_states += 1

		end_time = time.time()
		results['cost'] = bssf.cost if True else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_queue_size
		results['total'] = total_states
		results['pruned'] = pruned_states
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''



	def select_best_of(self, population, tournament_size):
		tournament = random.sample(population, tournament_size);
		best = TSPSolution();


		for t in tournament:
			if not t:
				continue
			if t.cost < best.cost:
				best = t

		if best.cost == np.inf:
			return self.get_random_solution()

		return best


	def select_mating_pool(self, population, pool_size):
		mating_pool = []
		while len(mating_pool) < pool_size:
			tournament_winner = self.select_best_of(population, TOURNAMENT_SIZE)
			if not tournament_winner in mating_pool:
				mating_pool.append(tournament_winner)
		return mating_pool

	def breed(self, father, mother):
		child = []
		if not father:
			return mother
		if not mother:
			return father
		father = father.route
		mother = mother.route



		if len(mother) == 0:
			return father

		if len(father) == 0:
			return mother

		gene_length = len(father)

		start = random.randint(0, gene_length-2)
		end = random.randint(start+1, gene_length-1)

		subsection = father[start:end]

		i = 0
		while i < start:
			if not mother[i] in subsection:
				child.append(mother[i])
			i += 1
		for city in subsection:
			child.append(city)
		while len(child) < gene_length:
			if not mother[i] in child:
				child.append(mother[i])
			i += 1

		return TSPSolution(child)

	def next_generation(self, population):
		mating_pool = self.select_mating_pool(population, MATING_POOL_SIZE)

		next_population = []
		next_population += mating_pool
		while len(next_population) < POPULATION_SIZE:
			index_1 = random.randint(0, MATING_POOL_SIZE - 1)
			index_2 = random.randint(0, MATING_POOL_SIZE - 1)
			child = self.breed(mating_pool[index_1], mating_pool[index_2])
			next_population.append(child)

		return next_population

	def mutate(self, route, rate):
		for city_index in range(len(route)):
			if random.random() < rate:
				second_city_index = int(random.random() * len(route))
				route[city_index], route[second_city_index] = route[second_city_index], route[city_index]
				if not TSPSolution(route).cost < np.inf:
					route[city_index], route[second_city_index] = route[second_city_index], route[city_index]
					# If the route is impossible, don't make the switch
		return route


	def get_initial_greedy_solutions(self):
		cities = self._scenario.getCities()
		greed_solutions = []
		for i in range(len(cities)):
			partialPath =[]
			partialPath.append(cities[i])
			while len(partialPath) < len(cities):
				# create a random permutation
				last = partialPath[len(partialPath)-1]

				lowestDist = math.inf
				lowestNode = None
				for c in cities:
					if c in partialPath:
						continue
					if last.costTo(c) < lowestDist:
						lowestNode = c
						lowestDist = last.costTo(c)
				if lowestNode is None:
					break
				partialPath.append(lowestNode)

			if len(partialPath) < len(cities):
				sol = TSPSolution()
			else:
				sol = TSPSolution(partialPath)
				greed_solutions.append(sol)

		return greed_solutions

	def get_random_solution(self):
		cities = self._scenario.getCities()
		perm = np.random.permutation(len(cities))
		route = []
		# Now build the route using the random permutation
		for i in range(len(cities)):
			route.append(cities[perm[i]])
		bssf = TSPSolution(route)

	def get_population(self, population_size):
		cities = self._scenario.getCities()
		current_population = []
		greedy_solutions = self.get_initial_greedy_solutions()

		if len(cities) < 31:
			current_population += greedy_solutions
			while len(current_population) < POPULATION_SIZE:
				current_population.append(self.defaultRandomTour()['soln'])
			return current_population

		current_population += greedy_solutions
		while len(current_population) < POPULATION_SIZE:
			random_index = random.randint(0, len(current_population) - 1)
			child = self.mutate(current_population[random_index].route, MUTATION_RATE)
			if TSPSolution(child).cost and child not in current_population:
				current_population.append(TSPSolution(child))
		return current_population





	def fancy( self,time_allowance=60.0 ):
		start_time = time.time()
		current_population = self.get_population(POPULATION_SIZE)
		bssf = TSPSolution()
		for s in current_population:
			if s.cost < bssf.cost:
				bssf = s
		for i in range(100):
			print(i)
			print(start_time  - time.time())
			children = self.next_generation(current_population)

			current_population = []

			for child in children:
				if not child:
					continue
				child = TSPSolution(self.mutate(child.route, MUTATION_RATE))
				current_population.append(child)


			for s in current_population:
				if s.cost < bssf.cost:
					bssf = s
					print(bssf.cost)
					print(i)
					print("hey")


		for s in current_population:
			if s.cost < bssf.cost:
				bssf = s
		results = {}
		results['cost'] = bssf.cost if True else math.inf
		results['time'] = time.time() - start_time
		results['count'] = 0 #count
		results['soln'] = bssf
		return results
		



