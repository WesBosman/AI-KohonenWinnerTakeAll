import random
import math
import matplotlib.pyplot as plt

# Use the Kohonen's Winner Take All Clustering Algorithm on Iris Data Set
def main():

	# Run the Example from the slides
	# The Example from the slides works
	#slides_example()
	
	"""
	
		Remember to Answer how changing the number of iterations, 
		learning constant and neurons will change the outcome of the neurons
		
	"""
 	
 	# Get the data from the file and graph it
	ex1_data = get_data_from_txt("Ex1_data.txt")
	plot_data(ex1_data, "red" , "o", "Ex1_Data", "X1", "X2", "Ex1_Data_Graph")
	
	# Normalize data and graph it
	norm_data = norm_the_data(ex1_data)
	plot_data(norm_data, "green", "^", "Normalized Ex1 Data", "X1", "X2", "Normalized_Ex1_Data_Graph")
	
	# Use the two neuron network and randomly assign weights
	# Then Repeat with 3 and 7 Neurons
	num_neurons = 7
	learn       = 0.9
	#print("Number of Neurons = %d , Learn = %.2f" %(num_neurons, learn))
	my_net_dictionary = dict()
	
	# Create Neurons
	neurons = create_neurons(num_neurons, learn)
	
	for i in range(0, 30):
		# Apply a new pattern recursively
		for pattern in norm_data:
			# Calculate the net weights to find the winner
			net_dictionary, nets = calculate_nets(neurons, pattern)
			# Find the winning neuron and update its weights
			winning_neuron = update_winning_neuron_weights(net_dictionary, nets, pattern, learn)
			# Normalize the winning neurons weights again
			winning_neuron = normalize_winning_neuron(winning_neuron)
			my_net_dictionary = net_dictionary
		
	my_neurons = my_net_dictionary.values()
	name = "Clusters_for_Ex1_%d_neurons" %(num_neurons)
 	plot_center_of_clusters(my_neurons, num_neurons, "*", "purple", name)
	
		

# Run the Example from the slides
def slides_example():
	slides_data = [[5.9630, 0.7258],
 					[4.1168, 2.9694],
 					[1.8184, 6.0148],
 					[6.2139, 2.4288],
 					[6.1290, 1.3876],
 					[1.0562, 5.8288],
 					[4.3185, 2.3792],
 					[2.6108, 5.4870],
 					[1.5999, 4.1317],
 					[1.1046, 4.1969] ]
 					
 	slide_weights = [[.9459, .3243],
					[.6690, .7433],
					[.3714, .9285]]
 					
 	# Plot Slides Data
 	plot_data(slides_data, "red", "^", "Slides_Data", "X1", "X2", "Slides_Data_Graph")
 	# Plot the Normed Slides Data
 	slides_data_norm = norm_the_data(slides_data)
 	plot_data(slides_data_norm, "green", "^", "Slides_Data_Norm", "X1", "X2", "Slides_Data_Norm_Graph") 
 	# Create Neuron
 	num_neurons = 3
 	learning_c  = 0.3
 	neurons     = []
 	
 	# Create three neurons with weights
 	for i in range(0, num_neurons):
 		n = Neuron((i + 1), slide_weights[i], learning_c)
 		neurons.append(n)
 		 
 	my_net_dictionary = dict()
 	
 	for i in range(0, 30):
 		# Loop through the data
 		for pattern in slides_data_norm:
 			# Find the winning net
 			net_dictionary, nets = calculate_nets(neurons, pattern)
 			winning_neuron = update_winning_neuron_weights(net_dictionary, nets, pattern, learning_c)
 			winning_neuron = normalize_winning_neuron(winning_neuron)
 			my_net_dictionary = net_dictionary
 		
 	
 	my_neurons = my_net_dictionary.values()
 	name = "Clusters_for_Slides_%d_neurons" %(num_neurons)
 	plot_center_of_clusters(my_neurons, num_neurons, "*", "orange", name)
 	
 	
		
# Create Neurons with a certain learning constant
def create_neurons(num_neurons, learn):
	# Use the two neuron network and randomly assign weights
	# Then Repeat with 3 and 7 Neurons
	print("Creating Neurons")
	print("Number of Neurons = %d , Learn = %.2f" %(num_neurons, learn))
	
	# Make a 2 neurons with randomized weights
	neurons = []
	for i in range(0, num_neurons):
		# Get random weights for the number of neurons
		w = get_random_weights(num_neurons)
		# Keep track of which neuron it is
		n = Neuron(( i+1 ), w, learn)
		print("Weights = %r , Neuron %d" %(w, n.number))
		neurons.append(n)
		
	return neurons
	
# Calculate the nets of the neurons
def calculate_nets(neurons, pattern):
	#print("")
	#print("Calculate Nets")
	net_dictionary = dict()
	nets = []
	for i in range(0, len(neurons)):
		neuron = neurons[i]
		#print("Neuron weights = %r" %(neuron.weights))
		#print("Neuron number  = %r" %(neuron.number))
		#print("Pattern        = %r" %(pattern))
		
		neuron_net = 0
		# Change this if the number of neurons increases!!!!!!!
		# The Normalized weight multiplied by the Normalized pattern
		for (x,y) in zip(pattern, neuron.weights):
			#print("X == %r     Y == %r" %(x , y))
			net = x * y
			#print("Net = %r" %(net))
			neuron_net = neuron_net + net
		#print("THE NEURON NET = %r" %(neuron_net))
		neuron.net = neuron_net
		nets.append(neuron_net)
		net_dictionary[neuron_net] = neuron
	
	#print("Neuron Nets = %r" %(nets))
	#print("Net Dictionary = %r" %(net_dictionary))

	return net_dictionary, nets
	

def update_winning_neuron_weights(net_dictionary, nets, pattern, learn):
	# Find the winning neuron
	#print("")
	#print("Update Winning Neuron Weights")
	winning_neuron = net_dictionary[max(nets)]
	#print("Updating Neuron %d" %(winning_neuron.number))
	#print("Pattern           = %r" %(pattern))
	#print("Weights of Winner = %r" %(winning_neuron.weights))
		
	# Update the weights of the winning neuron
	new_x_values = []
	new_weights  = []
	for x in pattern:
		new_x = x * learn
		new_x_values.append(new_x)

	for i in range(0, len(new_x_values)):
		wi = winning_neuron.weights[i]
		xi = new_x_values[i]
		new_w = wi + xi
		new_weights.append(new_w)
		#print("Weight = %r + X = %r == %r " %(wi, xi, (wi + xi)))
		
	#print("New winning neuron weights = %r" %(new_weights))
	winning_neuron.weights = new_weights
	return winning_neuron
	
	
def normalize_winning_neuron(winning_neuron):
	# Normalize the data for the winning neuron
	#print("")
	#print("Normalizing weights")
	new_weights = []
	norm_weight = 0
	for w in winning_neuron.weights:
		#print("W = %r" % w)
		norm_weight = norm_weight + w ** 2
	
	norm_weights_squared = math.sqrt(norm_weight)
	
	for w in winning_neuron.weights:
		new_weight = w / norm_weights_squared
		new_weights.append(new_weight)
	
	#print("Updated Weights %r for Neuron %d" %(new_weights, winning_neuron.number))
	winning_neuron.weights = new_weights
	
	return winning_neuron
	
	
# Plot the final neuron weights
def plot_center_of_clusters(neurons, number_clusters, m, c, name):
	print("")
	print("Plot Center of Clusters")
	for n in neurons:
		#print("Weights For N = %r" %(n.weights))
		for i in range(0, len(n.weights)):
			print("N Weights at %d : %r " %(i, n.weights[i]))
		
		plt.scatter(n.weights[0], n.weights[1], marker = m, color = c )
	
	title = "Center of Clusters (%d Neurons)" %(number_clusters)
	plt.suptitle(title, fontsize= 18, fontweight = "bold")
	plt.title("Wes Bosman")
	plt.savefig(name)
	plt.show()
	

# Get random weights for the random number of neurons	
def get_random_weights(r):
	weights = []
	for i in range(0, r):
		weights.append(random.random())
	return weights
	
	
# Get the data from the text file for the example 1 data 
def get_data_from_txt(name):
	results = []
	with open(name) as file:
		for line in file:
			x1, x2 = line.split(",")
			try:
				a = float(x1)
				b = float(x2)
				results.append([a, b])
			except:
				print("Exception occured trying to turn string to float")
	return results


def norm_the_data(data):
	normed_data = []
	
	for (x, y) in data:
		#print("X = % r / Y = %r" %(x,y))
		sum = x**2 + y**2
		xi = x/math.sqrt(sum)
		yi = y/math.sqrt(sum)
		#print("Normed X = %r , Normed Y = %r" %(xi, yi))
		normed_data.append([xi, yi])
	
	return normed_data
			
			
# Plot the example 1 data
def plot_data(data, c, m, title, x_label, y_label, save_name):
	for (x, y) in data:
		print(x, y)
		plt.scatter(x, y, color=c, marker = m)
	plt.suptitle(title, fontsize=18, fontweight="bold")
	plt.title("Wes Bosman")
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(save_name)
	plt.show()
	
	
# Create a neuron class for the Kohonen network
class Neuron():
	neurons = []
	# Initialize the neuron with number and weights
	def __init__(self, number, weights, learn):
		self.weights = weights
		self.learning_constant = learn
		self.number = number
		self.neurons.append(self)
		self.net = 0

		
if __name__ == "__main__":
	main()
	