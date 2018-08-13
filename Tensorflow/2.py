import tensorflow as tf

#any node we create in creating the graph is added to the default graph

x1 = tf.Variable(4)
print x1.graph is tf.get_default_graph()


#to make an independent graph
graph = tf.Graph()

with graph.as_default():
	x2 = tf.Variable(5)

#x2 variable does not belong to default graph
print x2.graph is tf.get_default_graph()
#it belongs to our graph
print x2.graph is graph

#to reset the default graph
#tf.reset_default_graph()
