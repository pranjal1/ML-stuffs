import tensorflow as tf


w = tf.Variable(4)
x = w + 2
y = x * x
z = y ** 2

init = tf.global_variables_initializer() 

'''
#in this code, the graph is evaluated twice, once for value of y and the other for value of z
with tf.Session() as sess:
	init.run()
	print y.eval()
	print z.eval()

'''

#efficient way
#in this way, the values are computed in a single execution of graph
with tf.Session() as sess:
	init.run()
	val_x,val_y = sess.run([y,z])
	print val_x,val_y

#the values stored in the nodes are only saved for a session, session over -> node's memory is freed.
#also, no two session (even if they use the same graph) share any state (no variables and so on)
#what is done to solve this issue in distributed systems is that, the variables are stored in a particular server and used by multiple sessions
