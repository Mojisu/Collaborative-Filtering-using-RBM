"""RBM for Collaborative Filtering using Theano library. This effort is based 
on the paper on Collaborative Filtering using Restricted Boltzmann Machines by
Salakhutdinov et al.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.

Collaborative Filtering is a form of recommendor system in which the system
would identify the users who share the same preferneces with the active users, 
and propose items which the like-minded users favoured.
"""

import numpy

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class CFRBM(object):
	""" Collaborative Filtering using Restricted Boltzmann Machine (RBM) """
	
	def __init__( self, n_visible, n_hidden, n_rating, W, hbias, vbias, \
		numpy_rng = None, theano_rng = None):

	'''CFRBM constructor'''

	self.n_visible = n_visible
	self.n_hidden = n_visible
	# n_rating is the maximun rating that can be given to a movie by a user,
	# denoted as 'k' in the paper
	self.n_rating = n_rating


	if numpy_rng is None:
		# Creates a numpy random number generator
		numpy_rng = numpy.random.RandomState(1234)

	if theano_rng is None:
		#  Creates a theano random number generator
		theano_rng = RandomStreams(numpy_rng.randint(2 ** 40))

	if W is none:
		# Weight parameter
		#  W(i,j,k):Symmetric Interaction parameter between feature j and 
		# rating k of movie i. n_visible : |i|, n_hidden : |j|,
		# n_rating : |k|
		intitial_W = numpy.asarray(numpy_rng.uniform(
					low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					size=(n_visible,n_hidden,n_rating)),
					dtype=theano.config.floatX)
		# theano shared variable
        W = theano.shared(value=initial_W, name='W', borrow=True)

       if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)
 
        if vbias is None:
            # create shared variable for visible units bias ie n_rating for
            # each softmax unit.
            vbias = theano.shared(value=numpy.zeros([n_visible,n_rating],
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

		self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]


    def energy(self, v_matrix_sample, h_sample):
       	'''Energy term used in the marginal distribution'''
       	# h_sample is a n_hidden x 1 array
       	# v_matrix_sample is a n_visible x n_rating array
       	for i in xrange(n_rating):
       		W_term += T.dot((T.dot(self.W[:,:,i],h_sample)).T, /
       			v_matrix_sample[:,i])
       	
       	Z_term = numpy.zeros(n_visible) 
       	for i in xrange(n_visible):
       		for l in xrange(n_rating):
       			for j in xrange(n_hidden):
       				loop_term += h_sample[j] * self.W[i,j,l]
       			Z_term[i] += T.exp(self.vbias[i,l] + loop_term)
       		Z_term +=  T.log(Z_term[i])

       	visible_term = numpy.trace(T.dot(v_matrix_sample, /
       		self.vbias.T))

       	hidden_term = T.dot(h_sample,self.hbias.T)

       	return - W_term + Z_term - visible_term - hidden_term
    
    def visible_model(self,i,hid):
    	'''The new conditional probability will take softmax function instead of 
    	sigmoid function. We use a conditional multinomial distribution (a “soft-
		max”) for modeling each column of the observed “visible” binary 
		rating matrix V'''

    	activation_i = self.vbias[i,:] + (T.dot(hid,self.W[i,:,:])).T
    	return [activation_i, T.nnet.softmax(activation_i)]

    def hidden_model(self,j,vis):
    	''' A conditional Bernoulli distribution for modeling “hidden” 
    	user features h '''

    	activation_j = self.hbias[j] + numpy.trace(T.dot(vis, self.W[:,j,:].T))
    	return [activation_j, T.nnet.sigmoid(activation_j)]

    def distribution_visible(self,):
    	''' The marginal distribution over the visible ratings V '''