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

import time
import PIL.Image

import numpy

import theano
import theano.tensor as T
import os

import pandas

from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images

class CFRBM(object):
    """ Collaborative Filtering using Restricted Boltzmann Machine (RBM) """
 
    def __init__( self, n_visible, n_hidden, n_rating, W=None, hbias=None, vbias=None, \
      numpy_rng=None, theano_rng=None):

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

        if W is None:
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
        
        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]


    def energy(self, v_matrix_sample, h_sample=None):
       	'''Energy term used in the marginal distribution'''
       	# h_sample is a n_hidden x 1 array
       	# v_matrix_sample is a n_visible x n_rating array
        if h_sample is None:
          h_sample = numpy.zeros(n_hidden,dtype=theano.config.floatX)

       	for i in xrange(n_rating):
       		W_term += T.dot((T.dot(self.W[:,:,i],h_sample)).T, \
            v_matrix_sample[:,i])
       	
       	Z_term = numpy.zeros(n_visible) 
       	for i in xrange(n_visible):
       		for l in xrange(n_rating):
       			for j in xrange(n_hidden):
       				loop_term += h_sample[j] * self.W[i,j,l]
       			Z_term[i] += T.exp(self.vbias[i,l] + loop_term)
       		Z_term +=  T.log(Z_term[i])

       	visible_term = numpy.trace(T.dot(v_matrix_sample, \
       		self.vbias.T))

       	hidden_term = T.dot(h_sample,self.hbias.T)

       	return - W_term + Z_term - visible_term - hidden_term
    
    def visible_model_dist(self,i,hid):
    	'''The new conditional probability will take softmax function instead of 
    	sigmoid function. We use a conditional multinomial distribution (a soft-
		  max) for modeling each column of the observed visible binary 
	    rating matrix V'''

    	activation_i = self.vbias[i,:] + (T.dot(hid,self.W[i,:,:])).T
    	return [activation_i, T.nnet.softmax(activation_i)]

    def hidden_model_dist(self,j,vis):
    	''' A conditional Bernoulli distribution for modeling hidden
    	user features h '''

    	activation_j = self.hbias[j] + numpy.trace(T.dot(vis, self.W[:,j,:].T))
    	return [activation_j, T.nnet.sigmoid(activation_j)]

    # def distribution_visible(self,):
    # 	''' The marginal distribution over the visible ratings V '''

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.hidden_model_dist(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default.
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.visible_model_dist(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default.
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hid_vis_hid(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vis_hid_vis(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self.gibbs_hid_vis_hid,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 6th output
                    outputs_info=[None,  None,  None, None, None, chain_start],
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        chain_end_v = nv_samples[-1]
        chain_end_h = nh_samples[-1]

        cost = T.mean(self.energy(self.input)) - T.mean(
            self.free_energy(chain_end_v,chain_end_h))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                    dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            # monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates,
                                                           pre_sigmoid_nvs[-1])

        return monitoring_cost, updates


    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error
        """

        cross_entropy = T.mean(
                T.sum(self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                      axis=1))

        return cross_entropy

def test_rbm(learning_rate=0.1,
             training_epochs=15,
             dataset='ml-100k/u.data',
             batch_size=20,
             n_chians=20,
             n_samples=10,
             output_folder='cfrbm_plots',
             n_hidden=500):

    """
      Training the RBM and afterwards sample from it using Theano.
      
      This is shown using the "MovieLens Dataset"
      
      :param learning_rate: learning rate used for training the RBM

      :param training_epochs: number of epochs used for training

      :param dataset: path the the dataset

      :param batch_size: size of a batch used to train the RBM

      :param n_chains: number of parallel Gibbs chains to be used for sampling

      :param n_samples: number of samples to plot for each chain

    """

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    raw_ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols) 
    
    ratings = raw_ratings[[0,1,2]]
    
    train_set_x = ratings.loc[1:50000][[0,1]]
    train_set_y = ratings.loc[1:50000][[2]]
    test = ratings.loc[1:50000][[0,1]]
    test_set_y = ratings.loc[1:50000][[2]]

    n_train_batches = train_set_x.shape[0]/ batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # A Dict consisting as each user as Key and the movies rated by that user as the Val.
    adict = {}
    df = train_set_x[['user_id','movie_id']]

    # yoyo = ratings[['user_id','movie_id']]
    # print yoyo[(ratings.user_id==196)].shape[0] # number of movies rated by user_id = '196'. 
    
    for userid in list(df[[0]].values.flatten()):
      if userid not in adict:
        movies_rated = yoyo[(ratings.user_id == userid)][[1]]
        adict[userid] = []
        mylist = adict.get(userid)
        mylist.append(list(movies_rated.values.flatten()))



    # Now for each user in the dict a new RBM has to be created. The number of hidden units in each of the RBM
    # will be the number of Values corresponding to that user's Key.

    

    # construct the RBM class
    rbm = RBM(input=x, n_visible=,
              n_hidden=n_hidden, n_rating=K, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)
   
    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index], cost,
           updates=updates,
           givens={x: train_set_x.loc[index * batch_size:
                                  (index + 1) * batch_size]},
           name='train_rbm')

    plotting_time = 0.
    start_time = time.clock()

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
                 X=rbm.W.get_value(borrow=True).T,
                 img_shape=(28, 28), tile_shape=(10, 10),
                 tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))

    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.shape[0

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    [presig_hids, hid_mfs, hid_samples, presig_vis,
     vis_mfs, vis_samples], updates =  \
                        theano.scan(rbm.gibbs_vhv,
                                outputs_info=[None,  None, None, None,
                                              None, persistent_vis_chain],
                                n_steps=plot_every)

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                updates=updates,
                                name='sample_fn')

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
                             dtype='uint8')
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                X=vis_mf,
                img_shape=(28, 28),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1))
        # construct image

    image = PIL.Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')

if __name__ == '__main__':
    test_rbm()













    


