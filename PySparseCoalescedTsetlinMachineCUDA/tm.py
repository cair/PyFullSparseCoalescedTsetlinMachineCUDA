# Copyright (c) 2023 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import numpy as np

import PySparseCoalescedTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix
import sys

from time import time

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state = -1,
			literal_sampling = 1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		print("Initialization of sparse structure.")

		self.number_of_clauses = number_of_clauses
		self.number_of_clause_chunks = (number_of_clauses-1)//32 + 1
		self.number_of_states = number_of_states
		self.T = int(T)
		self.s = s
		self.q = q
		self.max_included_literals = max_included_literals
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.absorbing_state = absorbing_state
		self.literal_sampling = literal_sampling
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

		self.X_train = np.array([])
		self.X_test = np.array([])
		self.encoded_Y = np.array([])
		
		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.encode = mod_encode.get_function("encode")
		self.encode.prepare("PPPiiiiiiii")
		
		self.restore = mod_encode.get_function("restore")
		self.restore.prepare("PPPiiiiiiii")

		self.produce_autoencoder_examples= mod_encode.get_function("produce_autoencoder_example")
		self.produce_autoencoder_examples.prepare("PPiPPiPPiPPiiii")

		self.initialized = False

	def allocate_gpu_memory(self):
		self.included_literals_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_literals*2*4) # Contains index and state of included literals per clause, none at start
		self.included_literals_length_gpu = cuda.mem_alloc(self.number_of_clauses*4) # Number of included literals per clause

		self.excluded_literals_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_literals*2*4) # Contains index and state of excluded literals per clause
		self.excluded_literals_length_gpu = cuda.mem_alloc(self.number_of_clauses*4) # Number of excluded literals per clause

		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)

	def ta_action(self, mc_tm_class, clause, ta):
		if np.array_equal(self.ta_state, np.array([])):
			self.ta_state = np.empty(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_states, dtype=np.uint32)
			cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
		ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_states))

		return (ta_state[mc_tm_class, clause, ta // 32, self.number_of_states-1] & (1 << (ta % 32))) > 0

	def get_state(self):
		if np.array_equal(self.clause_weights, np.array([])):
			self.included_literals = np.empty(self.number_of_clauses*self.number_of_literals*2, dtype=np.uint32)
			cuda.memcpy_dtoh(self.included_literals, self.included_literals_gpu)
			self.included_literals_length = np.empty(self.number_of_clauses, dtype=np.uint32)
			cuda.memcpy_dtoh(self.included_literals_length, self.included_literals_length_gpu)
			self.excluded_literals = np.empty(self.number_of_clauses*self.number_of_literals*2, dtype=np.uint32)
			cuda.memcpy_dtoh(self.excluded_literals, self.excluded_literals_gpu)
			self.excluded_literals_length = np.empty(self.number_of_clauses, dtype=np.uint32)
			cuda.memcpy_dtoh(self.excluded_literals_length, self.excluded_literals_length_gpu)
			self.clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses, dtype=np.int32)
			cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return(((self.included_literals, self.included_literals_length, self.excluded_literals, self.excluded_literals_length), self.clause_weights, self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.dim, self.patch_dim, self.number_of_patches, self.number_of_states, self.number_of_ta_chunks, self.append_negated, self.min_y, self.max_y))

	def set_state(self, state):
		self.number_of_outputs = state[2]
		self.number_of_clauses = state[3]
		self.number_of_literals = state[4]
		self.dim = state[5]
		self.patch_dim = state[6]
		self.number_of_patches = state[7]
		self.number_of_states = state[8]
		self.number_of_ta_chunks = state[9]
		self.append_negated = state[10]
		self.min_y = state[11]
		self.max_y = state[12]
		
		self.ta_state_gpu = cuda.mem_alloc(self.number_of_clauses*self.number_of_ta_chunks*self.number_of_states*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		cuda.memcpy_htod(self.ta_state_gpu, state[0])
		cuda.memcpy_htod(self.clause_weights_gpu, state[1])

		self.X_train = np.array([])
		self.X_test = np.array([])

		self.encoded_Y = np.array([])

		self.ta_state = np.array([])
		self.clause_weights = np.array([])

	# Transform input data for processing at next layer
	def transform(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_patches * self.number_of_ta_chunks*4))
		self.encode_X(X, encoded_X_gpu)

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATE_BITS %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define ABSORBING_STATE %d
#define LITERAL_SAMPING %f
#define S %f
#define THRESHOLD %d

#define NEGATIVE_CLAUSES %d

#define PATCHES %d

#define NUMBER_OF_EXAMPLES %d
		""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_states, self.boost_true_positive_feedback, self.absorbing_state, self.literal_sampling, self.s, self.T, self.negative_clauses, self.number_of_patches, number_of_examples)

		mod = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
		transform = mod.get_function("transform")

		X_transformed_gpu = cuda.mem_alloc(number_of_examples*self.number_of_clauses*4)
		transform(self.ta_state_gpu, encoded_X_gpu, X_transformed_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()
		X_transformed = np.empty(number_of_examples*self.number_of_clauses, dtype=np.uint32)
		cuda.memcpy_dtoh(X_transformed, X_transformed_gpu)
		
		return X_transformed.reshape((number_of_examples, self.number_of_clauses))

	def _init(self, X):
		if self.append_negated:
			self.number_of_literals = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))*2
		else:
			self.number_of_literals = int(self.patch_dim[0]*self.patch_dim[1]*self.dim[2] + (self.dim[0] - self.patch_dim[0]) + (self.dim[1] - self.patch_dim[1]))

		if self.max_included_literals == None:
			self.max_included_literals = self.number_of_literals

		self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1)*(self.dim[1] - self.patch_dim[1] + 1))
		self.number_of_ta_chunks = int((self.number_of_literals-1)//32 + 1)

		parameters = """
#define CLASSES %d
#define CLAUSES %d
#define FEATURES %d
#define STATES %d
#define BOOST_TRUE_POSITIVE_FEEDBACK %d
#define ABSORBING_STATE %d
#define LITERAL_SAMPLING %f
#define S %f
#define THRESHOLD %d
#define Q %f
#define MAX_INCLUDED_LITERALS %d
#define NEGATIVE_CLAUSES %d
#define PATCHES %d
#define NUMBER_OF_EXAMPLES %d
""" % (self.number_of_outputs, self.number_of_clauses, self.number_of_literals, self.number_of_states, self.boost_true_positive_feedback, self.absorbing_state, self.literal_sampling, self.s, self.T, self.q, self.max_included_literals, self.negative_clauses, self.number_of_patches, X.shape[0])

		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare = mod_prepare.get_function("prepare")

		self.allocate_gpu_memory()

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
		self.update = mod_update.get_function("update")
		self.update.prepare("PPPPPPPPPi")

		self.evaluate_update = mod_update.get_function("evaluate")
		self.evaluate_update.prepare("PPPPPPP")

		mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
		self.evaluate = mod_evaluate.get_function("evaluate")
		self.evaluate.prepare("PPPPPPP")

		encoded_X = np.zeros(((self.number_of_patches-1)//32 + 1, self.number_of_literals), dtype=np.uint32)

		if self.append_negated:
			for p_chunk in range((self.number_of_patches-1)//32 + 1):
				for k in range(self.number_of_literals//2, self.number_of_literals):
					encoded_X[p_chunk, k] = (~0) 

		for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
			for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
				p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x
				p_chunk = p // 32
				p_pos = p % 32

				for y_threshold in range(self.dim[1] - self.patch_dim[1]):
					patch_pos = y_threshold
					if patch_coordinate_y > y_threshold:
						encoded_X[p_chunk, patch_pos] |= (1 << p_pos)

						if self.append_negated:
							encoded_X[p_chunk, patch_pos + self.number_of_literals//2] &= ~(1 << p_pos)

				for x_threshold in range(self.dim[0] - self.patch_dim[0]):
					patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
					if patch_coordinate_x > x_threshold:
						encoded_X[p_chunk, patch_pos] |= (1 << p_pos)

						if self.append_negated:
							encoded_X[p_chunk, patch_pos + self.number_of_literals//2] &= ~(1 << p_pos)

		encoded_X = encoded_X.reshape(-1)
		self.encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes)
		cuda.memcpy_htod(self.encoded_X_gpu, encoded_X)

		self.initialized = True

	def _init_fit(self, X, encoded_Y, incremental):
		if not self.initialized:
			self._init(X)
			self.prepare(g.state, self.included_literals_gpu, self.included_literals_length_gpu, self.excluded_literals_gpu, self.excluded_literals_length_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(g.state, self.included_literals_gpu, self.included_literals_length_gpu, self.excluded_literals_gpu, self.excluded_literals_length_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
			cuda.Context.synchronize()

		if not np.array_equal(self.X_train, np.concatenate((X.indptr, X.indices))):
			self.train_X = np.concatenate((X.indptr, X.indices))
			self.X_train_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_indptr_gpu, X.indptr)

			self.X_train_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
			cuda.memcpy_htod(self.X_train_indices_gpu, X.indices)

		if not np.array_equal(self.encoded_Y, encoded_Y):
			self.encoded_Y = encoded_Y

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

	def _fit(self, X, encoded_Y, epochs=100, incremental=False):
		self._init_fit(X, encoded_Y, incremental)

		for epoch in range(epochs):
			for e in range(X.shape[0]):
				class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				self.encode.prepared_call(
				self.grid,
				self.block,
				self.X_train_indptr_gpu,
				self.X_train_indices_gpu,
				self.encoded_X_gpu,
					np.int32(e),
					np.int32(self.dim[0]),
					np.int32(self.dim[1]),
					np.int32(self.dim[2]),
					np.int32(self.patch_dim[0]),
					np.int32(self.patch_dim[1]),
					np.int32(self.append_negated),
					np.int32(0)
				)
				cuda.Context.synchronize()

				self.evaluate_update.prepared_call(
					self.grid,
					self.block,
					self.included_literals_gpu,
					self.included_literals_length_gpu,
					self.excluded_literals_gpu,
					self.excluded_literals_length_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu
				)
				cuda.Context.synchronize()

				self.update.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.included_literals_gpu,
					self.included_literals_length_gpu,
					self.excluded_literals_gpu,
					self.excluded_literals_length_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu,
					self.encoded_Y_gpu,
					np.int32(e)
				)
				cuda.Context.synchronize()

				self.restore.prepared_call(
					self.grid,
					self.block,
					self.X_train_indptr_gpu,
					self.X_train_indices_gpu,
					self.encoded_X_gpu,
					np.int32(e),
					np.int32(self.dim[0]),
					np.int32(self.dim[1]),
					np.int32(self.dim[2]),
					np.int32(self.patch_dim[0]),
					np.int32(self.patch_dim[1]),
					np.int32(self.append_negated),
					np.int32(0)
				)
				cuda.Context.synchronize()


		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def _score(self, X):
		if not self.initialized:
			print("Error: Model not trained.")
			sys.exit(-1)

		if not np.array_equal(self.X_test, np.concatenate((X.indptr, X.indices))):
			self.X_test = np.concatenate((X.indptr, X.indices))

			self.X_test_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
			cuda.memcpy_htod(self.X_test_indptr_gpu, X.indptr)

			self.X_test_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
			cuda.memcpy_htod(self.X_test_indices_gpu, X.indices)

		class_sum = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.int32)
		for e in range(X.shape[0]):
			cuda.memcpy_htod(self.class_sum_gpu, class_sum[e,:])

			self.encode.prepared_call(
				self.grid,
				self.block,
				self.X_test_indptr_gpu,
				self.X_test_indices_gpu,
				self.encoded_X_gpu,
				np.int32(e),
				np.int32(self.dim[0]),
				np.int32(self.dim[1]),
				np.int32(self.dim[2]),
				np.int32(self.patch_dim[0]),
				np.int32(self.patch_dim[1]),
				np.int32(self.append_negated),
				np.int32(0)
			)
			cuda.Context.synchronize()

			self.evaluate.prepared_call(
				self.grid,
				self.block,
				self.included_literals_gpu,
				self.included_literals_length_gpu,
				self.excluded_literals_gpu,
				self.excluded_literals_length_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				self.encoded_X_gpu
			)
			cuda.Context.synchronize()

			self.restore.prepared_call(
				self.grid,
				self.block,
				self.X_test_indptr_gpu,
				self.X_test_indices_gpu,
				self.encoded_X_gpu,
				np.int32(e),
				np.int32(self.dim[0]),
				np.int32(self.dim[1]),
				np.int32(self.dim[2]),
				np.int32(self.patch_dim[0]),
				np.int32(self.patch_dim[1]),
				np.int32(self.append_negated),
				np.int32(0)
			)
			cuda.Context.synchronize()

			cuda.memcpy_dtoh(class_sum[e,:], self.class_sum_gpu)

		return class_sum
	
class MultiClassConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			dim,
			patch_dim,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(
			number_of_clauses,
			T,
			s,
			q=q,
			max_included_literals=max_included_literals,
			boost_true_positive_feedback=boost_true_positive_feedback,
			absorbing_state=absorbing_state,
			number_of_states=number_of_states,
			append_negated=append_negated,
			grid=grid,
			block=block
		)
		self.dim = dim
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = int(np.max(Y) + 1)
	
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)

class MultiOutputConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
	"""
	This class ...
	"""
	
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			patch_dim,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.patch_dim = patch_dim
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		self.number_of_outputs = Y.shape[1]

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

	def score(self, X):
		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32).transpose()

class MultiOutputTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = Y.shape[1]
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32).transpose()

class MultiClassTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = int(np.max(Y) + 1)

		self.dim = (X.shape[1], 1, 1)
		self.patch_dim = (X.shape[1], 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			q=1.0,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)
		
		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return int(self.score(X) >= 0)

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			max_included_literals=None,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 0

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y

class AutoEncoderTsetlinMachine(CommonTsetlinMachine):
	def __init__(
			self,
			number_of_clauses,
			T,
			s,
			active_output,
			q=1.0,
			max_included_literals=None,
			accumulation = 1,
			boost_true_positive_feedback=1,
			absorbing_state=-1,
			literal_sampling=1.0,
			number_of_states=256,
			append_negated=True,
			grid=(16*13*4,1,1),
			block=(128,1,1)
	):
		super().__init__(number_of_clauses, T, s, q=q, max_included_literals=max_included_literals, boost_true_positive_feedback=boost_true_positive_feedback, absorbing_state=absorbing_state, literal_sampling=literal_sampling, number_of_states=number_of_states, append_negated=append_negated, grid=grid, block=block)
		self.negative_clauses = 1

		self.active_output = np.array(active_output).astype(np.uint32)
		self.accumulation = accumulation

	def _init_fit(self, X_csr, encoded_Y, incremental):
		if not self.initialized:
			self._init(X_csr)
			self.prepare(
				g.state,
				self.included_literals_gpu,
				self.included_literals_length_gpu,
				self.excluded_literals_gpu,
				self.excluded_literals_length_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				grid=self.grid,
				block=self.block
			)
			cuda.Context.synchronize()
		elif incremental == False:
			self.prepare(
				g.state,
				self.included_literals_gpu,
				self.included_literals_length_gpu,
				self.excluded_literals_gpu,
				self.excluded_literals_length_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				grid=self.grid,
				block=self.block
			)
			cuda.Context.synchronize()

		if not np.array_equal(self.X_train, np.concatenate((X_csr.indptr, X_csr.indices))):
			self.train_X = np.concatenate((X_csr.indptr, X_csr.indices))

			X_csc = X_csr.tocsc()
			
			self.X_train_csr_indptr_gpu = cuda.mem_alloc(X_csr.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_csr_indptr_gpu, X_csr.indptr)

			self.X_train_csr_indices_gpu = cuda.mem_alloc(X_csr.indices.nbytes)
			cuda.memcpy_htod(self.X_train_csr_indices_gpu, X_csr.indices)

			self.X_train_csc_indptr_gpu = cuda.mem_alloc(X_csc.indptr.nbytes)
			cuda.memcpy_htod(self.X_train_csc_indptr_gpu, X_csc.indptr)

			self.X_train_csc_indices_gpu = cuda.mem_alloc(X_csc.indices.nbytes)
			cuda.memcpy_htod(self.X_train_csc_indices_gpu, X_csc.indices)

			self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
			cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

			self.active_output_gpu = cuda.mem_alloc(self.active_output.nbytes)
			cuda.memcpy_htod(self.active_output_gpu, self.active_output)

	def _fit(self, X_csr, encoded_Y, number_of_examples, epochs, incremental=False):
		self._init_fit(X_csr, encoded_Y, incremental=incremental)

		for epoch in range(epochs):
			for e in range(number_of_examples):
				class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
				cuda.memcpy_htod(self.class_sum_gpu, class_sum)

				target = np.random.choice(self.number_of_outputs)
				self.produce_autoencoder_examples.prepared_call(
                                            self.grid,
                                            self.block,
											g.state,
                                            self.active_output_gpu,
                                            self.active_output.shape[0],
                                            self.X_train_csr_indptr_gpu,
                                            self.X_train_csr_indices_gpu,
                                            X_csr.shape[0],
                                            self.X_train_csc_indptr_gpu,
                                            self.X_train_csc_indices_gpu,
                                            X_csr.shape[1],
                                            self.encoded_X_gpu,
                                            self.encoded_Y_gpu,
                                            target,
                                            int(self.accumulation),
                                            int(self.T),
                                            int(self.append_negated)
				)
				cuda.Context.synchronize()

				self.evaluate_update.prepared_call(
					self.grid,
					self.block,
					self.included_literals_gpu,
					self.included_literals_length_gpu,
					self.excluded_literals_gpu,
					self.excluded_literals_length_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu
				)
				cuda.Context.synchronize()

				self.update.prepared_call(
					self.grid,
					self.block,
					g.state, 
					self.included_literals_gpu,
					self.included_literals_length_gpu,
					self.excluded_literals_gpu,
					self.excluded_literals_length_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu,
					self.encoded_Y_gpu,
					np.int32(0)
				)
				cuda.Context.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])
		
		return

	def fit(self, X, number_of_examples=2000, epochs=100, incremental=False):
		X_csr = csr_matrix(X)

		self.number_of_outputs = self.active_output.shape[0]

		self.dim = (X_csr.shape[1], 1, 1)
		self.patch_dim = (X_csr.shape[1], 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.zeros(self.number_of_outputs, dtype = np.int32)

		self._fit(X_csr, encoded_Y, number_of_examples, epochs, incremental = incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=1)
