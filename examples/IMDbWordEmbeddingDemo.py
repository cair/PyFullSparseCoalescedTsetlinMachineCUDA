import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from PySparseCoalescedTsetlinMachineCUDA.tm import AutoEncoderTsetlinMachine


target_words = ['awful', 'terrible', 'lousy', 'abysmal', 'crap', 'outstanding', 'brilliant', 'excellent', 'superb', 'magnificent', 'marvellous', 'truck', 'plane', 'car', 'cars', 'motorcycle',  'scary', 'frightening', 'terrifying', 'horrifying', 'funny', 'comic', 'hilarious', 'witty']

number_of_output_words = 2000

clause_weight_threshold = 0

number_of_examples = 20000
accumulation = 25

factor = 4
clauses = factor*20
T = factor*40
s = 5.0

print("Number of clauses:", clauses)

NUM_WORDS=10000
INDEX_FROM=2

print("Downloading dataset...")

train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)

train_x,train_y = train
test_x,test_y = test

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

print("Producing bit representation...")

id_to_word = {value:key for key,value in word_to_id.items()}

training_documents = []
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		terms.append(id_to_word[word_id].lower())

	training_documents.append(terms)

testing_documents = []
for i in range(test_y.shape[0]):
	terms = []
	for word_id in test_x[i]:
		terms.append(id_to_word[word_id].lower())

	testing_documents.append(terms)

def tokenizer(s):
	return s

vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)

X_train = vectorizer_X.fit_transform(training_documents)
#feature_names = vectorizer_X.get_feature_names_out()
#number_of_features = vectorizer_X.get_feature_names_out().shape[0]

feature_names = vectorizer_X.get_feature_names()
number_of_features = len(feature_names)

target_words = []
for word in feature_names:
	word_id = vectorizer_X.vocabulary_[word]

	target_words.append(word)
	if len(target_words) == number_of_output_words:
		break

X_test = vectorizer_X.transform(testing_documents)

output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
	target_word = target_words[i]
	target_id = vectorizer_X.vocabulary_[target_word]
	output_active[i] = target_id

tm = AutoEncoderTsetlinMachine(clauses, T, s, output_active, max_included_literals=3, accumulation=accumulation, append_negated=False)

print("\nAccuracy Over 250 Epochs:")
for e in range(250):
	start_training = time()
	tm.fit(X_train, epochs=1, number_of_examples=number_of_examples, incremental=True)
	stop_training = time()
	
	print("\nEpoch #%d\n" % (e+1))

	weights = tm.get_state()[1].reshape((len(target_words), -1))

	print("Clauses\n")

	for j in range(clauses):
		print("Clause #%d " % (j), end=' ')
		for i in range(len(target_words)):
			print("%s:W%d " % (target_words[i], weights[i,j]), end=' ')
		print()

		# l = []
		# for k in range(tm.clause_bank.number_of_literals):
		# 	if tm.get_ta_action(j, k) == 1:
		# 		if k < tm.number_of_features//2:
		# 			l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
		# 		else:
		# 			l.append("¬%s(%d)" % (feature_names[k-tm.number_of_features//2], tm.clause_bank.get_ta_state(j, k)))
		# print(" ∧ ".join(l))

	profile = np.empty((len(target_words), clauses))
	for i in range(len(target_words)):
		profile[i,:] = np.where(weights[i,:] >= clause_weight_threshold, weights[i,:], 0)

	similarity = cosine_similarity(profile)

	print("\nWord Similarity\n")

	for i in range(len(target_words)):
		print(target_words[i], end=': ')
		sorted_index = np.argsort(-1*similarity[i,:])
		for j in range(1, 10):
			print("%s(%.2f) " % (target_words[sorted_index[j]], similarity[i,sorted_index[j]]), end=' ')
		print()

	print("\nTraining Time: %.2f" % (stop_training - start_training))
