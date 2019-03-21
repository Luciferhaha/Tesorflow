import tensorflow as tf
import numpy as np
import random
import time
from model_seq2seq_contrib import Seq2seq
from mypreprocess import *


tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

source_int,target_int ,int2word, word2int= mypreprocess().process_data()
docs_source =mypreprocess().data_source
docs_target =mypreprocess().data_target
class Config(object):
	embedding_dim = 100
	hidden_dim = 50
	batch_size = 128
	learning_rate = 0.005
	source_vocab_size = None
	target_vocab_size = None




	
# def make_vocab(docs):
# 	w2i = {"<PAD>":0, "<GO>":1, "<EOS>":2}
# 	i2w = {0:"<PAD>", 1:"<GO>", 2:"<EOS>"}
# 	for doc in docs:
# 		for w in doc:
# 			if w not in w2i:
# 				i2w[len(w2i)] = w
# 				w2i[w] = len(w2i)
# 	return w2i, i2w
#
#
# def doc_to_seq(docs):
# 	word2int = {"<PAD>":0, "<GO>":1, "<EOS>":2}
# 	int2word = {0:"<PAD>", 1:"<GO>", 2:"<EOS>"}
# 	seqs = []
# 	for doc in docs:
# 		seq = []
# 		for w in doc:
# 			if w not in word2int:
# 				int2word[len(word2int)] = w
# 				word2int[w] = len(word2int)
# 			seq.append(word2int[w])
# 		seqs.append(seq)
# 	return seqs, word2int, int2word


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
	ps = []
	while len(ps) < batch_size:
		ps.append(random.randint(0, len(docs_source)-1))
	print(ps[0])
	print(ps[1])
	source_batch = []
	target_batch = []
	
	source_lens = [len(docs_source[p]) for p in ps]
	target_lens = [len(docs_target[p])+1 for p in ps]
	
	max_source_len = max(source_lens)
	max_target_len = max(target_lens)
		
	for p in ps:
		# print([w2i_source[w] for w in docs_source[p]])
		# print(p)
		source_seq = [w2i_source[w] for w in docs_source[p]] + [w2i_source["<PAD>"]]*(max_source_len-len(docs_source[p]))
		target_seq = [w2i_target[w] for w in docs_target[p]] + [w2i_target["<PAD>"]]*(max_target_len-1-len(docs_target[p]))+[w2i_target["<EOS>"]]
		source_batch.append(source_seq)
		target_batch.append(target_seq)
	
	return source_batch, source_lens, target_batch, target_lens
	
	
if __name__ == "__main__":

	print("(1)load data......")

	w2i_source, i2w_source = source_int,int2word
	w2i_target, i2w_target = word2int, int2word
	
	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(w2i_source)
	config.target_vocab_size = len(w2i_target)
	print(w2i_target)
	model = Seq2seq(config=config, w2i_target=w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1)
	
	
	print("(3) run model......")
	batches = 3000
	print_every = 100
	
	with tf.Session(config=tf_config) as sess:
		tf.summary.FileWriter('graph', sess.graph)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		
		losses = []
		total_loss = 0
		for batch in range(batches):
			source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target, w2i_target, config.batch_size)
			
			feed_dict = {
				model.seq_inputs: source_batch,
				model.seq_inputs_length: source_lens,
				model.seq_targets: target_batch,
				model.seq_targets_length: target_lens
			}
			
			loss, _ = sess.run([model.loss, model.train_op], feed_dict)
			total_loss += loss
			
			if batch % print_every == 0 and batch > 0:
				print_loss = total_loss if batch == 0 else total_loss / print_every
				losses.append(print_loss)
				total_loss = 0
				print("-----------------------------")
				print("batch:",batch,"/",batches)
				print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
				print("loss:",print_loss)
				
				print("samples:\n")
				predict_batch = sess.run(model.out, feed_dict)
				for i in range(3):
					print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
					print("out:",[i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
					print("tar:",[i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
					print("")
		
		print(losses)
		print(saver.save(sess, "checkpoint/model.ckpt"))		
		
	


