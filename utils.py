import tensorflow as tf

#----------------------------- cal attention -------------------------------
def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.mul(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.mul(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.mul(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.mul(norm_q, norm_a))
    return cos_sim_q_a

# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
	height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

	# do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
	lstm_out = tf.expand_dims(lstm_out, -1)
	output = tf.nn.max_pool(
		lstm_out,
		ksize=[1, height, 1, 1],
		strides=[1, 1, 1, 1],
		padding='VALID')

	output = tf.reshape(output, [-1, width])

	return output

def avg_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
    
    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
    	lstm_out,
    	ksize=[1, height, 1, 1],
    	strides=[1, 1, 1, 1],
    	padding='VALID')
    
    output = tf.reshape(output, [-1, width])
    
    return output

def cal_loss_and_acc(ori_cand, ori_neg):
    # the target function 
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), 0.2)
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.sub(margin, tf.sub(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses) 
    # cal accurancy
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc


def get_feature(input_q, input_a, att_W):
	h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
	h_a = int(input_a.get_shape()[1])

        output_q = max_pooling(input_q)

	reshape_q = tf.expand_dims(output_q, 1)
	reshape_q = tf.tile(reshape_q, [1, h_a, 1])
	reshape_q = tf.reshape(reshape_q, [-1, w])
	reshape_a = tf.reshape(input_a, [-1, w])

	M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
	M = tf.matmul(M, att_W['Wms'])

	S = tf.reshape(M, [-1, h_a])
	S = tf.nn.softmax(S)

	S_diag = tf.matrix_diag(S)
	attention_a = tf.batch_matmul(S_diag, input_a)
	attention_a = tf.reshape(attention_a, [-1, h_a, w])

        output_a = max_pooling(attention_a)

	return tf.tanh(output_q), tf.tanh(output_a)

