from visual import *

#Create a mapping from Image Name to Image Id in img_vocab; the mapping can be videoId_framenum->index, this index will be used for embedding
img_vocab = {}

#Load Image Embeddings
imgEmbeddings = loadImgVectors(img_vocab, filepath = "*.txt")

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

num_frames = 50
num_layers = 3
hidden_state_size = 1024
num_glimpses = 6
batch_size = 
scale = 1
fn_reward = -10
fp_reward = -0.1

def build_graph(batch_size, num_classes=len(vocab)):    #num_classes should be equal to len(vocab)

    reset_graph()

    # Placeholders
    img_placeholder = tf.placeholder(tf.int32, [batch_size,]) #the first frame to look at for all videos in the batch
    video_indices = tf.placeholder(tf.int32, [batch_size,]) #the offset needed for embed indices
    label_placeholder = tf.placeholder(, [batch_size,])
    dropout = tf.constant(0.7)

    #Embeddings
    img_embeddings =  tf.Variable(imgEmbeddings, dtype=tf.float32)
    img_inputs = tf.nn.embedding_lookup(img_embeddings, img_placeholder)

    #Initialize the LSTM network
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_state_size, state_is_tuple=False)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers, state_is_tuple=False)
    lstm_input = img_inputs
    state = stacked_lstm.zero_state(batch_size, tf.float32)

    #Save these for usage later
    lstm_results = []
    predictions = {}
    used_frames = {}

    #Unroll the LSTM network
    for i in range(num_glimpses):
        # One glimpse
        output, state = stacked_lstm(lstm_input, state)
        
        #Dropout
        top_h = tf.nn.dropout(state, keep_prob)
        
        #Next Location - l_n+1
        next_loc_mean = layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        #Reinforce Normal : Adding Gaussian Noise - next_loc = nn.ReinforceNormal(loc_std)(next_loc_mean)
        if(is_training == True):
            next_loc_mean = tf.random_normal(shape=next_loc_mean.shape, mean=0.0, stddev=loc_std)
            next_loc_mean += next_loc_mean
        else:
            next_loc = next_loc_mean
        
        #Confidence - p_n
        output_pred_dist = layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        #Reinforce Categorical - output_pred_dist = tf.nn.softmax(output_pred_dist) #avoiding this because tf.multinomial needs unnormalized log probs
        if(is_training == True):
            output_indices = tf.multinomial(output_pred_dist, 1)
        else:
            output_indices = tf.reduce_max(output_pred_dist)
        output_pred = tf.one_hot(indices=output_indices, depth=2)

        #d_n - (s_n, e_n, c_n)
        pred = layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        conf = layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=tf.nn.sigmoid, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        baseline = layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        
        #Store the results
        lstm_results += [(next_loc, output_pred, pred, conf, baseline)]
        predictions[t] = [output_pred, pred, conf, baseline]

        #Increment next_loc indices by video indices to get the right embedding indices?
        next_embed_indices = (video_indices-1)*50+tf.round(next_loc)
        used_frames[i] = next_embed_indices

        # Update the next frame to look at
        lstm_input = tf.nn.embedding_lookup(img_embeddings, next_embed_indices)

    final_state = state

    #Implementing this statement - reward = protos.reward_criterion:forward(predictions, label_placeholder)
    det_scores = tf.zeros((batch_size, num_glimpses))
    gt_scores = tf.zeros(batch_size,)
    for b in range(batch_size):
        gts = label_placeholder[b]
        use_pred = tf.zeros((num_glimpses, 2))
        pred = tf.zeros((num_glimpses, 2))
        conf = tf.zeros((num_glimpses, 1))

        #I don't think we can do this kind of assignment in tensorflow - can we directly copy the values form predictions?
        for s in range(num_glimpses):
            use_pred[s]=tf.identity(predictions[s][0][b])
            pred[s]=tf.identity(predictions[s][1][b])
            conf[s]=tf.identity(predictions[s][2][b])

        use_pred_true = tf.greater(tf.gather(use_pred,1),0.5)
        nonzero_use_pred_true = tf.where(use_pred_true)
        if tf.equal(tf.reduce_sum(use_pred_true)) == 0:
            if gts.shape[0] > 0:
                gt_scores[b] = fn_reward
            continue

        use_pred_idxs = nonzero_use_pred_true[:,0]
        pred = pred:index(1,use_pred_idxs)
        conf = conf:index(1,use_pred_idxs)

        sorted_conf,sorted_idxs=tf.sort(conf, 1, True)
        sorted_pred = {}
        for i in range(sorted_idxs.shape[0]):
            table.insert(sorted_pred, pred[sorted_idxs[i][1]])

        if sorted_pred.shape[0] > 0:

            indfree=tf.ones(sorted_pred.shape[0])
            ov = interval_overlap(gts, sorted_pred)

            for k in range(gts.shape[0]):
                indfree_idxs = tf.where(indfree)
                if indfree_idxs.shape[0] == 0: #there are no free indices
                   continue

                free_dets = indfree_idxs[:0]
                free_ov = ov[k]:index(1,free_dets)
                max_ov, max_idx = tf.reduce_max(free_ov, axis = 1)

                if max_ov[0] > 0.5:
                   max_idx = free_dets[max_idx[1]]
                   indfree[max_idx] = 0
                   sorted_idx = sorted_idxs[max_idx][0]
                   orig_idx = use_pred_idxs[sorted_idx]
                   det_scores[b][orig_idx] = 1

            unused_dets = tf.where(indfree)
            if unused_dets.shape[0] > 0:
                num_unused_dets = unused_dets.shape[0]
                for i in range(num_unused_dets):
                    sorted_idx = sorted_idxs[unused_dets[i][0]][0]
                    orig_idx = use_pred_idxs[sorted_idx]
                    det_scores[b][orig_idx] = fp_reward

    rewards = tf.reduce_sum.sum(det_scores, axis = 2)
    rewards = rewards + gt_scores
    rewards = rewards * scale

    #Implementing this statement - loss = protos.pred_loss_criterion:forward({predictions,used_frames,opt.seq_len}, y)
    baseline_loss = tf.squared_difference(predictions[opt.num_glimpses][4], reward)

    avg_reward = torch.sum(reward)/batch_size
    avg_loss = torch.sum(loss)/batch_size