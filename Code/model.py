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
loc_weight = 1

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
        next_loc_mean = tf.layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        #Reinforce Normal : Adding Gaussian Noise - next_loc = nn.ReinforceNormal(loc_std)(next_loc_mean)
        if(is_training == True):
            next_loc_mean = tf.random_normal(shape=next_loc_mean.shape, mean=0.0, stddev=loc_std)
            next_loc_mean += next_loc_mean
        else:
            next_loc = next_loc_mean
        
        #Confidence - p_n
        output_pred_dist = tf.layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        #Reinforce Categorical - output_pred_dist = tf.nn.softmax(output_pred_dist) #avoiding this because tf.multinomial needs unnormalized log probs
        if(is_training == True):
            output_indices = tf.multinomial(output_pred_dist, 1)
        else:
            output_indices = tf.reduce_max(output_pred_dist)
        output_pred = tf.one_hot(indices=output_indices, depth=2)

        #d_n - (s_n, e_n, c_n)
        pred = tf.layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        conf = tf.layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=tf.nn.sigmoid, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        baseline = tf.layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        
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
    gt_locs = tf.zeros((batch_size, num_glimpses, 2))
    loc_diffs = tf.zeros((batch_size, num_glimpses, 2))
    clf_losses = tf.zeros((batch_size, num_glimpses, 1))
    loc_losses = tf.zeros((batch_size, num_glimpses, 2))

    for b in range(batch_size):
        gts = tf.gather_nd(label_placeholder, b) #label_placeholder[b]
        gt_mapping = tf.zeros(seq_len)
        num_gts = gts.shape[0]
        for i in range(num_gts):
            cur_gt_start = tf.gather_nd(gts, [[i,1]]) #gts[i][1]
            mapping_start = 0
            if i>1:
                prev_gt_end = tf.gather_nd(gts, [[i-1][2]]) #gts[i-1][2]
                mapping_start = math.floor((prev_gt_end + cur_gt_start)/2)
            cur_gt_end = gts[i][2]
            mapping_end = seq_len
            if i < num_gts:
                next_gt_start = gts[i+1][1]
                mapping_end = math.floor((next_gt_start + cur_gt_end)/2)
            for s=mapping_start+1,mapping_end do
                gt_mapping[s] = i

        #can use tf.slice here - not sure
        pred = tf.gather_nd(predictions, [:,1,b])
        conf = tf.gather_nd(predictions, [:,2,b])
        # pred = tf.zeros((num_glimpses, 2))
        # conf = tf.zeros((num_glimpses, 1))
        # for s in range(num_glimpses):
        #     pred[s]:copy(predictions[s][2][b])
        #     conf[s]:copy(predictions[s][3][b])

        if num_gts > 0:
            clf_losses[b] = tf.log(conf+1e-12)
        else:
            clf_losses[b] = tf.log((conf*-1)+1+1e-12)

        if num_gts > 0:
            for s in range(num_glimpses):
                gt_idx = gt_mapping[used_frames[s][b][1]]
                gt_locs[b][s][1] = gts[gt_idx][1] / seq_len
                gt_locs[b][s][2] = gts[gt_idx][2] / seq_len
            loc_diffs[b] = pred - gt_locs[b]
            loc_losses[b] = torch.pow(loc_diffs[b],2):mul(loc_weight)
    clf_losses *= -1 # neg log likelihood
    loss = clf_losses + tf.reduce_sum(loc_losses,3)

    baseline_diff = predictions[num_glimpses][4] - reward
    baseline_loss = tf.square(predictions[num_glimpses][4], reward)
    avg_reward = tf.reduce_sum(reward)/batch_size
    avg_loss = tf.reduce_sum(loss)/batch_size

    ## Backward Pass
    drnn_state = {[num_glimpses] = {}}
    for i=init_hidden_offset+1,#init_state do
        table.insert(drnn_state[opt.num_glimpses], init_state[i]:clone():zero())
    end
    
    ## Implementing this line here - daction = protos.reward_criterion:backward(predictions,y)
    seq_len = len(predictions)
    baseline = predictions[seq_len][4]:double()
    rewardsRel = rewards
    rewardsGrad = rewardsRel:mul(-1)

    gradInput = {}
    for s=1,seq_len do
        gradInput[s] = {}
        gradInput[s][1] = torch.repeatTensor(rewardsGrad, 1,1) #dLoc
        gradInput[s][2] = torch.repeatTensor(rewardsGrad, 1,2) #dAction
    daction = gradInput
    
    ## Implementing this line here - local doutput = protos.pred_loss_criterion:backward(predictions,y)
    seq_len = #input

    gradInput = {}
    for s in range(seq_len):
        gradInput[s] = {}
        gradInput[s][1] = torch.Tensor(batch_size,2):zero()
        gradInput[s][2] = torch.Tensor(batch_size,1):zero()

        conf = predictions[s][3]
        for b in range(batch_size):
            gts = target[b]
            num_gts = gts.shape[0]
            if num_gts > 0:
                gradInput[s][1][b] = loc_diffs[b][s]:mul(2):mul(loc_weight)
                gradInput[s][2][b] = -1 / (conf[b][1] + 1e-12)
            else
                gradInput[s][1][b] = 0
                gradInput[s][2][b] = 1 / (1 - conf[b][1] + 1e-12)

    doutput = gradInput

    ##Implementing this line here - local dbaseline = protos.baseline_loss_criterion:backward(predictions[opt.num_glimpses][4], reward)
    dbaseline = 2*baseline_diff

    for t in reversed(range(num_glimpses)):
        doutput_t = tf.gather_nd(doutput, t) #doutput[t]
        daction_t = tf.gather_nd(daction, t) #daction[t]
        if t == num_glimpses:
            dbaseline_t = dbaseline
        else
            dbaseline_t = tf.zeros((batch_size, 1))

        #GPU for Lua
        # if opt.gpuid >= 0:
        #     daction_t[1] = daction_t[1]:cuda() #next_loc
        #     daction_t[2] = daction_t[2]:cuda() #use_pred
        #     doutput_t[1] = doutput_t[1]:cuda() #pred
        #     doutput_t[2] = doutput_t[2]:cuda() #conf
        #     dbaseline_t = dbaseline_t:cuda() #baseline
        table.insert(drnn_state[t], daction_t[1]) #next_loc
        table.insert(drnn_state[t], daction_t[2]) #use_pred
        table.insert(drnn_state[t], doutput_t[1]) #pred
        table.insert(drnn_state[t], doutput_t[2]) #conf
        table.insert(drnn_state[t], dbaseline_t) -- lstm baseline
        local dlst = clones.rnn[t]:backward({input_data[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > (1+init_hidden_offset) then -- k == 1 is gradient on x, which we dont need, k=2 is loc
                drnn_state[t-1][k-2] = v
    
    # clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return {avg_reward, avg_loss}, grad_params
