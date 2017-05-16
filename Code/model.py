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


def reinforce_categorical()

def build_graph(batch_size, num_classes=len(vocab)):    #num_classes should be equal to len(vocab)

    reset_graph()

    # Placeholders
    img_placeholder = tf.placeholder(tf.int32, [batch_size, num_frames])
    dropout = tf.constant(0.7)

    #Embeddings
    img_embeddings =  tf.Variable(imgEmbeddings, dtype=tf.float32)
    img_inputs = tf.nn.embedding_lookup(img_embeddings, img_placeholder)

    #Unroll the LSTM network
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_state_size, state_is_tuple=False)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers, state_is_tuple=False)
    lstm_input = img_inputs
    lstm_results = []
    state = stacked_lstm.zero_state(batch_size, tf.float32)
    for i in range(num_glimpses):
        # One glimpse
        output, state = stacked_lstm(lstm_input, state)
        
        #Dropout
        top_h = tf.nn.dropout(state, keep_prob)
        
        #Next Location - l_n+1
        next_loc_mean = layers.fully_connected(inputs=top_h, num_outputs=1, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        next_loc = nn.ReinforceNormal(loc_std)(next_loc_mean)
        
        #Confidence - p_n
        output_pred_dist = layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        output_pred_dist = tf.nn.softMax(output_pred_dist)
        output_pred = nn.ReinforceCategorical()(output_pred_dist)

        #d_n - (s_n, e_n, c_n)
        pred = layers.fully_connected(inputs=top_h, num_outputs=2, activation_fn=None, weights_initializer = layers.xavier_initializer(), biases_initializer = tf.constant_initializer(0.1))
        conf = nn.Sigmoid()(nn.Linear(rnn_size, 1)(top_h))
        baseline = nn.Linear(rnn_size, 1)(top_h)
        
        #Store the results
        lstm_results += [(next_loc, )]
        
        # Update the next frame to look at
        lstm_input = 

    final_state = state