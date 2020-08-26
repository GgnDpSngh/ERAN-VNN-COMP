import tensorflow as tf
import numpy as np
from tensorflow.contrib import graph_editor as ge

# https://arxiv.org/pdf/2003.06878.pdf
def create_pgd_graph( lb, ub, sess, tf_input, tf_output, target):
    # Replace graph
    tf_image = tf.Variable(lb, trainable=True)
    tf_output = ge.graph_replace(tf_output, {tf_input: tf_image + 0.0})

    # Output diversification
    tf_dir = tf.placeholder( shape=(tf_output.shape[1]), dtype=tf.float64 )
    tf_eps_init = tf.placeholder( shape=lb.shape, dtype=tf.float64 )
    tf_init_error = tf.reduce_sum( tf_dir * tf_output )
    tf_init_grad = tf.gradients( tf_init_error, [tf_image] )[0]
    tf_train_init = tf_image + tf_eps_init * tf.sign( tf_init_grad ) 
    tf_train_init = tf.assign( tf_image, tf_train_init )
   
    # PGD
    tf_train_error = tf.keras.utils.to_categorical( target, num_classes=tf_output.shape[-1] )
    tf_eps_pgd = tf.placeholder( shape=lb.shape, dtype=tf.float64 )
    tf_train_error = tf.keras.losses.categorical_crossentropy( tf_train_error, tf_output, from_logits=True)
    tf_train_grad = tf.gradients( tf_train_error, [tf_image] )[0]
    tf_train_pgd = tf_image - tf_eps_pgd * tf.sign( tf_train_grad ) 
    tf_train_pgd = tf.assign( tf_image, tf_train_pgd )
    
    # Clip
    tf_train_clip = tf.clip_by_value( tf_image, lb, ub ) 
    tf_train_clip = tf.assign( tf_image, tf_train_clip )

    # Seed
    tf_seed_pl = tf.placeholder( shape=lb.shape, dtype=tf.float64 )
    tf_seed = tf.assign( tf_image, tf_seed_pl )

    return tf_image, tf_dir, tf_seed_pl, tf_eps_init, tf_eps_pgd, tf_output, tf_train_init, tf_train_pgd, tf_train_clip, tf_seed

def pgd(sess, lb, ub, 
        tf_image, tf_dir, tf_seed_pl, tf_eps_init, tf_eps_pgd, 
        tf_output, tf_train_init, tf_train_pgd, tf_train_clip, tf_seed, 
        eps_init, eps_pgd, odi_its, pgd_its):

    seed = np.random.uniform( lb, ub, size=lb.shape )
    d = np.random.uniform( -1, 1, size=(tf_output.shape[1]) )

    sess.run( tf_seed, feed_dict={ tf_seed_pl: seed } )
    for i in range(odi_its):
        sess.run( tf_train_init, feed_dict={tf_dir : d, tf_eps_init : eps_init} )
        sess.run( tf_train_clip )
    seed = sess.run( tf_image )

    sess.run( tf_seed, feed_dict={ tf_seed_pl: seed } )
    for i in range(pgd_its):
        sess.run( tf_train_pgd, feed_dict={tf_eps_pgd : eps_pgd} )
        sess.run( tf_train_clip )
    seed = sess.run( tf_image )

    return seed
