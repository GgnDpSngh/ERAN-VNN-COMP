import os
import time
import csv
import numpy as np
import argparse
from multiprocessing import Process, Pipe, cpu_count
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', type=str, required=True, help='Path to model')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon')
parser.add_argument('--pgd_epsilon', type=float, required=True, help='Epsilon')
parser.add_argument('--im', type=int, required=True, help='Image number')
parser.add_argument('--it', type=int, default=500, help='Iterations')
parser.add_argument('--threads', type=int, default=None, help='Number of threads')
args = parser.parse_args()

csvfile = open('mnist_test_comp.csv', 'r')
tests = csv.reader(csvfile, delimiter=',')
tests = list( tests )
test = tests[ args.im ]
image= np.float64(test[1:len(test)])
corr_label = int( test[ 0 ] )
specLB = np.copy(image)
specUB = np.copy(image)
specLB -= args.epsilon
specUB += args.epsilon
specLB = np.maximum( 0, specLB )
specUB = np.minimum( 255, specUB )
pgd_args = ( args.pgd_epsilon*np.ones_like(specLB), args.pgd_epsilon*np.ones_like(specLB), 5, 100)

if args.threads is None:
    args.threads = cpu_count()

def create_pool( corr_label, args ):
    conns = []
    procs = []
    parent_pid = os.getpid()
    proc_id = 0
    for cpu in range( 10 ):
        if corr_label == cpu:
            continue
        parent_conn, child_conn = Pipe()
        conns.append( parent_conn )
        p = Process(target=thread, args=( proc_id % args.threads, cpu, args, child_conn ))
        p.start() 
        procs.append( p )
        proc_id += 1
    return conns, procs

def thread( proc_id, i, args, conn ):
    import tensorflow as tf
    from pgd_div import create_pgd_graph, pgd
    from tensorflow.contrib import graph_editor as ge
    print( 'Proc', proc_id )  
    os.sched_setaffinity(0,[proc_id])
    
    model_path = args.model
    
    sess = tf.Session()
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g = tf.import_graph_def(graph_def, name='')
    tf_out = sess.graph.get_operations()[-1].inputs[0]
    tf_in_new = tf.placeholder( shape=(784), dtype=tf.float64, name='x' )
    tf_in_old = tf.reshape( tf_in_new, (1,1,1,784) )
    tf_in_old = tf.cast( tf_in_old, tf.float32 )
    tf_in = tf.get_default_graph().get_tensor_by_name( 'input:0' )

    tf_output = ge.graph_replace(tf_out, {tf_in: tf_in_old})
    tf_output = tf.cast( tf_output, tf.float64 )

    pgd_obj = create_pgd_graph( specLB, specUB, sess, tf_in_new, tf_output, i )
    for j in range( args.it ):
        ex = pgd(sess, specLB, specUB, *pgd_obj, *pgd_args)
        status = conn.poll(0.001)
        if status:
            if conn.recv() == 'kill':
                return
        if np.argmax( sess.run( tf_output, feed_dict={tf_in_new: ex} )) == i:
            conn.send( (i,ex) )
            while True:
                status = conn.poll(1)
                if status:
                    if conn.recv() == 'kill':
                        return
    conn.send( (i,False) )
    
    while True:
        status = conn.poll(1)
        if status:
            if conn.recv() == 'kill':
                return

start = time.time()
print("img", args.im)
conns, procs = create_pool( corr_label, args )

mapping = []
for conn in range( len( conns ) ):
    mapping.append( True )

while True:
    if not np.any( mapping ):
        break
    for i in range( len( conns ) ):
        conn = conns[i]
        if mapping[ i ]:
            status = conn.poll(0.1)
            if status:
                res = conn.recv()
                mapping[ i ] = False
                conn.send( 'kill' )
                if not ( res[1] is False ):
                    print( 'Attack found for', res[0], ':' )
                    print( res[1] )
                    for i in range( len( conns ) ):
                        if mapping[ i ]:
                            conn = conns[i]
                            conn.send( 'kill' )
                    for proc in procs:
                        proc.join()
                    end = time.time()
                    print(end - start, "seconds")
                    exit()
                #else:
                    #print( 'No attacks for', res[0] ) 

print( 'No attacks' )
end = time.time()
print(end - start, "seconds")
