###  This file is to read data by tensorflow which was faster than before
# -------------------- Done -------------

import tensorflow as tf 
import numpy as np 
import datetime 



def get_shuffle_batch(file_data,file_label,data_size,batch_size,epoches=4,label_size=2):
    '''
    For getting batch_size data and labels from file_data under tensorflow
    if U use this file you should add Coordinator() in the session as:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        ...
        M,M2=sess.run([example2,labels2])
        ...
        coord.request_stop()
        coord.join(threads)
     Arg: 
        file_data: the path for the encoded data
        file_label: the path for the labels data 
        data_size: the size of every line of the input data
        batch_size: the batch_size for return 
        label_size: the size of every label 
     Return : 
        two tensor
        example2: a tensor with shape of [batch_size,data_size] for data
        labels2: a tensor with shape of [batch_size,label_size] for label
    '''    
    filenames = file_data
    labels=file_label

    filename_queue = tf.train.string_input_producer(filenames,shuffle=False,num_epochs=epoches)
    filename_queue2 = tf.train.string_input_producer(labels,shuffle=False,num_epochs=epoches)

    print ('start:',datetime.datetime.now())

    reader = tf.TextLineReader()
    reader2=tf.TextLineReader()

    key, value = reader.read(filename_queue)
    key2, value2 = reader2.read(filename_queue2)

    # if the paras have the same numbers of record_defaults, they will match 
    example= tf.decode_csv(value, record_defaults=[[] for col in range (data_size) ])
    label= tf.decode_csv(value2, record_defaults=[[] for _ in range(label_size)])

    #  --------- random batch ------------------
    # example_batch,label_batch= tf.train.shuffle_batch([example,label], batch_size=4,capacity=50000, min_after_dequeue=400,num_threads=4)
    #  --------- batch in order --------------
    example2,labels2=tf.train.shuffle_batch([example,label],batch_size=batch_size,capacity=200,min_after_dequeue=100,num_threads=2)
    return example2,labels2


def get_batch(file_data,file_label,data_size,batch_size,epoches=None,label_size=2):
    '''
    For getting batch_size data and labels from file_data under tensorflow
    if U use this file you should add Coordinator() in the session as:
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        ...
        M,M2=sess.run([example2,labels2])
        ...
        coord.request_stop()
        coord.join(threads)
     Arg: 
        file_data: the path for the encoded data
        file_label: the path for the labels data 
        data_size: the size of every line of the input data
        batch_size: the batch_size for return 
        label_size: the size of every label 
     Return : 
        two tensor
        example2: a tensor with shape of [batch_size,data_size] for data
        labels2: a tensor with shape of [batch_size,label_size] for label
    '''    
    filenames = file_data
    labels=file_label

    filename_queue = tf.train.string_input_producer(filenames,shuffle=False,num_epochs=epoches)
    filename_queue2 = tf.train.string_input_producer(labels,shuffle=False,num_epochs=epoches)

    print ('start:',datetime.datetime.now())

    reader = tf.TextLineReader()
    reader2=tf.TextLineReader()

    key, value = reader.read(filename_queue)
    key2, value2 = reader2.read(filename_queue2)

    # if the paras have the same numbers of record_defaults, they will match 
    example= tf.decode_csv(value, record_defaults=[[] for col in range (data_size) ])
    label= tf.decode_csv(value2, record_defaults=[[] for _ in range(label_size)])

    #  --------- random batch ------------------
    # example_batch,label_batch= tf.train.shuffle_batch([example,label], batch_size=4,capacity=50000, min_after_dequeue=400,num_threads=4)
    #  --------- batch in order --------------
    example2,labels2=tf.train.batch([example,label],batch_size=batch_size,capacity=6000)
    return example2,labels2





