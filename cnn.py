from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

#数据集地址
path='/Users/wangyingbo/Downloads/flower_photos'
#模型保存地址
model_path='/Users/wangyingbo/Downloads/flower_model/model.ckpt'

#将所有的图片resize成100*100
width=100
height=100
channel=3


#读取图片
def read_img(path):
    classes = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images=[]
    labels=[]
    for label,folder in enumerate(classes):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(width,height))
            images.append(img)
            labels.append(label)
    return np.asarray(images,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


ratio = 0.8
s = np.int(ratio * num_example)
x_train = data[:s]
y_train = label[:s]

x_validation = data[s:]
y_validation = label[s:]

#构建网络
x = tf.placeholder(tf.float32,shape=[None,width,height,channel],name='x')
y_ = tf.placeholder(tf.int32,shape=[None,],name = 'y_')

def inference(input_tensor,train,regulaizer):
    with tf.variable_scope('conv-layer1'):
        conv1_weight = tf.get_variable('weight',[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biase = tf.get_variable('bias',[32],initializer = tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weight,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biase))

    with tf.name_scope('pool-layer1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope('conv-layer2'):
        conv2_weight = tf.get_variable('weight',[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biase = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weight,strides=[1,1,1,1],padding='SAME')
        relue2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biase))
    
    with tf.name_scope('conv-pool2'):
        pool2 = tf.nn.max_pool(relue2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    with tf.variable_scope('conv_layer3'):
        con3_weight = tf.get_variable('weight',[3,3,64,128],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv3_biase = tf.get_variable('bias',[128],initializer = tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2,con3_weight,strides=[1,1,1,1],padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biase))
    
    with tf.name_scope('pool_layer3'):
        pool3 = tf.nn.max_pool(relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    with tf.variable_scope('conv_layer4'):
        conv4_weight = tf.get_variable('weight',[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biase = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3,conv4_weight,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biase))

    with tf.name_scope('pool_layer4'):
        pool4 = tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        nodes = 6*6*128
        reshaped = tf.reshape(pool4,[-1,nodes])
    
    #全连接层
    with tf.variable_scope('full_layer1'):
        full1_weight = tf.get_variable('weight',[nodes,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regulaizer != None:
            tf.add_to_collection('losses', regularizer(full1_weight))
        biase1 = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.0))
        full1 = tf.nn.relu(tf.matmul(reshaped,full1_weight) + biase1)
        if train:
            full1 = tf.nn.dropout(full1,0.5)

    with tf.variable_scope('full_layer2'):
        full2_weight = tf.get_variable('weight',[1024,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regulaizer != None:
            tf.add_to_collection('losses', regularizer(full2_weight))
        biase2 = tf.get_variable('bias',[512],initializer = tf.constant_initializer(0.0))
        full2 = tf.nn.relu(tf.matmul(full1,full2_weight) + biase2)
        if train:
            full2 = tf.nn.dropout(full2,0.5)
    
    with tf.variable_scope('full_layer3'):
        full3_weight = tf.get_variable('weight',[512,5],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regulaizer != None:
            tf.add_to_collection('losses', regularizer(full3_weight))
        biase3 = tf.get_variable('bias',[5],initializer = tf.constant_initializer(0.0))
        logit = tf.nn.relu(tf.matmul(full2,full3_weight) + biase3)
       

    return logit

#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logit  = inference(x,False,regularizer)

#(小处理)将logit 乘以1赋值给logit_eval,定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logit_eval = tf.multiply(logit,b,name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit,labels = y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logit,1),tf.int32),y_)
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练和测试数据，可将n_epoch设置更大一些

n_epoch=10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
batch_size=64
saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_validation, y_validation, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
saver.save(sess,model_path)
sess.close()





