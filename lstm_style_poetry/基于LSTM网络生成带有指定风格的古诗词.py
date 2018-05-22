
# coding: utf-8

# # 基于LSTM网络生成带有指定风格的古诗词

# ## 一、	**项目环境**
# 1.	python 3.6
# 2.	tensorflow 1.2
# 3.	mac环境
# 4.	诗词预料包

# ## 二、	项目简介
# 通过构建LSTM神经网络，训练神经网络，建立一个语言模型，学习古诗词的语言特征。同时让模型学习到不同风格的古诗词，并且提取相应的特质，结合到新的古诗词中。整个项目分为4个模块，如下:
# 
# LSTM_model.py：LSTM网络模型，提供了end_points接口，被其他部分调用
# 
# poetry_porcess.py：数据读取、预处理部分，会返回打包好的batch，被main调用
# 
# gen_poetry.py：古诗生成程序，拥有可选的风格参数，被main调用
# 
# main.py：主函数，既可以调用前两个程序获取预处理数据并使用LSTM网络进行训练，也可以调用gen_poetry.py生成古诗
# 
# 在`main.py`最后有如下指令，
# ```Python
# if __name__ == "__main__":
#     words,poetry_vector,to_num,x_batches,y_batches = poetry_porcess.poetry_process()
#     # train(words, poetry_vector, x_batches, y_batches)
#     # gen_poetry(words, to_num)
#     generate(words_, to_num_, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
# ```
# 此时实际上处于生成模式，对于最后的三行:
# 
# train：表示训练
# 
# gen_poetry：表示根据首字符生成
# 
# generate：表示根据首句和风格句生成古诗
# 
# 需要实现哪个功能就将其它功能注释掉。接下来让我们具体的讨论每一个模块的功能与实现。
# 
# 由于方便在线演示，在文章的最后，我们将所有功能整合为一个python文件，进行运行。

# ## 三、文本预处理
#   本节来简单介绍一下中文文字的预处理流程，其实有一大部分深度学习中，对于中文文本的处理就是接下来要介绍的方法，所以该方法具有普遍的参考意义，但也有部分项目通过word2vec来进行文本的预处理，本文介绍的是通过文字与ID的映射，再通过embedding来训练文本。整个部分分为两个部分：第一部分就文本的处理，主要的任务就是读取预料，进行一些简单的去杂操作。第二部分就是对文档进行文字与ID的映射，即辅助数据结构生成。
# 

# 第一部分：文本处理，重点在于读取+去除特殊符号
# 
# 核心代码：

# In[3]:

with open(poetry_file,'r',encoding='utf-8') as f:
        # 按行读取古诗词
        for line in f:
            try:
                # 将古诗词的诗名和内容分开
                title,content = line.strip().split(':')
                # 去空格，实际上没用到，但为了确保文档中没有多余的空格，还是建议写一下
                content = content.replace(' ','') 
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                # 选取长度在5～79之间的古诗词
                if len(content) < 5 or len(content) > 79:
                    continue
                # 添加首尾标签
                content = 'B' + content + 'E'
                # 添加到poetrys列表中
                poetrys.append(content)
            except Exception as e:
                pass


# 第二部分：辅助数据结构生成
# 
# 1.使用counter = collections.Counter()函数对所有字符进行计数。
# 
# 核心代码：
# 

# In[ ]:

all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = Counter(all_words)


# 2.通过count_pairs = sorted(counter.items(), key=lambda x: -x[1])对计数结果进行排序，返回的结果是一个tuple
# 
# 3.通过words, _ = zip(*count_pairs)对tuple进行解压，得到words列表代表所有字符。该字符的行号即为该字符索引
# 
# 4.通过word_int_map = dict(zip(words, range(len(words))))，得到字符与行号对应的索引字典
# 
# 5.通过poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]函数将字符数据集映射成为索引
# 
# 核心代码：
# 

# In[ ]:

words = words[:len(words)] + (' ',)  # 后面要用' '来补齐诗句长度
# print(words)
# 字典:word->int
# 每个字映射为一个数字ID  
word_num_map = dict(zip(words,range(len(words))))
# 把诗词转换为向量
to_num = lambda word: word_num_map.get(word,len(words))
#print(to_num)
poetry_vector = [list(map(to_num,poetry)) for poetry in poetrys]



# ## 四、 预料准备
#   该深度学习是一个有监督（带标签的）的学习模型，所以我们需要准备训练数据集。由于生成古诗词的本质是一种语言模型的建立，简单来说就是给定第一个字，生成第二个字，生成第三个字，从而生成一句诗词，使得该诗词符合自然语言的表达。所以我们准备的预料也是前一个字是输入，而后一个则是我们的预测数据，这样的模型我们有一个专门的模型名称，叫做序列模型，就想流水一样，一个字接一个字。到此为止，我们已经介绍了我们模型需要的数据集，还有一个问题，就是当我们进行训练的时候，需要定义batch大小，通过并行化提高内存利用率。让我们来实现这两个要求。
# 

# 先来求出诗向量的大小

# In[ ]:

n_chunk = len(poetry_vector) // batch_size
 #同时定义我们的输入数据X，以及输出数据Y
   x_batches = []
y_batches = []
 #通过start和end标签来实现batch大小的设置。
   for i in range(n_chunk):
       start_index = i * batch_size #起始位置
       end_index = start_index + batch_size #结束位置
       batches = poetry_vector[start_index:end_index]
       length = max(map(len,batches))  # 记录下最长的诗句的长度
 #补全同一批测试集数据中的句子长度大小
       xdata = np.full((batch_size,length),word_num_map[' '],np.int32)
 #实现前一个字与后一个字之间的生成关系
       ydata = np.copy(xdata)
       ydata[:,:-1] = xdata[:,1:]
       """
           xdata             ydata
           [6,2,4,6,9]       [2,4,6,9,9]
           [1,4,2,8,5]       [4,2,8,5,5]
       """


# ## 五、模型建立
#   本节要讨论是如何用tensorflow建立一个lstm模型。
#   
#   首先需要定义一个带有LSTM的RNN网络。 
#   
# 定义如下：
# 

# In[ ]:

rnn_model(num_of_word,input_data,output_data=None,rnn_size=128,num_layers=2,batch_size=128):


# 其中，参数意义如下：
# param num_of_word: 词的个数
# 
# param input_data: 输入向量
# 
# param output_data: 标签
# 
# param rnn_size: 隐藏层的向量尺寸
# 
# param num_layers: 隐藏层的层数
# 
# param batch_size: 批处理数据大小

# 然后定义LSTM的核心功能，定义如下：

# In[ ]:

cell_fun = tf.contrib.rnn.BasicLSTMCell


# 首先使用tf.nn.rnn_cell.BasicLSTMCell定义单个基本的LSTM单元。这里的size其实就是hidden_size。 在LSTM单元中，有2个状态值，分别是c和h，分别对应于下图中的c和h。其中h在作为当前时间段的输出的同时，也是下一时间段的输入的一部分。
# <img src="./图片 1.png" style="max-width:60%;"/>

# 那么当state_is_tuple=True的时候，state是元组形式，state=(c,h)。如果是False，那么state是一个由c和h拼接起来的张量，state=tf.concat(1,[c,h])。在运行时，则返回2值，一个是h，还有一个state。

# In[ ]:

cell = cell_fun(rnn_size,state_is_tuple=True) 


# 在这个示例中，我们使用了2层的LSTM网络。也就是说，前一层的LSTM的输出作为后一层的输入。使用tf.nn.rnn_cell.MultiRNNCell可以实现这个功能。这个基本没什么好说的，state_is_tuple用法也跟之前的类似。构造完多层LSTM以后，使用zero_state即可对各种状态进行初始化。

# In[ ]:

cell =tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)
    # 如果有标签(output_data)则初始化一个batch的cell状态，否则初始化一个
    if output_data is not None:
        initial_state = cell.zero_state(batch_size,tf.float32)
    else:
        initial_state = cell.zero_state(1,tf.float32)


# 接下来我们进行词向量的嵌入
# 
# •	构造一个num_of_word＋1 x rnn_size的矩阵，作为embedding容器
# 
# •	有num_of_word＋1个容量为rnn_size的向量，每个向量代表一个vocabulary，每个向量的中的分量的值都在-1到1之间随机分布
# 
# 之前有提到过，输入模型的input和target都是用词典id表示的。例如一个句子，“我/是/学生”，这三个词在词典中的序号分别是0,5,3，那么上面的句子就是[0,5,3]。显然这个是不能直接用的，我们要把词典id转化成向量,也就是embedding形式。实现的方法很简单。
# 
# •	第一步，构建一个矩阵，就叫embedding好了，尺寸为[vocab_size, embedding_size]，分别表示词典中单词数目，以及要转化成的向量的维度。一般来说，向量维度越高，能够表现的信息也就越丰富。
# 
# •	第二步，使用tf.nn.embedding_lookup(embedding,input_ids) 假设input_ids的长度为len，那么返回的张量尺寸就为[len,embedding_size]。
# 
# 具体实现：
# 

# In[ ]:

embedding =tf.get_variable('embedding',initializer=tf.random_uniform([num_of_word + 1,rnn_size],-1.0,1.0))
inputs = tf.nn.embedding_lookup(embedding,input_data)


# 在完成embedding工作之后，让我们来考虑一个问题，在每一个train step，传入model的是一个batch的数据（这一个batch的数据forward得到predictions，计算loss，backpropagation更新参数），这一个batch内的数据一定是padding成相同长度的。
# 那么，如果可以只在一个batch内部进行padding，例如一个batch中数据长度均在6-10这个范围内，就可以让这个batch中所有数据pad到固定长度10，而整个dataset上的数据最大长度很可能是100，这样就不需要让这些数据也pad到100那么长，白白浪费空间。
# 我们考虑dynamic_rnn，这个函数实现的功能就是可以让不同迭代传入的batch可以是长度不同数据，但同一次迭代一个batch内部的所有数据长度仍然是固定的。例如，第一时刻传入的数据shape=[batch_size, 10]，第二时刻传入的数据shape=[batch_size, 12]，第三时刻传入的数据shape=[batch_size, 8]等等。
# 但是rnn不能这样，它要求每一时刻传入的batch数据的[batch_size, max_seq]，在每次迭代过程中都保持不变。
# 所以我们来实现一下这个功能：

# In[ ]:

outputs,last_state =
tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
output = tf.reshape(outputs,[-1,rnn_size])


# 接着我们来看一下W，b的参数设计：

# In[ ]:

weights = tf.Variable(tf.truncated_normal([rnn_size,num_of_word + 1]))
bias = tf.Variable(tf.zeros(shape=[num_of_word + 1]))
logits = tf.nn.bias_add(tf.matmul(output,weights),bias=bias)


# 最后，我们看一下对于损失函数的定义，以及优化器的定义：

# In[ ]:

loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
total_loss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss)


# 到此为止，我们的LSTM模型就搭建完了。

# ## 六、训练
# 在完成模型的搭建以后，我们就要开始进行模型的训练，定义训练函数：
# 

# In[ ]:

def train(words,poetry_vector,x_batches,y_batches):
#然后定义输入数据，以及输出标签的容器，来存放数据。
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    output_targets = tf.placeholder(tf.int32,[batch_size,None])
#同时，定义数据保存节点：
    end_points = rnn_model(len(words),input_data=input_data,output_data=output_targets,batch_size=batch_size)
#建立svaer类，用来保存训练得到的参数
    saver = tf.train.Saver(tf.global_variables())
#进行初始化
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
#为了更直观的展示训练过程，进行可视化处理，将图形、训练过程等数据合并在一起
    merge = tf.summary.merge_all()
#通过上下文管理机制，建立一个新会话，来完成
    with tf.Session(config=config) as sess:
#将训练日志写在log文件夹下
        writer = tf.summary.FileWriter('./logs',sess.graph)
#开始训练,并建立模型保存目录
        sess.run(init_op)
        start_epoch = 0
        model_dir = "./model"
        epochs = 1


# 在训练的过程中，我们很有可能在训练到一半的时候出现意外导致训练停止，所以我们要及时记录已经训练的数据，从而不用在出现意外的时候从头训练，而是接着上次训练继续，所以我们需要设置一个检查点：

# In[ ]:

checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
            print('## start training...')
   


# ## 七、	诗歌生成
# 在完成训练模型以后，我们就需要利用训练完的模型进行诗歌生成。
# 该部分代码主要包含gen_poem和to_word函数。就是将训练结束后的last_state作为initial_state传入lstm，那么新生成的output即为预测的结果。
# 首先我们需要定义一个to_word函数，大家是否还记得，我们在处理预料数据的时候，定义过一个to_mun函数，作用是将文字转化为数字，而这里的to_word函数功能恰恰相反根据生成的词向量转换为文字。
# 

# In[ ]:

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    #t的长度为vocab_size + 1, 随机生成一个数然后判断能插入第几个位置来取字
    sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]  # [np.argmax(predict)]


# 定义诗歌生成函数

# In[ ]:

def gen_poetry(words, to_num):
…
#首先获取诗歌开头的标志‘B’
        x = np.array(to_num('B')).reshape(1, 1)
#并将‘B’的词向量放入模型，得到last_state
        _, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
#输入起始诗句：
        word = input('请输入起始字符:')
        poem_ = ''
        while word != 'E':
poem_ += word
#通过to_num函数转化为数字
x = np.array(to_num(word)).reshape(1, 1)
#把数据装入一个词典，并喂给模型，一个接一个，上一次的输出是下一次的输入：
predict, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x, end_points['initial_state']: last_state})
#将得到的预测的词向量转化为文字
word = to_word(predict, words)


# 这就是通过训练过的模型，生成古诗词的全过程。
# 

# 接下来，我们来看看如果生成带有指定风格的古诗词，其实主要过程和上面一样，唯一不一样的就是如何添加我们的指定风格的诗词风。让我们来定义带有指定风格的诗歌生成函数

# In[ ]:

def generate(words, to_num, style_words="狂沙将军战燕然，大漠孤烟黄河骑。"):
…
        if style_words:
for word in style_words:
#将指定风格的诗句，单独放入模型进行训练，原理一样。
                x = np.array(to_num(word)).reshape(1, 1)
                last_state = sess.run(end_points['last_state'],
                                      feed_dict={input_data: x, end_points['initial_state']: last_state})


# ## 八、后话
# 通过这个项目，可以进一步加深对于lstm模型的理解，由于该项目是一个简单的lstm实现的应用，在古诗词生成领域，还需要考虑更多问题，比如对仗问题，押韵问题，而本项目只是考虑每个字之间的关系，对于局部特征学习较好，而对于诗歌的整体特征学习较弱，有兴趣的同学，也可以尝试加入更多学习的特征，使得生成的古诗词更符合古诗词的音韵美，结构美。

# ## 九、运行测试

# In[1]:

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import numpy as np
from collections import Counter

batch_size = 64
poetry_file = './poems.txt'

'''
第一步：处理古诗词文本
:batch_size:batch大小
:poetry_file:古诗词文本位置
'''

def poetry_process(batch_size=64,poetry_file='./poems.txt'):
    # 处理完的古诗词放入poetrys列表中
    poetrys = []
    # 打开文件
    with open(poetry_file,'r',encoding='utf-8') as f:
        # 按行读取古诗词
        for line in f:
            try:
                # 将古诗词的诗名和内容分开
                title,content = line.strip().split(':')
                # 去空格，实际上没用到，但为了确保文档中没有多余的空格，还是建议写一下
                content = content.replace(' ','') 
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                # 选取长度在5～79之间的古诗词
                if len(content) < 5 or len(content) > 79:
                    continue
                # 添加首尾标签
                content = 'B' + content + 'E'
                # 添加到poetrys列表中
                poetrys.append(content)
            except Exception as e:
                pass

    # 依照每首诗的长度排序
    # poetrys = sorted(poetrys, key=lambda poetry: len(poetry))
    print('唐诗数量：',len(poetrys))

    # 统计字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = Counter(all_words)
    # print(counter.items())
    # item会把字典中的每一项变成一个2元素元组，字典变成大list
    count_pairs = sorted(counter.items(),key=lambda x: -x[1])
    # 利用zip提取，因为是原生数据结构，在切片上远不如numpy的结构灵活
    words,_ = zip(*count_pairs)
    # print(words)

    words = words[:len(words)] + (' ',)  # 后面要用' '来补齐诗句长度
    # print(words)
    # 字典:word->int
    # 每个字映射为一个数字ID  
    word_num_map = dict(zip(words,range(len(words))))
    # 把诗词转换为向量
    to_num = lambda word: word_num_map.get(word,len(words))
    #print(to_num)
    poetry_vector = [list(map(to_num,poetry)) for poetry in poetrys]
    # print(poetry_vector)


    # 每次取64首诗进行训练 
    n_chunk = len(poetry_vector) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size #起始位置
        end_index = start_index + batch_size #结束位置
        batches = poetry_vector[start_index:end_index]
        length = max(map(len,batches))  # 记录下最长的诗句的长度
        xdata = np.full((batch_size,length),word_num_map[' '],np.int32)
        for row in range(batch_size):
            xdata[row,:len(batches[row])] = batches[row]
        # print(len(xdata[0])) 每个batch中数据长度不相等
        ydata = np.copy(xdata)
        ydata[:,:-1] = xdata[:,1:]
        """
            xdata             ydata
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
        x_batches.append(xdata)  # (n_chunk, batch, length)
        y_batches.append(ydata)
    return words,poetry_vector,to_num,x_batches,y_batches

# Author : hellcat
# Time   : 18-3-12



def rnn_model(num_of_word,input_data,output_data=None,rnn_size=128,num_layers=2,batch_size=128):
    end_points = {}
    """

    :param num_of_word: 词的个数
    :param input_data: 输入向量
    :param output_data: 标签
    :param rnn_size: 隐藏层的向量尺寸
    :param num_layers: 隐藏层的层数
    :param batch_size: 
    :return: 
    """
    
    '''构建RNN核心'''
    # cell_fun = tf.contrib.rnn.BasicRNNCell
    # cell_fun = tf.contrib.rnn.GRUCell
    '''
BasicLSTMCell类是最基本的LSTM循环神经网络单元。 

num_units: LSTM cell层中的单元数 
forget_bias: forget gates中的偏置 
state_is_tuple: 还是设置为True吧, 返回 (c_state , m_state)的二元组 
activation: 状态之间转移的激活函数 
reuse: Python布尔值, 描述是否重用现有作用域中的变量
state_size属性：如果state_is_tuple为true的话，返回的是二元状态元祖。
output_size属性：返回LSTM中的num_units, 也就是LSTM Cell中的单元数，在初始化是输入的num_units参数
_call_()将类实例转化为一个可调用的对象，传入输入input和状态state，根据LSTM的计算公式, 返回new_h, 和新的状态new
    '''
    cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size,state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,state_is_tuple=True)

    # 如果是训练模式，output_data不为None，则初始状态shape为[batch_size * rnn_size]
    # 如果是生成模式，output_data为None，则初始状态shape为[1 * rnn_size]
    if output_data is not None:
        initial_state = cell.zero_state(batch_size,tf.float32)
    else:
        initial_state = cell.zero_state(1,tf.float32)

    # 词向量嵌入
    embedding = tf.get_variable('embedding',initializer=tf.random_uniform([num_of_word + 1,rnn_size],-1.0,1.0))
    inputs = tf.nn.embedding_lookup(embedding,input_data)
        
    
    outputs,last_state = tf.nn.dynamic_rnn(cell,inputs,initial_state=initial_state)
    output = tf.reshape(outputs,[-1,rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size,num_of_word + 1]))
    bias = tf.Variable(tf.zeros(shape=[num_of_word + 1]))
    logits = tf.nn.bias_add(tf.matmul(output,weights),bias=bias)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data,[-1]),depth=num_of_word + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss)
        tf.summary.scalar('loss',total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction
    return end_points

def to_word(predict, vocabs):
    t = np.cumsum(predict)
    s = np.sum(predict)
    # sample = int(np.searchsorted(t, np.random.rand(1) * s))
    sample = np.argmax(predict)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]  # [np.argmax(predict)]
 

def gen_poetry(words, to_num):
    batch_size = 1
    print('模型保存目录为： {}'.format('./model'))
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(len(words), input_data=input_data, batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
 
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
 
        x = np.array(to_num('B')).reshape(1, 1)
 
        _, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
 
        word = input('请输入起始字符:')
        poem_ = ''
        while word != 'E':
            poem_ += word
            x = np.array(to_num(word)).reshape(1, 1)
            predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                           feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, words)
        print(poem_)
        return poem_
 
 
def generate(words, to_num, style_words="狂沙将军战燕然，大漠孤烟黄河骑。"):
 
    batch_size = 1
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    end_points = rnn_model(len(words), input_data=input_data, batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session(config=config) as sess:
        sess.run(init_op)
 
        checkpoint = tf.train.latest_checkpoint('./model')
        saver.restore(sess, checkpoint)
 
        x = np.array(to_num('B')).reshape(1, 1)
        _, last_state = sess.run([end_points['prediction'], end_points['last_state']], feed_dict={input_data: x})
 
        if style_words:
            for word in style_words:
                x = np.array(to_num(word)).reshape(1, 1)
                last_state = sess.run(end_points['last_state'],
                                      feed_dict={input_data: x, end_points['initial_state']: last_state})
 
        # start_words = list("少小离家老大回")
        start_words = list(input("请输入起始语句："))
        start_word_len = len(start_words)
 
        result = start_words.copy()
        max_len = 200
        for i in range(max_len):
 
            if i < start_word_len:
                w = start_words[i]
                x = np.array(to_num(w)).reshape(1, 1)
                predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                               feed_dict={input_data: x, end_points['initial_state']: last_state})
            else:
                predict, last_state = sess.run([end_points['prediction'], end_points['last_state']],
                                               feed_dict={input_data: x, end_points['initial_state']: last_state})
                w = to_word(predict, words)
                # w = words[np.argmax(predict)]
                x = np.array(to_num(w)).reshape(1, 1)
                if w == 'E':
                    break
                result.append(w)
 
        print(''.join(result))


def train(words,poetry_vector,x_batches,y_batches):
    input_data = tf.placeholder(tf.int32,[batch_size,None])
    output_targets = tf.placeholder(tf.int32,[batch_size,None])
    end_points = rnn_model(len(words),input_data=input_data,output_data=output_targets,batch_size=batch_size)
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    merge = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./logs',sess.graph)
        sess.run(init_op)

        start_epoch = 0
        model_dir = "./model"
        epochs = 1
        checkpoint = tf.train.latest_checkpoint(model_dir)
        if checkpoint:
            saver.restore(sess,checkpoint)
            print("## restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
            print('## start training...')
        try:
            for epoch in range(start_epoch,epochs):
                n_chunk = len(poetry_vector) // batch_size
                for n in range(n_chunk):
                    loss,_,_ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op'],
                    ],feed_dict={input_data: x_batches[n],output_targets: y_batches[n]})
                    print('Epoch: %d, batch: %d, training loss: %.6f' % (epoch,n,loss))
                    if epoch % 5 == 0:
                        saver.save(sess,os.path.join(model_dir,"poetry"),global_step=epoch)
                        result = sess.run(merge,feed_dict={input_data: x_batches[n],output_targets: y_batches[n]})
                        writer.add_summary(result,epoch * n_chunk + n)
        except KeyboardInterrupt:
            print('## Interrupt manually, try saving checkpoint for now...')
            saver.save(sess,os.path.join(model_dir,"poetry"),global_step=epoch)
            print('## Last epoch were saved, next time will start from epoch {}.'.format(epoch))


if __name__ == "__main__":
      words,poetry_vector,to_num,x_batches,y_batches = poetry_process()
      train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
    # generate(words, to_num, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")
  


# 这里我们演示了，可以保存断点的训练过程，模型可以根据上次结束训练的时间点开始继续训练，由于时间有限，我们就不展示完全的训练的过程。

# In[1]:

if __name__ == "__main__":
      words,poetry_vector,to_num,x_batches,y_batches = poetry_process()
    # train(words, poetry_vector, x_batches, y_batches)
    # gen_poetry(words, to_num)
      generate(words, to_num, style_words="狂沙将军战燕然，大漠孤烟黄河骑。")


# 由于训练语料库的原因，在完成整体训练以后，诗词生成效果将变好。
# 
# 本地案例
# 
# head:少小离家老大回 + style:山雨欲来风满楼：
# 
# 少小离家老大回，四壁百月弄鸦飞。
# 扫香花间春风地，隔天倾似烂桃香。
# 近来谁伴清明日，两株愁味在罗帏。
# 仍通西疾空何处，轧轧凉吹日方明。
# 
# head:少小离家老大回 + style:铁马冰河入梦来：
# 
# 少小离家老大回，化空千里便成丝。
# 官抛十里同牛颔，莫碍风光雪片云。
# 饮水远涛飞汉地，云连城户翠微低。
# 一树铁门万象耸，白云三尺各关高。
# 同言东甸西游子，谁道承阳要旧忧。
# 
# 少小离家老大回，含颦玉烛拂楼台。
# 初齐去府芙蓉死，细缓行云向国天。
# 
