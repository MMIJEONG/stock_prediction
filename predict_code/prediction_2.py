#python:3.7.x,,,,tensorflow==1.14,,,,,,

#pip uninstall tensorflow
#pip install tensorflow==1.14
##pip intsall --upgrade pip
#pip intsall tensorflow
#pip install keras-on-lstm
#pip install pandas_datareader
#pip install yfinance #야후 주식 데이터 불러오기

from pandas_datareader import data
import datetime
import yfinance as yf
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dic = {'현대글로비스':'086280.KS', '금호석유':'011780.KS', 'CJ제일제당': '097950.KS'}
stock_name=input()

tf.reset_default_graph()

yf.pdr_override()
tf.set_random_seed(777)

start_date = '2010-01-01'
#name = '034730.KS'
name = dic[stock_name]
stock = data.get_data_yahoo(name, start_date)
stock = stock[:-1]
#print(stock)#####출력해서최근주식종가 확인

# 정규화
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차

# 역정규화 : 정규화된 값을 원래의 값으로 되돌림
def reverse_min_max_scaling(org_x, x):  # 종가 예측값
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

input_dcm_cnt = 6 #입력데이터의 컬럼 개수
output_dcm_cnt = 1 #결과데이터의 컬럼 개수

seq_length = 28 #1개 시퀸스의 길이(시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20 #각 셀의 히든 출력 크기
forget_bias = 1.0 #망각편향(기본값 1.0)
num_stacked_layers = 1 #Stacked LSTM Layers 개수
keep_prob = 1.0 #Dropout 할때 Keep할 비율

epoch_num = 1000 #에포크 횟수 (몇회 반복 학습)
learning_rate = 0.01 #학습률

stock_info = stock.values[1:].astype(np.float)

price = stock_info[:,:-1] # <- here
norm_price = min_max_scaling(price)
norm_price.shape

volume = stock_info[:,-1:] # <- here
norm_volume = min_max_scaling(volume)
norm_volume.shape

x = np.concatenate((norm_price, norm_volume), axis=1)
y = x[:, [-2]]
dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None,seq_length, input_dcm_cnt])
Y = tf.placeholder(tf.float32, [None,1])
print("X:",X)
print("Y:",Y)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
print("targets", targets)
print("predictions", predictions)

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                       forget_bias=forget_bias,
                                       state_is_tuple=True,
                                       activation=tf.nn.softsign)
    if keep_prob < 1.0 :
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell

stackedRNNs = [lstm_cell()for _ in range(num_stacked_layers)] #Stacked LSTM Layers 개수 1
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True)if num_stacked_layers > 1 else lstm_cell()

hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
hypothesis = tf.contrib.layers.fully_connected(hypothesis[:,-1], output_dcm_cnt, activation_fn=tf.identity)
hypothesis.shape

loss = tf.reduce_sum(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []
test_error_summary = []
test_predict = ''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#학습
start_time = datetime.datetime.now()
print('학습 시작...')

for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1, train_error, test_error,
                                                                                 test_error - train_error))

end_time = datetime.datetime.now() #종료시간을 기록
elapsed_time = end_time - start_time # 경과시간을 구한다
#시간측정
'''print('elapsed_time:',elapsed_time)
print('elapsed_time per epoch:',elapsed_time/epoch_num)

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))'''

# 결과 그래프 출력
plt.figure(1)
plt.plot(train_error_summary, 'gold')
plt.plot(test_error_summary, 'b')
plt.xlabel('Epoch(x100)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.show()

recent_data = np.array([x[len(x)-seq_length : ]])
#print("recent_data.shape:", recent_data.shape)
#print("recent_data:", recent_data)
#print(price)####여기도 확인
print("The previous day's stock price",price[-1][-1])###그전날 종가 출력
#recent_data_for_print = reverse_min_max_scaling(price,recent_data)
#print("preday's stock price", recent_data_for_print[0][0][0])

test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

#print("test_predict", test_predict[0])


test_predict = reverse_min_max_scaling(price, test_predict)
print("Tomorrow's stock price", test_predict[0])
stock_percent=((test_predict[0]-price[-1][-1])/price[-1][-1])*100
if stock_percent>0:
    print("%s의 주식 가격은 %f%% 증가 될 것으로 예상됩니다." % (stock_name,stock_percent))
elif stock_percent==0:
    print("%s의 주식 가격은 전날과 동일 할 것으로 예상됩니다." % stock_name)
else:
    print("%s의 주식 가격은 %f%% 감소 될 것으로 예상됩니다." % (stock_name, abs(stock_percent)))



