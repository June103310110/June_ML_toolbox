from sklearn.datasets import load_iris
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
from pyngrok import ngrok

def run_webApi():

  os.system('killall ngrok')
  print('Clear existed ngrok.')
  print('Run ngrok.')
  
  public_url = ngrok.connect(port='80')
  public_url = str(public_url).split(sep='"')[1]
  print('頁面連結', public_url)
  os.system('streamlit run --server.port 80 app.py >/dev/null')


def EDA_iris():
  iris = load_iris()

  #　鳶尾花的種類
  print('>>>鳶尾花的種類')
  print(iris.target_names)
  # 花朵的特徵
  print('>>>花朵的特徵')
  print(iris.feature_names)

  print('\n>>>IRIS Dataframe')
  data = pd.DataFrame(iris.data, columns = iris.feature_names)
  print(data.head(3))  
  return iris, data

def simple_vis(data_):
  tmp = data_.copy().loc[:, 'sepal length (cm)']


  fig = plt.figure(figsize=(16,8))
  plt.subplot(1,2,1)
  plt.scatter(x = range(len(tmp)), y = tmp)
  plt.title('All of the data point'), plt.xlabel('Index of data point'), plt.ylabel('value')

  plt.subplot(1,2,2)

  ax = [0, len(tmp)]
  ay = [tmp.mean()]*2

  plt.scatter(x = range(len(tmp)), y = tmp)
  plt.plot(ax, ay, linewidth=4, color='red')
  plt.title('Mean of all datapoint'), plt.xlabel('Index of data point'), plt.ylabel('value')

  plt.savefig("simple_datapoint.png")
  # # plt.scatter(x = range(len(tmp)), y = [tmp.mean()]*len(tmp), )
  plt.show()

def aaa():
  print('aaa')

class plt_stat():
  def __init__(self, data, figsize):
    self.name = 'plt_stat'
    self.data = data.copy()
    self.figsize = figsize
    # self.toolbox = {'mean':self.mean(), 'minmax':self.minmax, 'quartile':self.quartile(), 'std':self.std(c = 1)}
    os.makedirs('./stat_fig', exist_ok=True)

  def simple_scatter(self, no_title=None):
    tmp = self.data.copy()
    plt.scatter(x = range(len(tmp)), y = tmp)
    if not no_title:
      plt.title('All of the data point'), plt.xlabel('Index of data point'), plt.ylabel('value')

  def simple_mean(self, no_title=None):
    tmp = self.data.copy()
    ax = [0, len(tmp)]
    ay = [tmp.mean()]*2
    plt.plot(ax, ay, linewidth=1, color='red')
    if not no_title:
      plt.title('Mean of all datapoint'), plt.xlabel('Index of data point'), plt.ylabel('value')

  def mean(self):
    fig = plt.figure(figsize=self.figsize)
    plt.subplot(1, 2, 1)
    self.simple_scatter()

    plt.subplot(1, 2, 2)
    self.simple_scatter(no_title = True)

    self.simple_mean()
    
    path = './stat_fig/datapoint_mean.png'
    print('save image in '+path)
    plt.savefig(path)

    plt.show()
    return 

  def minmax(self):
    fig = plt.figure(figsize=self.figsize)
    plt.subplot(1, 2, 1)
    self.simple_scatter()

    plt.subplot(1, 2, 2)
    self.simple_scatter(no_title = True)

    tmp = self.data.copy()
   
    ax = [0, len(tmp)]
    lis = [tmp.max(), tmp.min()]
    for _ in range(2):
      ay = [lis[_]]*2
      plt.plot(ax, ay, linewidth=1, color='red')

    plt.title('MinMax of all datapoint'), plt.xlabel('Index of data point'), plt.ylabel('value')
    path = './stat_fig/datapoint_minmax.png'
    print('save image in '+path)
    plt.savefig(path)

    plt.show()
    return 

  def quartile(self):

    fig = plt.figure(figsize=self.figsize)
    plt.subplot(1, 2, 1)
    self.simple_scatter()

    plt.subplot(1, 2, 2)
    self.simple_scatter(no_title = True)
    
    tmp = self.data.copy()
    tmp.sort_values()

    lis = [0.25, 0.75]
    ay = [tmp.min()-0.2, tmp.max()+0.2]
    for _ in range(2):
      cond = np.ceil(_*len(tmp))
      cond = [cond]*2
    
      plt.plot(cond, ay, linewidth=1, color='red') # 折線圖

    plt.title('Quartile of all datapoint'), plt.xlabel('Index of data point'), plt.ylabel('value')
    path = './stat_fig/datapoint_quartile.png'
    print('save image in '+path)
    plt.savefig(path)

    plt.show()
    return 

  def std(self, c=1):
    fig = plt.figure(figsize=self.figsize)
    plt.subplot(1, 2, 1)
    self.simple_scatter()

    plt.subplot(1, 2, 2)
    
    self.simple_scatter(no_title = True)
    self.simple_mean(no_title = True)

    tmp = self.data.copy()
    ax = [0, len(tmp)]
    lis = [1*c, -1*c]
    for _ in lis:
      ay = [tmp.std()*_ + tmp.mean()]*2
      plt.plot(ax, ay, linewidth=2, color='green')

    plt.title('Mean of all datapoint'), plt.xlabel('Index of data point'), plt.ylabel('value')
    path = './stat_fig/datapoint_std.png'
    print('save image in '+path)
    plt.savefig(path)
    plt.show()
    return 