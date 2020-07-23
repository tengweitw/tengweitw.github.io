---
title: 【图解例说机器学习】感知机 (Perceptron)
mathjax: true
date: 2018-07-05 12:01:30
tags:
---



感知机是二分类的线性分类模型，是神经网络和支持向量机的基础。

--------

<!--more-->

## 引例

一个常见的线性二分类问题如下：

<img src="https://cdn.jsdelivr.net/gh/tengweitw/FigureBed@latest/20200505/Perceptron_fig001.jpg" width="400" height="300" title="图11" alt="图1test" >

如图1，假设有一个线性可分的训练集，其中有三个样例 ($\mathrm x_1,\mathrm x_2, \mathrm x_3$)，分别标记为正例(红色方块)，反例(蓝色圆圈)。这里的 $x^{(1)},x^{(2)}$为训练样例的$2$个特征。我们的目的就是找到一个超平面 (在二维空间为一条直线) 能够将这三个样例分开。显然，这样的直线有无数条，比如图中的直线 $f(\mathrm x)=x^{(1)}+x^{(2)}-3=0, f(\mathrm x)=2x^{(1)}+x^{(2)}-5=0$ 就是其中的两条。我们发现$f(\mathrm x_1)>0,f(\mathrm x_2>0),f(\mathrm x_3)<0$，于是乎，我们可以用函数表达式$f(\mathrm x)$输出值的正负来判断新的样例$\mathrm x$属于哪一类。

------

## 定义

由上面的例子，我们可以得到下面关于感知机的数学表达式：
$$
f(\mathrm x)=\mathbb I(\omega_0+\mathrm w^{\mathrm T}\mathrm x)\tag{1}
$$
其中，$\mathbb I$是指示函数，定义为：
$$
\mathbb I(x)=\begin{cases}
-1\quad x<0\\
+1\quad x>0
\end{cases}\tag{2}
$$
由公式(1),(2)可知，上面例子中，$f(\mathrm x_1)=f(\mathrm x_2=1>0),f(\mathrm x_3)=-1<0$。 注意：上述指示函数的取值为$-1,+1$是用来区分正例，反例。使用其它的值也是可以的。

----------------------

## 误差函数

分类好坏的依据是能够将训练样例正确地分类，一个自然而然的误差函数就是分类错误数。但是，这样的误差函数不是连续的，不利于我们求解最优的。为此，我们考虑误分类点到分类超平面的距离，作为我们的误差函数。在中学时，我们学过一个点$x$到一条直线$ax+by+c=0$的距离可以为$\lvert ax+by+c\rvert/\sqrt{a^2+b^2}$。类似地，空间中一点$\mathrm x$ 到一个超平面$\omega_0+\mathrm w^{\mathrm T}\mathrm x=0$的距离为：
$$
d=\frac{\lvert\omega_0+\mathrm w^{\mathrm T}\mathrm x\rvert}{\lvert\mathrm w\rvert}\tag{3}
$$

那么，对于误分类的样例$\mathrm x_i$，其预测的输出为$\hat y_i=f(\mathrm x_i)$。假定预测输出为负例，即$\hat y_i=-1$，由于被错误分类，其实际的输出为$y_i=+1$，其输入为$\omega_0+\mathrm w^{\mathrm T}\mathrm x_i>0$。为此，我们根据公式(3)，可以计算当$\mathrm x_i$被误分类时，该点到分类超平面的距离为：
$$
d(\mathrm x_i)=-\frac{\omega_0+\mathrm w^{\mathrm T}\mathrm x_i}{\lvert\mathrm w\rvert} y_i\tag{4}
$$
注意：公式(4)是公式(3)在误分类情况下的具体表达式。在误分类的情况下，输入$\omega_0+\mathrm w^{\mathrm T}\mathrm x_i$ 与实际输出$y_i$ 异号。那么对于所有误分类的样例集合$\mathcal D_{error}$，误差函数可以表示为：
$$
E=\sum\limits_{\mathrm x_i\in\mathcal D_{error}}{d(\mathrm x_i)}=-\sum\limits_{i=1}^{\lvert\mathcal D_{error}\rvert}\frac{\omega_0+\mathrm w^{\mathrm T}\mathrm x_i}{\lvert\mathrm w\rvert}y_i\tag{5}
$$
一般来说，为了使用梯度下降求解最小化损失函数$E$方便，我们可以添加约束$\lvert\mathrm w\rvert=1$ 将公式(5)转化为：
$$
E=\sum\limits_{\mathrm x_i\in\mathcal D_{error}}{d(\mathrm x_i)}=-\sum\limits_{i=1}^{\lvert\mathcal D_{error}\rvert}(\omega_0+\mathrm w^{\mathrm T}\mathrm x_i)y_i\tag{6}
$$
然而，我们发现，约束$\lvert\mathrm w\rvert=1$ 对最终优化结果 (得到最优的分类超平面) 没有影响。以上面给的例子来说，比如对于一个给定的训练集$\mathrm x_1,\mathrm x_2,\mathrm x_3$，其最优的分类超平面为$f(\mathrm x)=x^{(1)}+x^{(2)}-3=0, \lvert\mathrm w\rvert=\sqrt{2}$ 。当添加约束$\lvert\mathrm w\rvert=1$ 后，此时可以求得最优的超平面一样，只是需要将参数归一化：$f(\mathrm x)/\sqrt{2}=x^{(1)}/\sqrt{2}+x^{(2)}/\sqrt{2}-3/\sqrt{2}=0, \lvert\mathrm w\rvert=\sqrt{2}$ 。为此，我们可以不用考虑约束$\lvert\mathrm w\rvert=1$。

--------------

## 梯度下降法求解问题

#### 算法理论

为了找到最优的分类超平面$\hat y=f(\mathrm x)$， 我们需要最小化误差函数$E$来求得最佳的参数$\mathrm{\bar w}=\{\omega_0,\mathrm w\}$。这里我们采用梯度下降法。分别求$E$关于$\omega_0,\omega_j$的偏导数：
$$
\frac{\partial E}{\partial\omega_0}=-\sum\limits_{i=1}^{\lvert\mathcal D_{error}\rvert}{y}_i\tag{7}\\
$$

$$
\frac{\partial E}{\partial\omega_j}=-\sum\limits_{i=1}^{\lvert\mathcal D_{error}\rvert}x_i^{(j)}{y}_i\tag{8}
$$

由于偏导数只与被错误分类的样例有关，我们可以采用随机梯度下降法，即每次只用一个被错误分类的训练样例 (e.g., $\mathrm x_i$ ) 来更新参数：
$$
\omega_0^{t+1}=\omega_0^t+\eta^ty_i\tag{9}
$$

$$
\omega_j^{t+1}=\omega_j^{t}+\eta^tx_i^{(j)}y_i\tag{10}
$$

公式(9)和(10)的直观解释：对于每一个被误分类的样例点，我们调整$\omega_0,\mathrm w$的值，使分类超平面朝着该误分类的样例点移动，从而减少该分类点到分界面的距离，即误差。

---------------

#### 算法实现

我们还是以上面的例子来具体说明算法的具体步骤。这例子只考虑了2个特征$x^{(1)},x^{(2)}$，于是乎我们要求的分类超平面为一条直线$f(\mathrm x)=\omega_0+\omega_1x^{(1)}+\omega_2x^{(2)}$=0. 那么上述的随机梯度算法步骤如下：

1. 初始化参数: $\omega_0^0=\omega_1^0=\omega_2^0=0,\eta^t=\eta=1$;

2. 迭代过程：

   - 根据当前得到的分类直线，从训练集中找到一个会被误分类(即$f(\mathrm x_i)y_i\le 0$)的样例；

   - 比如，在第一次迭代时，$\mathcal x_1$被误分类。我们可以把样例$\mathrm x_1$代入公式(9)和(10)中，我们有:
     $$
     \omega_0^{1}=\omega_0^1+\eta y_1=0+1*1=1\tag{11}
     $$

     $$
     \omega_1^{1}=\omega_1^{0}+\eta x_1^{(1)}y_1=0+1*3*1=3\tag{12}
     $$

     $$
     \omega_2^{1}=\omega_2^{0}+\eta x_1^{(2)}y_1=0+1*3*1=3\tag{13}
     $$

     此时得到的分类直线为$1+3x^{(1)}+3x^{(2)}=0$。

3. 重复步骤2，直到训练集中找不到被误分类的训练样例。

最后得到该训练集的分类超平面为:
$$
-3+x^{(1)}+x^{(2)}=0\tag{14}
$$
此时所求的表达式为:
$$
f(\mathrm x)=\mathbb I(-3+x^{(1)}+x^{(2)})\tag{15}
$$
注意：这里的分类超平面有很多，与初始化的值和在步骤2中选取被误分类样例的不同有关。例如，$-5+x^{(1)}+2x^{(2)}=0$也是一个分类超平面。

------------------------

## 具体算法实现

这里我们以上面的例子以及iris数据集(sklearn库中自带)进行感知机二分类算法实现。

- 简单的例子

  在上述的小例子中，输入为$\{\mathrm x_1=(3,3),\mathrm x_2=(3,4),\mathrm x_3=(1,1)\}$ ，其类别为$\{y_1=+1,y_2=+1,y_3=-1\}$。我们假定一个新的测试样例为$\mathrm x=(4,4)$，其实际类别为$y=+1$。采用上面提及的随机梯度算法，我们可以得到如图2的实验结果 (具体实现的python源代码见附录)：

  <table>
      <tr>
          <td ><center><img src="https://cdn.jsdelivr.net/gh/tengweitw/FigureBed@latest/20200505/Perceptron_fig002.jpg"  ></center>  <center>图2 </center></td>
      </tr>
  </table>

- Iris数据集
  在iris数据集中，有150个训练样例，4个feature, 总共分3类。我们只考虑了前2个feature，这么做是为了在二维图3和图4中展示分类结果。并且将类别2和类别3划分为同一类别，这样我们考虑的是一个二分类问题。

<table>
    <tr>
        <td ><center><img src="https://cdn.jsdelivr.net/gh/tengweitw/FigureBed@latest/20200505/Perceptron_fig003.jpg"  >图3</center></td>
        <td ><center><img src="https://cdn.jsdelivr.net/gh/tengweitw/FigureBed@latest/20200505/Perceptron_fig004.jpg"  >图4</center></td>
    </tr>
</table>
从图3和图4中可以看出，我们找到了一个分类直线$99-62.6x^{(1)}+79.5x^{(2)}=0$，可以正确对iris数据集正确分类。

----------------

## 附录


{% spoiler "图1的python源代码：" %}

```py
# -*- coding: utf-8 -*-
# @Time : 2020/5/3 14:48
# @Author : tengweitw

import numpy as np
import matplotlib.pyplot as plt
# Set the format of labels
def LabelFormat(plt):
    ax = plt.gca()
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16,
            }
    return font


x = [3, 4, 1]
y = [3, 3, 1]
c = [r'$\mathrm{x}_1$',r'$\mathrm{x}_2$',r'$\mathrm{x}_3$']

x1=np.linspace(0,2.5,10)
y1=5-2*x1

x2=np.linspace(0,3,10)
y2=3-x2

plt.figure()

plt.plot(x[:2],y[:2],'rs')
plt.plot(x[-1],y[-1],'bo')

plt.plot(x1,y1,'k-')
plt.plot(x2,y2,'k-')

for i in range(0, len(x)):
    plt.annotate(c[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05, y[i] + 0.05),fontsize=16)
plt.annotate('$2x^{(1)}+x^{(2)}-5=0$', xy=(1, 3), xycoords='data',
             xytext=(0, 60), textcoords='offset points', color='g', fontsize=16, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc,rad=0", color='k'))
plt.annotate('$x^{(1)}+x^{(2)}-3=0$', xy=(2.5, 0.5), xycoords='data',
             xytext=(30, 30), textcoords='offset points', color='g', fontsize=16, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc,rad=0", color='k'))

# Set the labels
font = LabelFormat(plt)
plt.xlabel('$x^{(1)}$', font)
plt.ylabel('$y^{(2)}$', font)

plt.xlim(0,6)
plt.ylim(0,6)
plt.show()


```
{% endspoiler %}


{% spoiler "图2的python源代码:" %}
```python
# -*- coding: utf-8 -*-
# @Time : 2020/5/5 11:40
# @Author : tengweitw


import numpy as np


def Perceptron_gradient_descend(train_data, train_target, test_data):
    # learning rate
    eta = 1
    M = np.size(train_data, 1) # number of features
    N = np.size(train_data, 0) # number of instances
    w_bar = np.zeros((M + 1, 1)) #initialization

    # the 1st column is 1 i.e., x_0=1
    temp = np.ones([N, 1])
    # X is a N*(1+M)-dim matrix
    X = np.concatenate((temp, train_data), axis=1)
    train_target = np.array(train_target).reshape(N,1)

    iter = 0
    num_iter = 10
    while iter < num_iter:
        print('The %s-th iteration:'%(iter+1))
        # Compute f(x_i)y_i and find a wrongly-classified instance
        z = np.matmul(X, w_bar)
        fxy=z*train_target
        index_instance=np.argwhere(fxy<=0)

        if index_instance.size:
            # Get the first instance, you can also pick the instance randomly
            index_x_selected=index_instance[0][0]
        else:
            print('There is no instance that is classified by mistake.\n')
            break
        x_selected=X[index_x_selected]
        # update w according to eqs. (9) and (10)
        w_bar=w_bar+eta*x_selected.reshape(M+1,1)*train_target[index_x_selected]
        print(np.transpose(w_bar))

        iter += 1

    # Predicting, let x0=1 to be multiplied by \omega_0
    x0 = np.ones((np.size(test_data, 0), 1))
    test_data1 = np.concatenate((x0, test_data), axis=1)
    y_predict_test_temp = np.matmul(test_data1, w_bar)
    if y_predict_test_temp>0: #Note that here is only one test data, otherwise changes needed
        y_predict_test=1
    else:
        y_predict_test=-1

    return y_predict_test,w_bar


# x1 x2 x3
data = [[3,3],[4,3],[1,1]]
# The labels
label = [1,1,-1]

# testing points [4,4]
test_data = np.array([4,4]).reshape(1,2)
test_target = [1]

train_data = data
train_target = label

y_predict_test,w_bar=Perceptron_gradient_descend(train_data, train_target, test_data)
print('The point x={} whose true class is {}, is grouped as class {}.'.format(test_data,test_target,y_predict_test))


```
{% endspoiler %}

{% spoiler "图3-图4的python源代码：" %}


```python
# -*- coding: utf-8 -*-
# @Time : 2020/5/5 11:42
# @Author : tengweitw


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

# Create color maps for three types of labels
cmap_light = ListedColormap(['tomato', 'limegreen', 'cornflowerblue'])


# Set the format of labels
def LabelFormat(plt):
    ax = plt.gca()
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16,
            }
    return font


# Plot the training points:
def PlotTrainPoint(train_data, train_target):
    for i in range(0, len(train_target)):
        if train_target[i] == 1:
            plt.plot(train_data[i][0], train_data[i][1], 'rs', markersize=6, markerfacecolor="r")
        else:
            plt.plot(train_data[i][0], train_data[i][1], 'bs', markersize=6, markerfacecolor="b")

# Plot the testing points:
def PlotTestPoint(test_data, y_predict_test):
    for i in range(0, len(y_predict_test)):
        if y_predict_test[i] == 1:
            plt.plot(test_data[i][0], test_data[i][1], 'rs', markerfacecolor='none', markersize=6)
        else:
            plt.plot(test_data[i][0], test_data[i][1], 'bs', markersize=6, markerfacecolor="none")

# Plot the super plane
def Plot_segment_plane(w_bar):
    x0 = 1
    x1 = np.linspace(4, 8, 100)
    x2 = -(w_bar[0] * x0 + w_bar[1] * x1) / w_bar[2]
    plt.plot(x1, x2, 'k-')


def Perceptron_stochastic_gradient_descend(train_data, train_target, test_data):
    # learning rate
    eta = 1
    M = np.size(train_data, 1)  # number of features
    N = np.size(train_data, 0)  # number of instances
    w_bar = np.zeros((M + 1, 1))  # initialization

    # the 1st column is 1 i.e., x_0=1
    temp = np.ones([N, 1])
    # X is a N*(1+M)-dim matrix
    X = np.concatenate((temp, train_data), axis=1)
    train_target = np.array(train_target).reshape(N, 1)

    iter = 0
    num_iter = 10000

    while iter < num_iter:
        # print('The %s-th iteration:'%(iter+1))
        # Compute f(x_i)y_i and find a wrongly-classified instance
        z = np.matmul(X, w_bar)
        fxy = z * train_target
        index_instance = np.argwhere(fxy <= 0)

        if index_instance.size:
            # Get the first instance, you can also pick the instance randomly
            index_x_selected = index_instance[0][0]
        else:
            print('There is no instance that is classified by mistake.\n')
            print('The derived parameters w=', np.transpose(w_bar))
            break
        x_selected = X[index_x_selected]
        # update w
        w_bar = w_bar + eta * x_selected.reshape(M + 1, 1) * train_target[index_x_selected]

        iter += 1

    # Predicting
    x0 = np.ones((np.size(test_data, 0), 1))
    test_data1 = np.concatenate((x0, test_data), axis=1)
    y_predict_test = np.matmul(test_data1, w_bar)
    for i in range(len(y_predict_test)):
        if y_predict_test[i] > 0:
            y_predict_test[i] = 1
        else:
            y_predict_test[i] = -1

    return y_predict_test, w_bar


# import dataset of iris
iris = datasets.load_iris()

# The first two-dim feature for simplicity
data = iris.data[:, :2]
# Group 1 (labeled 0 initially) is labeled as +1
label = iris.target + 1

# Group 2 and 3 as one group, and label them as -1
label[50:] = -1

# Choose the 25,75,125th instance as testing points
test_data = [data[25, :], data[75, :], data[125, :]]
test_target = label[[25, 75, 125]]

data = np.delete(data, [25, 75, 125], axis=0)
label = np.delete(label, [25, 75, 125], axis=0)

train_data = data
train_target = label

y_predict_test, w_bar = Perceptron_stochastic_gradient_descend(train_data, train_target, test_data)

print('The point x={} \n whose true class is {}, is grouped as class {}.'.format(test_data, test_target, np.transpose(y_predict_test)))

plt.figure()
PlotTrainPoint(train_data, train_target)
PlotTestPoint(test_data, y_predict_test)
Plot_segment_plane(w_bar)
plt.annotate('$99-62.6x^{(1)}+79.5x^{(2)}=0$', xy=(7.5, 4.7), xycoords='data',
             xytext=(-300, 20), textcoords='offset points', color='g', fontsize=16, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc,rad=0", color='k'))
font = LabelFormat(plt)
plt.xlabel('$x^{(1)}$', font)
plt.ylabel('$x^{(2)}$', font)
plt.show()

```
{% endspoiler %}
































































