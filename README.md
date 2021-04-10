# 工程介绍
## 文件介绍

1. ./README.md  
该文件下用到了公式和流程图，如果您的版本不能支持这两个功能，请用浏览器打开文件下的README.xml文件，这两个文件里面的内容是一致的。

2. ./run.sh  
run.sh里面会执行5个python文件，前2个对应特征提取，后3个对应模型训练以及预测，run.sh运行时，请在run.sh存在的目录（tencent_ad_2019_repetition/）下运行。可能需要较长的时间才能执行完成。

3. ./rawdata  
rawdata下面用于存放原始数据，比赛数据BTest文件夹以及total_data文件夹请放置于该文件目录下。

4. ./processedData  
所有代码运行过程中产生的数据文件都存于该目录下，代码未执行过时，该文件夹下面为空。

5. ./models  
该文件夹下面存放的是三个模型训练预测的python文件，分别对应着lgb_fm,lgb以及神经网络。

6. ./src  
src目录下面有三个文件夹，分别是dataprocess,lxyTools以及specialTools。dataprocess下面有两个数据处理的文件，这两个文件会在run.sh中被执行。lxyTools是本人整理过的本人平时常用的一些代码(这里面代码只有少部分在此次比赛中调用到)，specialTools是针对此次比赛写的一些工具性质的代码。

## 硬件依赖
运行过程需要gpu参与，本地内存消耗大概为20G(本人本地运行时使用的配置为i7 9700k, gtx 1070ti, 24G内存, 该配置下运行时间大致为4-5小时)

## 环境依赖
|name|version|
|-|-|
torch| 1.0.1.post2
lightgbm| 2.2.3  
scikit-learn| 0.20.0  
pandas| 0.24.1  
numpy| 1.16.4  
tqdm| 4.31.1  
visdom| 0.1.8.8
cuda|10.0  
这些是需要使用到的环境，如果使用过程中依然发现由于缺乏某些module而无法运行，请参考module_info.md，该文件内列出了本人本地pip list后的所有module的版本。

## 代码运行流程

1. 数据清洗  
该过程在./src/dataprocess/dataClean.py下面完成，该过程会完成训练数据的生成以及部分特征的提取。训练数据是选取的<font color=blue>每天曝光数据中出现过的每一个广告</font>（就算没有被曝光也会被选取在里面）。  
提取的特征包括:
   - 该广告对应当天在所有请求曝光队列中出现的次数  
   - 该广告当天内存在的所有请求曝光队列中对应的不同用户数目
   - 该广告当天被屏蔽的概率
   - 该广告当天平均的total_ecpm
   - 该广告当天平均的q_ecpm
   - 该广告当天平均的pctr
   - 该广告当天内所有参与竞争的对手的平均total_ecpm
   - 该广告当天内所有参与竞争的对手的平均pctr
   - 该广告当天内所有参与竞争的对手的平均q_ecpm
   - 该广告当天内所有参与竞争的对手的被屏蔽的平均概率
   - 该广告当天内存在的所有请求曝光队列中的广告的最大total_ecpm的平均值
   - 该广告当天内存在的所有请求曝光队列中每一个队列的平均参与竞争的广告数

2. 训练数据，验证数据以及线上测试数据生成  
该过程将在./src/dataprocess/traindataGenerator.py下面完成，该过程主要包括数据集划分以及特征生成。
数据集划分，这里本人将训练数据里面的最后两天作为验证集，剩下的用作训练集。
提取的特征包括:
    ***
    <center>曝光量特征</center>

    ***
    - 该广告在前天以及大前天的日曝光量（注: 这里是两个特征，下面类似说法都是指的两个特征）
    - 该广告对应的广告账户在前天以及大前天的旗下广告日曝光量的中位数
    - 该广告对应的商品在前天以及大前天的旗下广告日曝光量的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告日曝光量的中位数
    ***
    <center>对应同广告id下特征</center>

    ***
    - 该广告在前天以及大前天的平均被屏蔽率
    - 该广告在前天以及大前天的所有参与竞争的对手的平均total_ecpm
    - 该广告在前天以及大前天的存在的所有请求曝光队列中的广告的最大total_ecpm的平均值
    - 该广告在前天以及大前天的存在的所有请求曝光队列中对应的不同用户数目
    - 该广告在前天以及大前天的所有参与竞争的对手的平均pctr
    - 该广告在前天以及大前天的所有参与竞争的对手的被屏蔽的平均概率
    - 该广告在前天以及大前天的所有参与竞争的对手的平均q_ecpm
    ***
    <center>对应同账户下中位数特征</center>

    ***
    - 该广告对应的广告账户在前天以及大前天的旗下广告的平均被屏蔽率的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均total_ecpm的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的存在的所有请求曝光队列中的广告的最大total_ecpm的平均值的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的存在的所有请求曝光队列中对应的不同用户数目的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均pctr的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的被屏蔽的平均概率的中位数
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均q_ecpm的中位数
    ***
    <center>对应同广告尺寸下中位数特征</center>

    ***
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的平均被屏蔽率的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的所有参与竞争的对手的平均total_ecpm的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的存在的所有请求曝光队列中的广告的最大total_ecpm的平均值的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的存在的所有请求曝光队列中对应的不同用户数目的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的所有参与竞争的对手的平均pctr的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的所有参与竞争的对手的被屏蔽的平均概率的中位数
    - 该广告对应的广告尺寸在前天以及大前天的旗下广告的所有参与竞争的对手的平均q_ecpm的中位数
    ***
    <center>对应同账户下标准差特征</center>

    ***
    - 该广告对应的广告账户在前天以及大前天的旗下广告的平均被屏蔽率的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均total_ecpm的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的存在的所有请求曝光队列中的广告的最大total_ecpm的平均值的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的存在的所有请求曝光队列中对应的不同用户数目的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均pctr的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的被屏蔽的平均概率的标准差
    - 该广告对应的广告账户在前天以及大前天的旗下广告的所有参与竞争的对手的平均q_ecpm的标准差
    ***
3. 模型训练以及预测线上测试数据
    1. 首先对训练集以及验证集里面曝光量为0的样本进行一个降采样，只保留原来的40%的曝光量为0的样本
    2. ./models/lgb_fm_with_kfold.py  这一步说的操作都是在kfold下面完成，折数为5折，首先训练一个lgb回归模型，回归模型参数如下所示:   
        ```python
        lgb_param = {
            'boosting_type': 'gbdt',
            'objective': special_Objective,
            'num_leaves': 1024,
            'max_depth': -1,
            'learning_rate': 0.05,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 600,
            'reg_alpha': 10,
        }
        ```
        其目标函数如下所示：
        ```python
        def special_Objective(y_true, y_pred):

            y_true = pd.Series(y_true).apply(lambda x: max(x, 0.1)).values
            y_pred = np.array(y_pred)

            err = y_pred - y_true
            err_sign = np.sign(err)

            err_abs = np.abs(err)
            delta = 150

            grad = np.zeros(err.shape[0])
            hess = np.zeros(err.shape[0])

            grad[err_abs<delta] = (2*err/((y_true)))[err_abs<delta]
            hess[err_abs<delta] = (2/((y_true)))[err_abs<delta]

            grad[err_abs>=delta] = (delta * err_sign/((y_true)))[err_abs>=delta]
            return grad, hess
        ```
        训练好的模型，将会用于特征提取，即通过lgb模型预测的每一个样本的叶子节点信息作为特征提取出来，然后根据这些叶子节点信息来训练一个形式特殊的FM(因子分解机)模型。
        该模型结果如下所示:
        <center>


        ```mermaid
        graph TD
          leaf --稀疏数据--> embedding
          embedding --稠密数据--> dropout
          dropout --> FM
          FM --+bias--> out
        ```  
        </center>
        该模型中的参数如下所示：

        ```python
        param = {
          'dropout': 0.1,
          'embedding_dim': 3,
          'FM_k': 1,
          'initial_bias': 100,
          'batch_size': 2048,
          'epoch_num': 50
        }
        ```

        该模型损失函数如下所示：
        $$
        loss = \frac{|Relu(x-1) - Relu(y-1)|}{|Relu(x-1) + Relu(y-1)+2|}
        $$

        这个模型训练完成以后将会用于预测线上测试数据(每一折上训练的模型预测的结果将被求均值)，将预测的结果保留下来。
    3. ./models/static_nn.py 该步操作也是在kfold下完成，折数为5折。首先先将数据里面的取值多的类别特征例如账户id以及商品id转化成对应的频数，然后把数据分成连续特征以及离散特征给到模型训练，该模型如下所示

       <center>


       ```mermaid
       graph TD
       discrete_data --> embedding
       embedding --> torch.cat_1
       continuous_data --> torch.cat_1
       torch.cat_1 --> FC_out_dim_128_relu
       FC_out_dim_128_relu --> FC_out_dim_256_relu
       FC_out_dim_256_relu --> FC_out_dim_512_relu
       FC_out_dim_512_relu --> FC_out_dim_600_relu
       FC_out_dim_600_relu --> dropout_0.05
       dropout_0.05 --> torch.cat_2
       torch.cat_1 --> torch.cat_2
       torch.cat_2 --> FC_out_dim_1
       ```

       </center>

       上面提到的模型使用的损失函数如下所示：
       $$
       loss = \frac{|leakyRelu(x-1) - Relu(y-1)|}{|Relu(x-1) + Relu(y-1)+2|}
       $$
       该模型涉及到的batch_size为2048，epoch_num为80，学习率为0.001，几个离散特征对应的embedding_dim分别为2,2,1,2
       这个模型训练完成以后将会用于预测线上测试数据(每一折上训练的模型预测的结果将被求均值)，将预测的结果保留下来。

    4. ./models/lgb.py 该步骤也是在kold下完成，折数为5折。该部分我们会再次训练一个lgb回归模型，该模型的训练用的数据，参数，损失函数都与步骤2中的一致，训练完的模型用来预测线上测试数据，这样我们就有3个预测的结果，将这三个结果以如下形式加权得到预测结果:
    $$answer = 0.65*lgbFm + 0.35*(0.7*lgb+0.3*nn)$$
       该预测结果再根据广告的出价来调整预测曝光量，调整后的结果即为最后的输出。
