
**输入**：品牌，场景，应用
**输出**：厂商：移动 电信 联通

==1.决策树==
==2.神经网络（卷积？）==
==3.XGBoost==

### 数据预处理sklearn
1.01二值化 行业、场景、影片精度、使用位置
2.哑编码 除去二值化的枚举类型
3.是否有缺失值

### 特征选择
首先考虑两个点：一是所选特征是否发散，如果方差接近0那就没有参考价值，
二是特征与目标的相关性。


### 降维（视数据量特征维度而定）
PCA
LDA

## 现有问题
特征中存在连续特征和离散特征


特征工程？（或者就把所有数据都利用，可能会造成特征冗余，特征维度过高）
将任意数据格式转换为机器可学习的数字特征（one-hot编码(独热编码)）

分析厂商受哪些信息的影响，最直观的就是品牌，观察品牌是什么厂商旗下的。
选出用户常填写的一些信息来作为输入层

***
1.对原始特征分别进行归一化 避免连续数值范围不同
2.离散特征则用热读编码
