# p-classification

## Introduction
&emsp;&emsp;本项目用于记录比赛，包括部分论文复现。  

## Performance
|Type of Task|Model|Precision|Recall|F1_score(marco)|  
|:--:|:---|:----:|:--:|:-----:|
|p-classification|CNN_5fold-cv|||0.8962|
|p-classification|Capsule_5fold-cv|||0.8890|
|p-classification|BiGRU_5fold-cv|||0.8963|
|p-classification|BiLSTM_5fold-cv||||
|sequence-labeling|BiLSTM+CRF||||
|Total|Ensemble||||

## Detail
- [x] 使用 **position** 特征
- [x] 使用 **postag** 特征
- [x] 使用 **hypernym** 特征
- [x] 固定窗口使用 **entity** 特征
- [x] 字向量使用 **sgns (negative sampling)**
- [x] 词向量使用 **topic word embedding** 和 **glove**
- [x] 字词向量融合使用 **add** 形式
- [x] 使用 **3-grams concat (RNN)** 输入 
- [x] 直接 **capsules flatten** 替代 **capsules 模长概率** 计算
- [x] 对 **NA** 做特殊处理, 包括使用 **grid search** 阈值判断
- [x] **不采用** softmax cross-entropy 作为损失函数 

## Structure  
&emsp;&emsp;**Embedding部分**：  

<img src="https://drive.google.com/uc?export=view&id=1-By8e5CuQTXkm3Fong7FZ0s_0lTTmhxV" width = "650" height = "400" alt="sentence_model" align=center />  


&emsp;&emsp;**CNN部分**：  
<img src="https://drive.google.com/uc?export=view&id=12z_QHLjA4zW2wScbTSxQiYHGNfUCDbz-" width = "420" height = "400" alt="sentence_model" align=center />  
