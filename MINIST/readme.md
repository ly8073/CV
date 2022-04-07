## 第一个CV项目----MINIST
### 现存问题点： 
1. 如何在网络的卷积层转为全连接层时自动计算全连接层的输入维度？  
* 增加_calculate_fc_dim方法，通过给定随机输入，计算出卷积层的输出

### 卷积尺寸计算公式
* `out_size = （in_size - K + 2P）/ S +1`  
K为kernel_size的设置值，P为padding的设置值，S为stride的设置值
* `out_size = （in_size - 1) * stride + outpadding - 2*padding + kernelsize`  
反卷积