# 归积
归积 一款新型Transformer架构, 解决了传统Transformer存在的一些问题:
+ 平方复杂度
+ 复杂推理能力弱

本架构经过了大量测试，见https://zhuanlan.zhihu.com/p/23202768443

更多介绍见https://zhuanlan.zhihu.com/p/1996599793196225828

#### 运行环境
+ 支持NVIDIA sm_75/sm_80/sm_86/sm_89/sm_90系列显卡
+ Ubuntu20.04/Ubuntu22.04
+ Python3, NumPy, PyTorch

推荐使用docker运行

#### 第一个例子
```
git clone https://github.com/myhub/tf
cd ./tf
python main.py
```


#### 软件说明
本软件在<a href=https://github.com/myhub/uc>myhub/uc</a>项目基础上开发，受<a href=https://github.com/OpenBMB/MiniCPM>MiniCPM</a>, <a href=https://github.com/BlinkDL/RWKV-LM>RWKV</a>等项目启发，全程使用<a href=https://www.deepseek.com/>DeepSeek</a>大语言模型编写代码，主要依赖以下开源库：
+ <a href=https://github.com/NVIDIA/cutlass>cutluss</a>
+ <a href=https://github.com/NVIDIA/cccl>cccl</a>



