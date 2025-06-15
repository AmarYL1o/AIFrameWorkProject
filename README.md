# Reproduction: Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation

## 介绍

本项目旨在复现论文[Probabilistic Metric Learning with Adaptive Margin for Top-K Recommendation](https://arxiv.org/pdf/2101.04849)的相关代码，使用PaddlePaddle进行代码的构建，并对来自[Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)的Books、CDs两个数据集进行测试。

在信息爆炸时代，用户面临海量在线内容和服务的选择困难，个性化推荐系统成为缓解信息过载的关键工具。其核心任务是通过用户历史行为（如点击、购买）预测用户偏好，并为每个用户生成个性化的Top-K推荐列表（即推荐用户最可能感兴趣的K个物品）。这一任务要求模型不仅需准确捕捉用户-物品交互关系，还需保证推荐结果的多样性和可解释性。

论文提出了一种概率度量学习模型PMLAM，其核心贡献包括：
- 通过高斯分布表示用户和物品，捕捉嵌入的不确定性；
- 将边距生成建模为双层优化问题，动态生成边距；
- 显示建模用户-用户和物品-物品关系，提升推荐性能。

## 数据集

原论文使用[Amazon Review Dataset](https://nijianmo.github.io/amazon/index.html)数据集。Amazon Review Dataset数据集收录了亚马逊平台上用户对商品的评价信息，涵盖图书、电子产品、影视、音乐等33个商品类别常用于：
1. 推荐系统
    - 基于用户历史评论预测下一个可能的商品购买（序列推荐）；
    - 多个用户共同购买图谱的构建以及优化关联推荐。
2. 自然语言处理
    - 根据评论文本预测评分；
    - 分析评论真实性。
3. 用户行为研究
    - 结合用户的评价时间、物品价格等因素，研究消费市场周期性与商品热度关系；
    - 从差评中分析提取产品改进设计。
    
在Amazon Review Dataset中，我们依照原论文选择Books、CDs两个数据集，且选用`Rating-only`数据集。数据集的形式如下：

| Item | User | Rating  | Timestamp |
|:------------:|:-----------:|:------------:|:------------:|
| 1713353 |A1C6M8LCIX4M6M | 5.0  |1123804800|
| 0060786817  | A5EXND10WD4PM  | 3.0  | 1137542400|

其中`item`为物品编号，`user`为用户标识，`rating`为用户对物品给出的评分，`timestamp`为用户进行评价的时间标记。

## 模型设计

