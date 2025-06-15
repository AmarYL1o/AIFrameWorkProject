import paddle
import paddle.nn as nn


class PMLAM(nn.Layer):
    def __init__(self, num_users, num_items, embed_dim):
        super(PMLAM, self).__init__()
        self.embed_dim = embed_dim

        # 用户和物品的高斯分布参数
        self.user_mu = nn.Embedding(num_users, embed_dim)
        self.user_sigma = nn.Embedding(num_users, embed_dim)
        self.item_mu = nn.Embedding(num_items, embed_dim)
        self.item_sigma = nn.Embedding(num_items, embed_dim)

        # 自适应边距生成网络
        self.margin_net = nn.Sequential(
            nn.Linear(3 * embed_dim, 64),  # 输入：用户-正样本-负样本的差异
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保边距 > 0
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 高斯分布参数初始化
        nn.initializer.Normal(std=0.01)(self.user_mu.weight)
        nn.initializer.Normal(std=0.01)(self.user_sigma.weight)
        nn.initializer.Normal(std=0.01)(self.item_mu.weight)
        nn.initializer.Normal(std=0.01)(self.item_sigma.weight)

    def forward(self, users, pos_items, neg_items):
        # 获取高斯分布参数
        u_mu = self.user_mu(users)
        u_sigma = paddle.exp(self.user_sigma(users))  # 保证方差为正
        pos_mu = self.item_mu(pos_items)
        pos_sigma = paddle.exp(self.item_sigma(pos_items))
        neg_mu = self.item_mu(neg_items)
        neg_sigma = paddle.exp(self.item_sigma(neg_items))

        # 计算Wasserstein距离
        def wasserstein_dist(mu1, sigma1, mu2, sigma2):
            return paddle.sum((mu1 - mu2)**2, axis=1) + paddle.sum((paddle.sqrt(sigma1) - paddle.sqrt(sigma2))**2, axis=1)

        pos_dist = wasserstein_dist(u_mu, u_sigma, pos_mu, pos_sigma)
        neg_dist = wasserstein_dist(u_mu, u_sigma, neg_mu, neg_sigma)

        # 自适应边距生成
        s_ij = (u_mu - pos_mu)**2  # 用户-正样本差异
        s_ik = (u_mu - neg_mu)**2  # 用户-负样本差异
        s_input = paddle.concat([s_ij, s_ik, s_ij*s_ik], axis=1)  # Eq. 11
        margin = self.margin_net(s_input)

        return pos_dist, neg_dist, margin

    def predict(self, users):
        # 为指定用户生成所有物品的Wasserstein距离
        u_mu = self.user_mu(users)
        u_sigma = paddle.exp(self.user_sigma(users))
        all_items_mu = self.item_mu.weight
        all_items_sigma = paddle.exp(self.item_sigma.weight)

        # 计算用户与所有物品的距离
        dist = paddle.sum((u_mu.unsqueeze(1) - all_items_mu)**2, axis=2) + \
            paddle.sum((paddle.sqrt(u_sigma).unsqueeze(1) -
                       paddle.sqrt(all_items_sigma))**2, axis=2)
        return dist
