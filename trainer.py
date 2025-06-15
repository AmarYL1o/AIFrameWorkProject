import paddle
import paddle.nn as nn
import numpy as np
from model import PMLAM
from AmazonDataset import RecDataset
from paddle.io import DataLoader


def train_and_evaluate(folds, num_users, num_items, user_pos_items, embed_dim=50, epochs=10):
    fold_metrics = []
    for fold_id, (train_pairs, test_pairs) in enumerate(folds):
        print(f"\n=== Fold {fold_id + 1} ===")

        # 初始化模型
        model = PMLAM(num_users, num_items, embed_dim)

        # print("Model parameters:")
        # for name, _ in model.named_parameters():
        #     print(name)

        # 分组参数
        margin_params = [p for name,
                         p in model.named_parameters() if "margin_net" in name]
        other_params = [p for name, p in model.named_parameters()
                        if "margin_net" not in name]

        # 初始化优化器
        inner_optim = paddle.optimizer.Adam(
            parameters=other_params, learning_rate=0.001)
        outer_optim = paddle.optimizer.Adam(
            parameters=margin_params, learning_rate=0.001)

        train_dataset = RecDataset(
            train_pairs, num_users, num_items, user_pos_items)
        test_dataset = RecDataset(
            test_pairs, num_users, num_items, user_pos_items)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                users, pos_items, neg_items = batch

                # 内层优化
                inner_optim.clear_grad()
                users = paddle.cast(users, dtype='int64')
                pos_items = paddle.cast(pos_items, dtype='int64')
                neg_items = paddle.cast(neg_items, dtype='int64')
                pos_dist, neg_dist, margin = model(users, pos_items, neg_items)
                loss = paddle.mean(paddle.nn.functional.relu(
                    pos_dist - neg_dist + margin))
                loss.backward()
                inner_optim.step()

                # 外层优化
                outer_optim.clear_grad()
                with paddle.no_grad():
                    pos_dist_fixed, neg_dist_fixed, _ = model(
                        users, pos_items, neg_items)
                    loss_outer = paddle.mean(paddle.nn.functional.relu(
                        pos_dist_fixed - neg_dist_fixed + 1))
                loss_outer.backward()
                outer_optim.step()

                total_loss += loss.item()
            print(
                f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

        # 验证
        recall = evaluate(model, test_pairs, num_users,
                          num_items, user_pos_items)
        fold_metrics.append(recall)
        print(f"Fold {fold_id + 1} Recall@10: {recall:.4f}")

    print(
        f"\nMean Recall@10: {np.mean(fold_metrics):.4f} (±{np.std(fold_metrics):.4f})")


def evaluate(model, test_pairs, num_users, num_items, user_pos_items, k=10):
    model.eval()
    recalls = []

    for user in range(num_users):
        # 获取用户所有正样本（包括训练集和测试集）
        all_pos_items = user_pos_items.get(user, [])

        # 获取该用户的测试集正样本
        test_pos_items = [i for (u, i) in test_pairs if u == user]
        if not test_pos_items:
            continue

        # 训练集中该用户的正样本（即从 all_pos_items 中去掉 test ）
        train_pos_items = list(set(all_pos_items) - set(test_pos_items))

        # 模型预测
        dist = model.predict(paddle.to_tensor([user]))  # [1, num_items]
        dist = dist.squeeze()  # [num_items]

        # 创建mask
        mask = paddle.ones([num_items], dtype='float32')
        if train_pos_items:
            train_pos_tensor = paddle.to_tensor(train_pos_items, dtype='int64')
            zero_tensor = paddle.zeros_like(train_pos_tensor, dtype='float32')
            mask = paddle.scatter(mask, train_pos_tensor, zero_tensor)

        # 给训练集正样本加大值
        dist = dist + mask * 1e9

        # 取 top-k 最小的 item（即推荐的）
        _, topk = paddle.topk(dist, k=k, largest=False)

        # 计算 recall
        hit = len(set(topk.numpy()) & set(test_pos_items))
        recalls.append(hit / len(test_pos_items))

    return np.mean(recalls) if recalls else 0.0
