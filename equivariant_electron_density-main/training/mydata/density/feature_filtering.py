import torch

def feature_filtering(feature,abc):
    # 把点位转换成与原子位置相关的 径向基函数 * 球谐函数
    diff = feature[:, :2] - torch.Tensor([abc[0][0] / 2, abc[1][1] / 2]).unsqueeze(0).repeat(feature.shape[0], 1)
    squared_diff = diff ** 2
    sum_of_squares = torch.sqrt(torch.sum(squared_diff, dim=1))
    mask = (sum_of_squares > abc[0][0]*0.35) & (sum_of_squares < abc[0][0]*0.46) & (feature[:, 2] < (abc[2][2]*0.5))
    # mask = feature[:, 2] < (abc[2][2]*0.5)
    features_indices = torch.where(mask)
    return features_indices

