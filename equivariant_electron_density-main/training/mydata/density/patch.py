# def density_patch(density,patch_num,charge_num):
#     # feature = feature.reshape(charge_num[2], charge_num[1], charge_num[0])
#     density = density.reshape(charge_num[2], charge_num[1], charge_num[0])
#     # feature = feature.unsqueeze(1)  ##[nx,ny,nz,atom_num,num_l] -> [batch_size=nz,channel_num=1,h,w]
#     density = density.unsqueeze(1)  ##[nx,ny,nz]->[batch_size,channel_num,h,w]
#     nn_Unfold_1 = nn.Unfold(kernel_size=(patch_num, patch_num), dilation=1, padding=0,
#                             stride=(patch_num, patch_num))
#     nn_Unfold_2 = nn.Unfold(kernel_size=(patch_num, 1), dilation=1, padding=0, stride=(patch_num, 1))
#     density = nn_Unfold_1(density).transpose(0, 1).unsqueeze(1)
#     density = nn_Unfold_2(density)
#     # feature = nn_Unfold_1(feature).transpose(0, 1).unsqueeze(1)
#     # feature = nn_Unfold_2(feature)
#     density = density.reshape(-1, density.size(2)).transpose(0, 1)
#     # feature = feature.reshape(-1, feature.size(2)).transpose(0, 1)
#     density = density[:, 0]
#     # feature = feature[:, 0]
#     return density