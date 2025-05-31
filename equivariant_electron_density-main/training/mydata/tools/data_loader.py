import torch
import numpy as np
from sklearn.model_selection import train_test_split
from .Dataset import Dataset
import torch.nn.functional as F
import time

def binary_split(dataset, targ_name, test_size, seed):
    """
    Split the dataset with proportional amounts of a binary label in each.
    Args:
        dataset (nff.data.dataset): NFF dataset
        targ_name (str, optional): name of the binary label to use
            in splitting.
        test_size (float, optional): fraction of dataset for test
    Returns:
        idx_train (list[int]): indices of species in the training set
        idx_test (list[int]): indices of species in the test set
    """

    # get indices of positive and negative values
    pos_idx = [i for i, targ in enumerate(dataset.props[targ_name])
               if targ]
    neg_idx = [i for i in range(len(dataset)) if i not in pos_idx]

    # split the positive and negative indices separately
    pos_idx_train, pos_idx_test = train_test_split(pos_idx,
                                                   test_size=test_size,
                                                   random_state=seed)
    neg_idx_train, neg_idx_test = train_test_split(neg_idx,
                                                   test_size=test_size,
                                                   random_state=seed)

    # combine the negative and positive test idx to get the test idx
    # do the same for train

    idx_train = pos_idx_train + neg_idx_train
    idx_test = pos_idx_test + neg_idx_test

    return idx_train, idx_test

def split_train_test(dataset,
                     test_size=0.2,
                     binary=False,
                     targ_name=None,
                     seed=None):
    if binary:
        idx_train, idx_test = binary_split(dataset=dataset,
                                           targ_name=targ_name,
                                           test_size=test_size,
                                           seed=seed)
    else:
        idx = list(range(len(dataset)))
        idx_train, idx_test = train_test_split(idx, test_size=test_size,
                                               random_state=seed)

    train = Dataset(
        props={item[0]: item[1][idx_train]
               for item in dataset.items()},
        units=dataset.units
    )

    test = Dataset(
        props={key: [val[i] for i in idx_test]
               for key, val in dataset},
        units=dataset.units
    )

    return train, test


def split_train_validation_test(dataset,
                                val_size=0.2,
                                test_size=0.2,
                                seed=None,
                                **kwargs):
    train, validation = split_train_test(dataset,
                                         test_size=val_size,
                                         seed=seed,
                                         **kwargs)
    train, test = split_train_test(train,
                                   test_size=test_size / (1 - val_size),
                                   seed=seed,
                                   **kwargs)

    return train, validation, test

REINDEX_KEYS = ['atoms_nbr_list', 'nbr_list', 'bonded_nbr_list',
                'angle_list', 'mol_nbrs']
NBR_LIST_KEYS = ['bond_idx', 'kj_idx', 'ji_idx']
MOL_IDX_KEYS = ['atomwise_mol_list', 'directed_nbr_mol_list',
                'undirected_nbr_mol_list']
IGNORE_KEYS = ['rd_mols']

# TYPE_KEYS = {
#     'atoms_nbr_list': torch.long,
#     'nbr_list': torch.long,
#     'num_atoms': torch.long,
#     'bond_idx': torch.long,
#     'bonded_nbr_list': torch.long,
#     'angle_list': torch.long,
#     'ji_idx': torch.long,
#     'kj_idx': torch.long,
# }

def collate_dicts(dicts):
    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if key in IGNORE_KEYS:
            continue
        if type(val) == str:
            batch[key] = [data[key] for data in dicts]
        elif hasattr(val, 'shape') and len(val.shape) > 0:
            # if (key=='feature')|(key=='feature_bond'):
            if (key == 'feature'):
                max_shape = max([data[key].shape[1] for data in dicts])
                if dicts[0][key].is_sparse:
                    if np.all(np.array([data[key].shape[1] for data in dicts]) == max_shape):
                        batch[key] = torch.cat([data[key].to_dense() for data in dicts], dim=0)
                    else:
                        batch[key] = torch.cat([F.pad(data[key].to_dense(), pad=(0, 0, 0, max_shape - data[key].shape[1]), mode='constant',value=0) for data in dicts], dim=0)
                else:
                    if np.all(np.array([data[key].shape[1] for data in dicts])==max_shape):
                        batch[key] = torch.cat([data[key] for data in dicts], dim=0)
                    else:
                        batch[key] = torch.cat([torch.cat([data[key],torch.zeros(data[key].shape[0],max_shape-data[key].shape[1],data[key].shape[2])],1) for data in dicts], dim=0)
            elif key == 'edge_index':
                atom_num = 0
                for i, data in enumerate(dicts):
                    if atom_num == 0 :
                        batch[key] = data[key]
                        batch['ptr'] = [0]
                        batch['batch'] = torch.full((data['num_a']), i, dtype=torch.long)
                    else:
                        batch[key] = torch.cat((batch[key],data[key]+atom_num),1)
                        batch['ptr'].append(batch['ptr'][-1]+atom_num)
                        batch['batch'] = torch.cat((batch['batch'],torch.full((data['num_a']), i, dtype=torch.long)),0)
                    atom_num += data['x'].shape[0]
            elif (key == 'x') and ('dos' in dicts[0]):
                max_shape = max([data[key].shape[1] for data in dicts])
                if np.all(np.array([data[key].shape[1] for data in dicts])==max_shape):
                    batch[key] = torch.cat([
                        torch.Tensor(data[key])
                        for data in dicts
                    ], dim=0)
                else:
                    batch[key] = torch.cat([
                        torch.cat([torch.Tensor(data[key]),torch.zeros(data[key].shape[0],max_shape-data[key].shape[1])],1)
                        for data in dicts
                    ], dim=0)
            elif key == 'dos' and len(dicts[0]['dos'].shape) == 3:
                max_shape = dicts[0]['dos'].shape[2]
                if np.all(np.array([data[key].shape[2] for data in dicts])==max_shape):
                    batch[key] = torch.cat([
                        torch.Tensor(data[key])
                        for data in dicts
                    ], dim=0)
                    batch['spd'] = torch.cat([torch.ones(data[key].shape[0],data[key].shape[2]) for data in dicts],dim=0)
                else:
                    batch[key] = torch.cat([
                        torch.cat([torch.Tensor(data[key]),torch.zeros(data[key].shape[0],data[key].shape[1],max_shape-data[key].shape[2])],2)
                        for data in dicts
                    ], dim=0)
                    batch['spd'] = torch.cat([
                        torch.cat([torch.ones(data[key].shape[0],data[key].shape[2]),torch.zeros(data[key].shape[0],max_shape-data[key].shape[2])],1)
                        for data in dicts
                    ], dim=0)
                    print(batch['spd'])
            elif key == 'scaling' and len(dicts[0]['dos'].shape) == 3:
                for data in dicts:
                    max_shape = dicts[0]['dos'].shape[2]
                    if np.all(np.array([data[key].shape[1] for data in dicts]) == max_shape):
                        batch[key] = torch.cat([
                            torch.Tensor(data[key])
                            for data in dicts
                        ], dim=0)
                    else:
                        batch[key] = torch.cat([
                            torch.cat([torch.Tensor(data[key]), torch.zeros(data[key].shape[0],max_shape - data[key].shape[1])], 1)
                            for data in dicts
                        ], dim=0)
            else:
                batch[key] = torch.cat([
                    torch.Tensor(data[key])
                    for data in dicts
                ], dim=0)
        else:
            if dicts[0][key] == None:
                continue
            else:
                batch[key] = torch.stack(
                    [torch.Tensor(data[key]) for data in dicts],
                    dim=0
                )
    # # adjusting the data types:
    # for key, dtype in TYPE_KEYS.items():
    #     if key in batch:
    #         batch[key] = batch[key].to(dtype)

    return batch