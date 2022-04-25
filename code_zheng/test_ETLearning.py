import numpy as np
from ETLearning import ETL
from scipy.io import savemat
np.random.seed(0)
import matplotlib.pyplot as plt


def test_beijing5(dim_embedding=3, save=False):
    data_file = '../../data/Beijing/beijing_15k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(1):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        # max_t = np.max([np.max(t), np.max(t_te)]) * 0.5
        # t = t / max_t
        # t_te = t_te / max_t

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 10, 10, dim_embedding])
        nrmse, nmae = model.train(ind, t, y, ind_te, t_te, y_te, test_every=50, total_epoch=10, lr=1e-2, batch_size=1000)
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        plt.figure()
        plt.plot(np.arange(len(nrmse)), nrmse, label='nrmse')
        plt.plot(np.arange(len(nmae)), nmae, label='nmae')
        plt.savefig('Beijing_{}.png'.format(fold))
        plt.close()
    
    with open('log.txt', 'a') as f:
        f.write('beijing_5fold_d_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(dim_embedding, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))


def test_ctr5(dim_embedding=3, save=False):
    data_file = '../data/clickthrough/ctr_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        max_t = np.max([np.max(t), np.max(t_te)])
        t = t / max_t
        t_te = t_te / max_t

        # max_t = np.max([np.max(t), np.max(t_te)]) * 0.5
        # t = t / max_t
        # t_te = t_te / max_t

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding])
        nrmse, nmae = model.train(ind, t, y, ind_te, t_te, y_te, test_every=50, total_epoch=2000, lr=1e-2, batch_size=1000)
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        plt.figure()
        plt.plot(np.arange(len(nrmse)), nrmse, label='nrmse')
        plt.plot(np.arange(len(nmae)), nmae, label='nmae')
        plt.savefig('ctr_{}.png'.format(fold))
        plt.close()
    
    with open('log.txt', 'a') as f:
        f.write('ctr_5fold_d_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(dim_embedding, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_server5(dim_embedding=3, save=False):
    data_file = '../data/server/server_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        max_t = np.max([np.max(t), np.max(t_te)])
        t = t / max_t
        t_te = t_te / max_t

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding])
        nrmse, nmae = model.train(ind, t, y, ind_te, t_te, y_te, test_every=50, total_epoch=2000, lr=1e-3, batch_size=1000)
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        plt.figure()
        plt.plot(np.arange(len(nrmse)), nrmse, label='nrmse')
        plt.plot(np.arange(len(nmae)), nmae, label='nmae')
        plt.savefig('ctr_{}.png'.format(fold))
        plt.close()
    
    with open('log.txt', 'a') as f:
        f.write('server_5fold_d_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(dim_embedding, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

def test_mv5(dim_embedding=3, save=False):
    data_file = '../data/movielens/mv_10k.npy'
    data = np.load(data_file, allow_pickle=True).item()
    nvec = data['ndims']
    data = data['data']
    res = []
    rmse_list = []
    ll_list = []
    mae_list = []
    for fold in range(5):
        ind = data[fold]['tr_ind']
        t = data[fold]['tr_T']
        y = data[fold]['tr_y'].astype(np.float32)
        ind_te = data[fold]['te_ind']
        t_te = data[fold]['te_T']
        y_te = data[fold]['te_y'].astype(np.float32)

        # max_t = np.max([np.max(t), np.max(t_te)]) * 0.5
        # t = t / max_t
        # t_te = t_te / max_t

        # m = np.mean(tr_y)
        # std = np.std(tr_y)
        # tr_y = (tr_y - m) / std
        # te_y = (te_y - m) / std
        # (tr_idx, tr_T, tr_y), (te_idx, te_T, te_y), U = gen_data([10, 10, 10], [cos_func, cos_func, cos_func], 300, 300)

        model = ETL(nvec, dim_embedding, 100, [dim_embedding+1, 50, 50, dim_embedding])
        nrmse, nmae = model.train(ind, t, y, ind_te, t_te, y_te, test_every=50, total_epoch=2000, lr=1e-2, batch_size=1000)
        rmse_list.append(np.min(nrmse))
        mae_list.append(np.min(nmae))
        # res.append([tr_rmse_list, tr_ll_list, tr_pred_m_list, tr_pred_std_list, te_rmse_list, te_ll_list, te_pred_m_list, te_pred_std_list])
        plt.figure()
        plt.plot(np.arange(len(nrmse)), nrmse, label='nrmse')
        plt.plot(np.arange(len(nmae)), nmae, label='nmae')
        plt.savefig('Beijing_{}.png'.format(fold))
        plt.close()
    
    with open('log.txt', 'a') as f:
        f.write('radar_5fold_d_{}:\n \
                 rmse: {}\t{}\n\
                 mae: {}\t{}\n'.format(dim_embedding, np.mean(rmse_list), np.std(rmse_list)/np.sqrt(5), np.mean(mae_list), np.std(mae_list)/np.sqrt(5)))

        
if __name__ == '__main__':
    # test_mv5(3)
    test_beijing5(3)
    # test_server5(3)
    # test_radar5(3)
    # test_traffic5(3)
    # test_beijing_extrap(3)
    # test_ctr5(3)
    # test_server5(3)
    # test_dblp5(3)
    


