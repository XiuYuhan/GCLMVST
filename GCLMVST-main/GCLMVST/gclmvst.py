import numpy as np
import torch
import tqdm.notebook as tq
from .model import Model
from .utils import fix_seed
from .data import preprocess_data
import time


def training_model(adata, arg):
    fix_seed(seed=arg.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if arg.mode_his == 'his':
        adata, spatial_adj, feature_adj, histology_ew = preprocess_data(adata, arg)
        raw_feature = torch.FloatTensor(adata.obsm['raw_feature'].copy()).to(device)

        g_h, w_h = histology_ew
        sh_edge = torch.concatenate([spatial_adj, g_h.to(device)], dim=-1)
        s_h = torch.ones(spatial_adj.size()[1]).to(device)
        sh_weight = torch.concatenate([s_h, w_h.to(device)], dim=-1)

        model = Model(raw_feature.shape[1], arg).to(device)
        print('model',model)
        # 使用 Adam 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)

        for ep in tq.tqdm(range(1, arg.epoch+1)):
            model.train()
            optimizer.zero_grad()
            loss = model(raw_feature, spatial_adj, feature_adj, sh_edge, sh_weight)[-1]
            if ep % 100 == 0:
                print(f'EP[{ep:4d}]: loss={loss.data:.4f}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            latent_rep = model(raw_feature, spatial_adj, feature_adj, sh_edge, sh_weight)[0]

        print('latent_rep shape:', latent_rep.shape)
        print('latent_rep (first 5 elements):', latent_rep[:5])
        latent = latent_rep.to('cpu').detach().numpy()

        adata.obsm['GCLMVST'] = latent
        print('adata.obsm["GCLMVST"] shape:', adata.obsm['GCLMVST'].shape)
        print('adata.obsm["GCLMVST"] (first 5 elements):', adata.obsm['GCLMVST'][:5])

    elif arg.mode_his == 'noh':
        adata, spatial_adj, feature_adj, s_e_attr, sf_e_attr = preprocess_data(adata, arg)
        raw_feature = torch.FloatTensor(adata.obsm['raw_feature'].copy()).to(device)


        model = Model(raw_feature.shape[1], arg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)

        losses = []
        for ep in tq.tqdm(range(1, arg.epoch + 1)):
            model.train()
            optimizer.zero_grad()
            if arg.use_eattr:
                loss = model(raw_feature, spatial_adj, feature_adj, w_h=s_e_attr, w_h_a=sf_e_attr)[-1]
            else:
                loss = model(raw_feature, spatial_adj, feature_adj)[-1]
                # loss = model(raw_feature, spatial_adj)[-1]
            timestamp = str(int(time.time()))
            if arg.epoch == ep:
                file_path_model = f'train_model/epoch{ep}_{timestamp}_ST_Gatkpca.pth'
                torch.save(model.state_dict(), file_path_model)
            losses.append(loss.item())

            if ep % (arg.epoch / arg.log_step) == 0:
                print(f'EP[%4d]: loss=%.4f.' % (ep, loss.data))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()


        import matplotlib.pyplot as plt
        x = range(1, len(losses) + 1)
        plt.plot(x, losses)
        plt.show()

        model.eval()
        with torch.no_grad():
            if arg.use_eattr:
                latent_rep, reconst_rep, _ = model(raw_feature, spatial_adj, feature_adj, w_h=s_e_attr, w_h_a=sf_e_attr)
            else:
                latent_rep, reconst_rep, _ = model(raw_feature, spatial_adj, feature_adj)
            if arg.epoch == ep:
                file_path_model = f'eval_model/epoch{ep}_{timestamp}_ST_Gatkpca.pth'
                torch.save(model.state_dict(), file_path_model)
        np.set_printoptions(suppress=True, precision=4, formatter={'float': '{:0.4f}'.format})
        latent = latent_rep.to('cpu').detach().numpy()
        reconst_rep = reconst_rep.to('cpu').detach().numpy()
        adata.obsm['GCLMVST'] = latent
        adata.obsm['rec'] = reconst_rep

    return adata
