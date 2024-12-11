# 导入 GCLMVST
import datetime
import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics.cluster import adjusted_rand_score
from GCLMVST.gclmvst import training_model
from GCLMVST.utils import mclust
from GCLMVST.config import set_arg
import matplotlib
matplotlib.use('Agg')
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
experiment_name = '1'
experiment_dir = os.path.join(r'E:\论文\MuC\dlpfc\151672', experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = 'D:\software\R\R-4.4.1'
os.environ['R_USER'] = r'D:\ProgramData\Anaconda3\Lib\site-packages\rpy2'


opt = set_arg()
arg = opt.parse_args(['--mode_his', 'noh'])
arg.n_domain=5
arg.radius = 150
arg.epoch = 100
arg.drop_feat_p = 0.2
print(arg)

section_id = '151672'
input_dir = os.path.join('Data/DLPFC', section_id)
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()


Ann_df = pd.read_csv(os.path.join('Data/DLPFC/151672', section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
print('Ann_df.columns111111111111111111111111111111',Ann_df.columns)
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
plt.rcParams["figure.figsize"] = (15, 15)
sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"], title=' ', frameon=False)
output_path = os.path.join(experiment_dir, f"{section_id}_Ground Truth_{timestamp}.png")
plt.savefig(output_path)
print(f"Figure saved to {output_path}")

training_model(adata, arg)

adata = mclust(adata, arg, refine=True)

adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
print('Adjusted rand index = %.4f' %ARI)
nmi_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['Ground Truth'])
print('NMI', nmi_score)

fmi_score = fowlkes_mallows_score(obs_df['mclust'], obs_df['Ground Truth'])
print('FMI', fmi_score)


plot_color=[ "#d62728", "#9467bd","#e377c2", "#8c564b", "#ff7f0e", "#2ca02c","#1f77b4" ]
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.spatial(adata, color=["mclust"],palette=plot_color, title=' ', frameon=False)

output_path = os.path.join(experiment_dir, f"{section_id}_空间架构_{timestamp}.png")
plt.savefig(output_path)
print(f"Figure saved to {output_path}")

