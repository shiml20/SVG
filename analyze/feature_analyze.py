# feature_token_space_analysis.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# plt.style.use('seaborn-deep')

def load_features(path):
    x = torch.load(path, map_location='cpu')
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        x = torch.stack(list(x)).numpy()
    else:
        x = np.array(x)
    return x

def ensure_hw_shape(feats):
    """
    Ensure feats shape is (N, C, H, W). If shape is (N_tokens, C) attempt to reshape to (1,C,H,W)
    But we expect inputs like:
      VAE: (N_images, C, H, W) or (1, C, H, W)
      DINO: (N_images, C, H, W) or (1, C, H, W)
    """
    if feats.ndim == 4:
        return feats
    # if (T, C) or (tokens, C), try guess H,W by square
    if feats.ndim == 2:
        tokens, C = feats.shape
        s = int(np.sqrt(tokens))
        if s * s == tokens:
            return feats.reshape(1, C, s, s)
    raise ValueError(f"Unsupported feature shape: {feats.shape}")

def token_norm_map(feats):
    # feats: (N, C, H, W) -> returns norms (N, H, W) and flattened tokens (N*H*W, C)
    N, C, H, W = feats.shape
    toks = feats.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    norms = np.linalg.norm(toks, axis=1).reshape(N, H, W)
    return norms, toks, (H, W), C

def upsample_token_map(arr, target_shape):
    """
    Nearest-neighbor upsample by integer factor using np.repeat.
    arr: (H, W) or (N, H, W)
    target_shape: (Ht, Wt)
    """
    single = False
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
        single = True
    N, H, W = arr.shape
    Ht, Wt = target_shape
    if (Ht % H != 0) or (Wt % W != 0):
        # fallback: simple numpy.repeat with non-integer requires interpolation.
        # but keep simple: use nearest by scaling indexes
        ys = (np.arange(Ht) * H / Ht).astype(int)
        xs = (np.arange(Wt) * W / Wt).astype(int)
        out = arr[:, ys[:, None], xs[None, :]]
    else:
        fy = Ht // H
        fx = Wt // W
        out = np.repeat(np.repeat(arr, fy, axis=1), fx, axis=2)
    return out[0] if single else out

def plot_heatmap(mat, title, fname, cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(4,4))
    plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def analyze(vae_path, dino_path, out_dir='analysis_out', pca_compare_dim=4):
    os.makedirs(out_dir, exist_ok=True)

    vae = load_features(vae_path)   # expected (1,C,H,W)
    dino = load_features(dino_path)

    vae = ensure_hw_shape(vae)
    dino = ensure_hw_shape(dino)

    print("Shapes (after ensure): VAE", vae.shape, "DINO", dino.shape)

    # -- token norm maps & flattened tokens --
    vae_norms_map, vae_toks, vae_hw, vae_C = token_norm_map(vae)
    dino_norms_map, dino_toks, dino_hw, dino_C = token_norm_map(dino)

    H_v, W_v = vae_hw
    H_d, W_d = dino_hw

    # save norm heatmaps (first image only)
    plot_heatmap(vae_norms_map[0], f'VAE token L2-norm ({H_v}x{W_v})', os.path.join(out_dir,'vae_norm_map.png'))
    plot_heatmap(dino_norms_map[0], f'DINO token L2-norm ({H_d}x{W_d})', os.path.join(out_dir,'dino_norm_map.png'))

    # if resolutions differ, upsample smaller to bigger for comparison (nearest neighbor)
    # choose target = VAE resolution
    target_hw = (H_v, W_v)
    dino_norm_up = upsample_token_map(dino_norms_map[0], target_hw)
    plot_heatmap(dino_norm_up, f'DINO norm up->VAE ({H_v}x{W_v})', os.path.join(out_dir,'dino_norm_up_to_vae.png'))

    # -- token norm statistics --
    def stats_from_normmap(name, normmap):
        vals = normmap.flatten()
        mean = vals.mean(); var = vals.var(); rel_var = var / (mean**2 + 1e-12)
        print(f"[{name}] tokens: mean_norm={mean:.4f}, var_norm={var:.4f}, rel_var={rel_var:.6f}")
        return vals

    vae_norm_vals = stats_from_normmap('VAE', vae_norms_map[0])
    dino_norm_vals = stats_from_normmap('DINO', dino_norms_map[0])
    dino_up_vals = dino_norm_up.flatten()

    # correlation between token norms (aligned by spatial position after upsample)
    corr_coef, pval = pearsonr(vae_norm_vals, dino_up_vals)
    print(f"Pearson corr (VAE norm vs DINO norm upsampled): r={corr_coef:.4f}, p={pval:.3e}")

    # -- channel PCA: per-feature-type intrinsic dimension estimation --
    def pca_cumulative(feats_tokens, name, out_prefix):
        # feats_tokens: (num_tokens, C)
        C = feats_tokens.shape[1]
        ncomp = min(feats_tokens.shape[0], C)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(feats_tokens)
        pca = PCA(n_components=ncomp)
        pca.fit(Xs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # plot
        plt.figure(figsize=(5,3))
        plt.plot(cumvar, marker='o', markersize=3)
        plt.xlabel('Principal Components')
        plt.ylabel('Cumulative explained var')
        plt.title(f'PCA cumvar - {name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{out_prefix}_pca_cumvar.png'), dpi=150)
        plt.close()
        # find elbow like index for 90%,95%
        def find_k(th):
            ks = np.where(cumvar >= th)[0]
            return int(ks[0] + 1) if len(ks)>0 else None
        k90 = find_k(0.90); k95 = find_k(0.95)
        print(f"{name} PCA: dims={C}, tokens={feats_tokens.shape[0]}, k90={k90}, k95={k95}")
        return pca, scaler, cumvar

    pca_vae, scaler_vae, cum_vae = pca_cumulative(vae_toks, 'VAE_channels', 'vae')
    pca_dino, scaler_dino, cum_dino = pca_cumulative(dino_toks, 'DINO_channels', 'dino')

    # -- project DINO channels to VAE channel-dim (so per-token vectors become comparable) --
    # We'll reduce DINO -> pca_compare_dim via PCA on DINO channels; also reduce VAE -> same dim
    k = min(pca_compare_dim, vae_C, dino_C)
    print(f"Projecting both to {k} dims for token-wise vector similarity comparison.")
    pca_dino_k = PCA(n_components=k).fit(scaler_dino.transform(dino_toks))
    dino_proj = pca_dino_k.transform(scaler_dino.transform(dino_toks))  # (tokens_d, k)

    pca_vae_k = PCA(n_components=k).fit(scaler_vae.transform(vae_toks))
    vae_proj = pca_vae_k.transform(scaler_vae.transform(vae_toks))        # (tokens_v, k)

    # if tokens counts differ, upsample token maps (nearest) in spatial grid then compare
    # reshape projected to spatial map
    vae_proj_map = vae_proj.reshape(1, H_v, W_v, k)[0]   # (H_v, W_v, k)
    dino_proj_map = dino_proj.reshape(1, H_d, W_d, k)[0] # (H_d, W_d, k)
    # upsample dino_proj_map -> target_hw
    # use repeat if integer factor, else index scaling
    if H_v % H_d == 0 and W_v % W_d == 0:
        fy = H_v // H_d; fx = W_v // W_d
        dino_proj_up = np.repeat(np.repeat(dino_proj_map, fy, axis=0), fx, axis=1)
    else:
        # general nearest mapping
        ys = (np.arange(H_v) * H_d / H_v).astype(int)
        xs = (np.arange(W_v) * W_d / W_v).astype(int)
        dino_proj_up = dino_proj_map[ys[:, None], xs[None, :], :]

    # compute per-token cosine similarities between projected vectors
    V = vae_proj_map.reshape(-1, k)
    D = dino_proj_up.reshape(-1, k)
    # normalize
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    cos_sim_per_token = (Vn * Dn).sum(axis=1)
    mean_cos = cos_sim_per_token.mean()
    std_cos = cos_sim_per_token.std()
    print(f"Token-wise cosine sim (after projecting to {k} dims): mean={mean_cos:.4f}, std={std_cos:.4f}")

    # save cos map
    cos_map = cos_sim_per_token.reshape(H_v, W_v)
    plot_heatmap(cos_map, f'Cosine sim per-token (proj {k}D)', os.path.join(out_dir,'token_cos_map.png'), cmap='RdYlBu', vmin=-1, vmax=1)

    # -- pairwise distance distributions (sample subset if too large) --
    def pairwise_dist_hist(feats, name, fname, max_samples=2000):
        m = feats.shape[0]
        idx = np.arange(m)
        if m > max_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(m, size=max_samples, replace=False)
        sample = feats[idx]
        # compute pairwise distances efficiently
        dists = pairwise_distances(sample, metric='euclidean')
        # take upper triangle
        tri = dists[np.triu_indices_from(dists, k=1)]
        plt.figure(figsize=(5,3))
        plt.hist(tri, bins=80, alpha=0.7)
        plt.title(f'Pairwise dist hist - {name}')
        plt.xlabel('euclidean distance')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()
        print(f"{name} pairwise dist median={np.median(tri):.4f}, mean={np.mean(tri):.4f}, std={np.std(tri):.4f}")

    pairwise_dist_hist(vae_toks, 'VAE_tokens', 'vae_pairwise_dist.png')
    pairwise_dist_hist(dino_toks, 'DINO_tokens', 'dino_pairwise_dist.png')

    # -- save some numeric summaries to file --
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(f"VAE shape: {vae.shape}, DINO shape: {dino.shape}\n")
        f.write(f"VAE token stats: mean_norm={vae_norm_vals.mean():.6f}, var_norm={vae_norm_vals.var():.6f}\n")
        f.write(f"DINO token stats: mean_norm={dino_norm_vals.mean():.6f}, var_norm={dino_norm_vals.var():.6f}\n")
        f.write(f"Pearson(norm) VAE vs DINO_up: r={corr_coef:.6f}, p={pval:.6e}\n")
        f.write(f"Token-wise cosine sim (proj {k}D): mean={mean_cos:.6f}, std={std_cos:.6f}\n")
        f.write("PCA cumvar samples: VAE first 10 cumvar: " + ','.join([f"{x:.4f}" for x in cum_vae[:10]]) + "\n")
        f.write("PCA cumvar samples: DINO first 10 cumvar: " + ','.join([f"{x:.4f}" for x in cum_dino[:10]]) + "\n")

    print("Saved outputs to", out_dir)
    return os.path.abspath(out_dir)

if __name__ == '__main__':
    vae_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/analyze/sdxl_vae.pt"
    dino_path = "/ytech_m2v3_hdd/yuanziyang/sml/FVG/analyze/dinov3_sp_feature.pt"
    analyze(vae_path, dino_path)
