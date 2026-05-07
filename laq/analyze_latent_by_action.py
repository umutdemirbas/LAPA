#!/usr/bin/env python3
"""
Latent space analysis with action/verb labels for Something-Something V2.

Outputs:
- PCA and t-SNE plots colored by verb and action template
- Clustering/separability metrics for verbs and actions
- Codebook usage heatmaps per verb
- Embeddings and metadata for downstream analysis
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from einops import pack

from laq_model.latent_action_quantization import LatentActionQuantization


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_template(template: str) -> str:
    cleaned = template.replace("[", "").replace("]", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def extract_verb(template: str) -> str:
    parts = template.strip().split()
    if not parts:
        return "unknown"
    return parts[0].lower()


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_label_metadata(labels_root: Path, split: str):
    labels_path = labels_root / "labels.json"
    train_path = labels_root / "train.json"
    val_path = labels_root / "validation.json"

    labels_map = read_json(labels_path)

    train_items = read_json(train_path)
    val_items = read_json(val_path)

    if split == "train":
        items = train_items
    elif split == "validation":
        items = val_items
    else:
        items = train_items + val_items

    sample_meta = {}
    for item in items:
        sample_id = str(item["id"])
        raw_template = item["template"]
        template = normalize_template(raw_template)
        verb = extract_verb(template)

        template_id = labels_map.get(template, None)
        sample_meta[sample_id] = {
            "template": template,
            "verb": verb,
            "template_id": int(template_id) if template_id is not None else -1,
            "raw_label": item.get("label", ""),
        }

    return sample_meta, labels_map


class Sthv2ActionDataset(Dataset):
    def __init__(self, data_root: Path, sample_meta: dict, image_size: int, offset: int):
        self.data_root = data_root
        self.sample_meta = sample_meta
        self.offset = offset

        self.items = []
        for entry in sorted(self.data_root.iterdir()):
            if not (entry.is_dir() or entry.suffix.lower() == ".webm"):
                continue

            sample_id = entry.stem if entry.suffix else entry.name
            if sample_id in self.sample_meta:
                self.items.append((entry, sample_id))

        if not self.items:
            raise RuntimeError(f"No labeled samples found under {self.data_root}")

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def _read_video_frame(self, video_path: Path, frame_index: int) -> Image.Image:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _load_pair(self, item_path: Path):
        if item_path.is_dir():
            frame_paths = [
                p
                for p in sorted(item_path.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            if not frame_paths:
                raise RuntimeError(f"No frames in directory: {item_path}")

            first_idx = 0
            second_idx = min(first_idx + self.offset, len(frame_paths) - 1)

            first_img = Image.open(frame_paths[first_idx])
            second_img = Image.open(frame_paths[second_idx])
        else:
            cap = cv2.VideoCapture(str(item_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {item_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count <= 0:
                raise RuntimeError(f"No frames in video: {item_path}")

            first_idx = 0
            second_idx = min(first_idx + self.offset, frame_count - 1)

            first_img = self._read_video_frame(item_path, first_idx)
            second_img = self._read_video_frame(item_path, second_idx)

        first_tensor = self.transform(first_img).unsqueeze(1)
        second_tensor = self.transform(second_img).unsqueeze(1)
        return torch.cat([first_tensor, second_tensor], dim=1)

    def __getitem__(self, index: int):
        item_path, sample_id = self.items[index]
        meta = self.sample_meta[sample_id]

        try:
            video = self._load_pair(item_path)
        except Exception:
            next_index = (index + 1) % len(self.items)
            return self.__getitem__(next_index)

        return {
            "video": video,
            "sample_id": sample_id,
            "verb": meta["verb"],
            "action": meta["template"],
            "template_id": meta["template_id"],
        }


def create_model(device: torch.device):
    model = LatentActionQuantization(
        dim=1024,
        quant_dim=32,
        codebook_size=8,
        image_size=256,
        patch_size=32,
        spatial_depth=8,
        temporal_depth=8,
        dim_head=64,
        heads=16,
        code_seq_len=4,
    )
    model = model.to(device)
    model.eval()
    return model


def unwrap_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model", "state_dict", "ema_model", "ema", "module"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    if hasattr(ckpt_obj, "state_dict"):
        return ckpt_obj.state_dict()
    raise RuntimeError("Unsupported checkpoint format")


def load_model_checkpoint(model, checkpoint_path: Path, device: torch.device):
    ckpt_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = unwrap_state_dict(ckpt_obj)

    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned)
    return {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_preview": missing[:10],
        "unexpected_preview": unexpected[:10],
    }


def extract_latents(model, loader, device: torch.device, max_samples: int, latent_strategy: str = "quantized_flat"):
    """
    Extract latents with different strategies:
    - 'quantized_flat': flatten VQ quantized output (original)
    - 'encoded_tokens_pooled': mean-pool encoded tokens before VQ (RECOMMENDED)
    - 'encoded_tokens_flat': flatten encoded tokens before VQ
    """
    latent_list = []
    index_list = []
    verbs = []
    actions = []
    sample_ids = []

    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["video"].to(device)

            # Preprocess videos through patch embedding (required before encode)
            first_frame, rest_frames = videos[:, :, :1], videos[:, :, 1:]
            first_frame_tokens = model.to_patch_emb_first_frame(first_frame)
            rest_frames_tokens = model.to_patch_emb_first_frame(rest_frames)
            tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim=1)

            first_tokens, last_tokens = model.encode(tokens)
            
            # Pack tokens for VQ (flatten spatial dimensions: [b, t, h, w, d] -> [b, t*h*w, d])
            first_tokens_packed, first_packed_shape = pack([first_tokens], 'b * d')
            last_tokens_packed, last_packed_shape = pack([last_tokens], 'b * d')
            
            # VQ takes both tokens together
            quantized, perplexity, codebook_usage, indices = model.vq(first_tokens_packed, last_tokens_packed)
            
            # Choose latent representation strategy
            if latent_strategy == "encoded_tokens_pooled":
                # Mean-pool encoded tokens: more stable, avoids VQ collapse
                latents = (first_tokens_packed.mean(dim=1) + last_tokens_packed.mean(dim=1)) / 2
            elif latent_strategy == "encoded_tokens_flat":
                # Flatten encoded tokens before VQ
                latents = torch.cat([first_tokens_packed, last_tokens_packed], dim=1)
            else:  # quantized_flat (original)
                # Flatten VQ quantized output
                latents = quantized.flatten(start_dim=1)
            
            codes = indices

            latent_list.append(latents.detach().cpu().numpy())
            index_list.append(codes.detach().cpu().numpy())

            verbs.extend(batch["verb"])
            actions.extend(batch["action"])
            sample_ids.extend(batch["sample_id"])

            seen += latents.shape[0]
            if seen >= max_samples:
                break

    if not latent_list:
        raise RuntimeError("No latents extracted")

    X = np.concatenate(latent_list, axis=0)[:max_samples]
    idx = np.concatenate(index_list, axis=0)[:max_samples]
    verbs = np.array(verbs[:max_samples])
    actions = np.array(actions[:max_samples])
    sample_ids = np.array(sample_ids[:max_samples])

    return X, idx, verbs, actions, sample_ids


def _sample_for_metrics(X, y, limit=4000):
    if len(X) <= limit:
        return X, y
    indices = np.random.choice(len(X), size=limit, replace=False)
    return X[indices], y[indices]


def separability_metrics(X, labels, metric_name: str):
    enc = LabelEncoder()
    y = enc.fit_transform(labels)

    out = {
        "metric_name": metric_name,
        "num_samples": int(len(labels)),
        "num_classes": int(len(enc.classes_)),
    }

    if len(enc.classes_) < 2:
        out["error"] = "Need at least 2 classes"
        return out

    X_s, y_s = _sample_for_metrics(X, y)

    try:
        out["silhouette"] = float(silhouette_score(X_s, y_s))
    except Exception as e:
        out["silhouette_error"] = str(e)

    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_s, y_s))
    except Exception as e:
        out["calinski_harabasz_error"] = str(e)

    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X_s, y_s))
    except Exception as e:
        out["davies_bouldin_error"] = str(e)

    # Linear probe accuracy
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.2, random_state=42
        )

    try:
        # Scale features for better convergence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=5000, solver='lbfgs', tol=1e-4, random_state=42)
        clf.fit(X_train_scaled, y_train)
        pred = clf.predict(X_test_scaled)
        out["linear_probe_acc"] = float(accuracy_score(y_test, pred))
    except Exception as e:
        out["linear_probe_error"] = str(e)

    # Unsupervised agreement with labels
    try:
        k = min(len(enc.classes_), 30, len(X_s) - 1)
        if k >= 2:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_s)
            out["kmeans_k"] = int(k)
            out["ari"] = float(adjusted_rand_score(y_s, km_labels))
            out["nmi"] = float(normalized_mutual_info_score(y_s, km_labels))
    except Exception as e:
        out["kmeans_error"] = str(e)

    return out


def scatter_top_labels(X2, labels, title: str, out_path: Path, top_n=12):
    counts = Counter(labels)
    top_labels = [name for name, _ in counts.most_common(top_n)]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for label in top_labels:
        mask = labels == label
        ax.scatter(X2[mask, 0], X2[mask, 1], s=8, alpha=0.6, label=f"{label} ({mask.sum()})")

    other_mask = ~np.isin(labels, np.array(top_labels))
    if other_mask.any():
        ax.scatter(X2[other_mask, 0], X2[other_mask, 1], s=6, alpha=0.2, color="gray", label="other")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_centroid_heatmap(X, labels, out_path: Path, top_n=12):
    counts = Counter(labels)
    top_labels = [name for name, _ in counts.most_common(top_n)]

    centroids = []
    kept = []
    for label in top_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue
        centroids.append(X[mask].mean(axis=0))
        kept.append(label)

    if len(centroids) < 2:
        return

    C = np.vstack(centroids)
    # cosine distance via normalized dot product
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    sim = Cn @ Cn.T
    dist = 1.0 - sim

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    sns.heatmap(dist, xticklabels=kept, yticklabels=kept, cmap="mako", square=True, ax=ax)
    ax.set_title("Verb Centroid Cosine Distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_codebook_heatmap(indices, verbs, out_path: Path, top_n=12):
    if indices.size == 0:
        return

    codebook_size = int(indices.max()) + 1
    counts = Counter(verbs)
    top_verbs = [v for v, _ in counts.most_common(top_n)]

    matrix = np.zeros((len(top_verbs), codebook_size), dtype=np.float64)

    for i, verb in enumerate(top_verbs):
        mask = verbs == verb
        verb_codes = indices[mask].reshape(-1)
        if verb_codes.size == 0:
            continue
        hist = np.bincount(verb_codes, minlength=codebook_size).astype(np.float64)
        hist = hist / (hist.sum() + 1e-12)
        matrix[i] = hist

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(matrix, xticklabels=list(range(codebook_size)), yticklabels=top_verbs, cmap="viridis", ax=ax)
    ax.set_title("Codebook Usage by Verb (row-normalized)")
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Verb")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def global_codebook_metrics(indices):
    flat = indices.reshape(-1)
    if flat.size == 0:
        return {}

    total = float(flat.size)
    counts = np.bincount(flat)
    probs = counts / total
    probs = probs[probs > 0]

    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    perplexity = float(np.exp(entropy))
    utilization = float((counts > 0).sum() / max(1, len(counts)))

    return {
        "num_codes_total": int(len(counts)),
        "num_codes_used": int((counts > 0).sum()),
        "utilization": utilization,
        "entropy": entropy,
        "perplexity": perplexity,
    }


def run(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    labels_root = Path(args.labels_root)

    sample_meta, labels_map = load_label_metadata(labels_root, args.split)

    dataset = Sthv2ActionDataset(
        data_root=data_root,
        sample_meta=sample_meta,
        image_size=args.image_size,
        offset=args.offset,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = create_model(device)
    load_info = load_model_checkpoint(model, checkpoint_path, device)

    # Extract latents with specified strategy
    latent_strategy = getattr(args, 'latent_strategy', 'quantized_flat')
    X, indices, verbs, actions, sample_ids = extract_latents(
        model=model,
        loader=loader,
        device=device,
        max_samples=args.max_samples,
        latent_strategy=latent_strategy,
    )

    step = checkpoint_path.stem.split(".")[-1]
    out_dir = Path(args.output_dir) / f"action_verb_analysis_step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature normalization (important for t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Log diagnostic info about latent quality
    print(f"Latent strategy: {latent_strategy}")
    print(f"Raw latent shape: {X.shape}, mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"Scaled latent shape: {X_scaled.shape}, mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    # Dimensionality reductions
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_scaled)

    # Stratified sampling for t-SNE (ensures representation of all verbs/actions)
    tsne_n = min(args.tsne_samples, len(X_scaled))
    
    # Use stratified sampling if dataset is large enough
    unique_verbs = len(np.unique(verbs))
    if tsne_n > unique_verbs * 10:  # enough samples for stratification
        from sklearn.model_selection import train_test_split
        # Stratified split to get representative sample
        indices_to_keep, _ = train_test_split(
            np.arange(len(X_scaled)),
            train_size=tsne_n,
            stratify=verbs,
            random_state=42
        )
        tsne_idx = indices_to_keep
    else:
        tsne_idx = np.random.choice(len(X_scaled), size=tsne_n, replace=False)
    
    X_tsne_in = X_scaled[tsne_idx]
    verbs_tsne = verbs[tsne_idx]
    actions_tsne = actions[tsne_idx]

    if tsne_n >= 10:
        # Adaptive perplexity: larger datasets benefit from higher perplexity
        # Rule of thumb: perplexity should be roughly 5-50, but can go higher for large datasets
        adaptive_perplexity = min(
            args.tsne_perplexity,
            max(5, int(np.sqrt(tsne_n) / 2))  # adaptive: sqrt(n)/2, capped at tsne_perplexity
        )
        
        print(f"t-SNE config: n_samples={tsne_n}, perplexity={adaptive_perplexity} (adaptive)")
        
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=adaptive_perplexity,
            init="pca",
            learning_rate="auto",
            n_iter=1000,  # explicit iterations for reproducibility
            verbose=1,
        )
        X_tsne = tsne.fit_transform(X_tsne_in)
    else:
        X_tsne = X_tsne_in[:, :2]

    # Metrics
    verb_metrics = separability_metrics(X_scaled, verbs, metric_name="verb")
    action_metrics = separability_metrics(X_scaled, actions, metric_name="action")
    codebook_metrics = global_codebook_metrics(indices)
    
    # Print diagnostic info to console
    print("\n" + "="*80)
    print("LATENT SPACE QUALITY DIAGNOSTICS")
    print("="*80)
    print(f"Verb Metrics: linear_probe_acc={verb_metrics.get('linear_probe_acc', 'N/A'):.3f}, silhouette={verb_metrics.get('silhouette', 'N/A'):.3f}")
    print(f"Action Metrics: linear_probe_acc={action_metrics.get('linear_probe_acc', 'N/A'):.3f}, silhouette={action_metrics.get('silhouette', 'N/A'):.3f}")
    if verb_metrics.get('linear_probe_acc', 0) < 0.5:
        print("⚠ WARNING: Poor verb linear probe accuracy suggests weak semantic encoding!")
        print("   Consider: (1) using encoded_tokens_pooled strategy, (2) larger model, (3) better pretraining")
    print("="*80 + "\n")

    # Visuals
    scatter_top_labels(
        X_pca,
        verbs,
        "PCA of Latents (top verbs)",
        out_dir / "pca_top_verbs.png",
        top_n=args.top_n_labels,
    )
    scatter_top_labels(
        X_pca,
        actions,
        "PCA of Latents (top actions)",
        out_dir / "pca_top_actions.png",
        top_n=args.top_n_labels,
    )
    scatter_top_labels(
        X_tsne,
        verbs_tsne,
        "t-SNE of Latents (top verbs)",
        out_dir / "tsne_top_verbs.png",
        top_n=args.top_n_labels,
    )
    scatter_top_labels(
        X_tsne,
        actions_tsne,
        "t-SNE of Latents (top actions)",
        out_dir / "tsne_top_actions.png",
        top_n=args.top_n_labels,
    )

    plot_codebook_heatmap(
        indices=indices,
        verbs=verbs,
        out_path=out_dir / "codebook_usage_by_verb.png",
        top_n=args.top_n_labels,
    )
    plot_centroid_heatmap(
        X=X,
        labels=verbs,
        out_path=out_dir / "verb_centroid_distance_heatmap.png",
        top_n=args.top_n_labels,
    )

    # Save artifacts
    np.savez_compressed(
        out_dir / "latent_embeddings.npz",
        latents=X,
        code_indices=indices,
        verbs=verbs,
        actions=actions,
        sample_ids=sample_ids,
    )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_samples": int(len(X)),
        "latent_dim": int(X.shape[1]),
        "num_unique_verbs": int(len(np.unique(verbs))),
        "num_unique_actions": int(len(np.unique(actions))),
        "pca_explained_variance_2d": pca2.explained_variance_ratio_.tolist(),
        "verb_metrics": verb_metrics,
        "action_metrics": action_metrics,
        "codebook_metrics": codebook_metrics,
        "checkpoint_load_info": load_info,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 80)
    print("ACTION/VERB LATENT ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Samples analyzed: {len(X)}")
    print(f"Latent dimension: {X.shape[1]}")
    print(f"Unique verbs: {len(np.unique(verbs))}")
    print(f"Unique actions: {len(np.unique(actions))}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results_v2/vae.14000.pt")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/cluster/scratch/udemirbas/LAPA/sth_v2_data/20bn-something-something-v2",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default="/cluster/scratch/udemirbas/LAPA/sth_v2_data/labels",
    )
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "all"])
    parser.add_argument("--output-dir", type=str, default="latent_analysis_results")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--offset", type=int, default=30)

    parser.add_argument(
        "--latent-strategy",
        type=str,
        default="quantized_flat",
        choices=["quantized_flat", "encoded_tokens_pooled", "encoded_tokens_flat"],
        help="Strategy for extracting latent representations: "
             "quantized_flat (original, VQ quantized + flattened), "
             "encoded_tokens_pooled (RECOMMENDED: mean pooled encoded tokens), "
             "encoded_tokens_flat (flattened encoded tokens before VQ)"
    )

    parser.add_argument("--tsne-samples", type=int, default=2500)
    parser.add_argument("--tsne-perplexity", type=int, default=50)
    parser.add_argument("--top-n-labels", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
