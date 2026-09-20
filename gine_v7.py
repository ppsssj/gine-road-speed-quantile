# Historical experimental baseline.
# This implementation is preserved for research reproducibility.
# See README.md and docs/research-notes.md for known limitations.

# ====== GINE (Quantile Classification + Speed Regression, v6-posthoc, mini-batch) ======
# - 미니배치(DataLoader)로 그래프별 독립 학습 → edge_index 재사용 문제 제거
# - 좌표 스냅(그리드 스냅, 기본 8m)로 실제 도로 연결 복원
# - kNN 보조엣지(가중 상향) + DropEdge 완화
# - Train-only 표준화(노드/엣지/속도), Soft Label, Prior-bias, 회귀 힌트, T-scaling 유지
# ----------------------------------------------------------------------------------------

import os, glob, json, math, random, argparse, datetime as dt
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm

# ---------------- Config (기본값) ----------------
SEED=42
NUM_CLASSES = 4
K_NEIGHBORS = 3
KNN_WEIGHT  = 0.25        # ↑ 단절 환경에서 보조엣지 영향 강화
EPOCHS      = 300
LR          = 5e-4
PATIENCE    = 30
WEIGHT_DECAY= 3e-4
GRAD_CLIP   = 1.0
DROPE_P     = 0.10        # ↓ 완화
REG_LAMBDA  = 0.05
SOFT_LAMBDA = 0.30
TEMP_SCALE  = 1.1
MARGIN_KMH  = 5.0
ETA_SPEED_HINT = 0.6
SNAP_GRID_M = 8.0         # 좌표 스냅 그리드(미터 기준 좌표일 때 효과적)
BATCH_SIZE  = 2
VAL_RATIO   = 0.2         # 노드가 아니라 그래프(파일) 기준 검증 분할

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Utils ----------------
def get_device(arg: str):
    """'auto'면 CUDA 가능 시 cuda, 아니면 cpu. 사용자가 'cpu', 'cuda', 'cuda:0' 등 지정 시 그대로 사용."""
    if arg is None or str(arg).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(arg)
    except Exception:
        print(f"[WARN] Unsupported device='{arg}', fallback to CPU")
        return torch.device("cpu")

def _bearing(p,q):
    dx,dy = q[0]-p[0], q[1]-p[1]
    return math.atan2(dy,dx)

def _euclid(p,q): return math.hypot(q[0]-p[0], q[1]-p[1])

def _one_hot(idx,n):
    v=[0.0]*n
    if 0<=idx<n: v[idx]=1.0
    return v

def _roadtype_to_idx(rt):
    try: return max(0, int(rt) % 10)
    except: return 0

def _mode(lst, default="000"):
    return max(set(lst), key=lst.count) if lst else default

def parse_name(path):
    # "YYYY-MM-DD_HH.json" → (date_str, hour, weekday, is_weekend)
    name=os.path.basename(path).split(".")[0]
    dpart, hpart = name.split("_")
    y,m,d = map(int, dpart.split("-"))
    hour  = int(hpart)
    wk    = dt.date(y,m,d).weekday()
    is_wkend = 1 if wk>=5 else 0
    return dpart, hour, wk, is_wkend

def hour_to_slot6(hour):
    if 0<=hour<4: return 0
    if 4<=hour<8: return 1
    if 8<=hour<12: return 2
    if 12<=hour<16: return 3
    if 16<=hour<20: return 4
    return 5

def collect_speed_from_json(path):
    with open(path,"r",encoding="utf-8") as f:
        data=json.load(f)
    feats=data["features"] if isinstance(data,dict) else data
    speeds=[]
    for f in feats:
        p=f["properties"]
        sp=p.get("speed", None)
        if sp is None:
            dist=p.get("distance"); tm=p.get("time")
            if dist is not None and tm not in (None,0):
                sp=(float(dist)/float(tm))*3.6
        if sp is not None and np.isfinite(float(sp)):
            speeds.append(float(sp))
    return speeds

def compute_global_thresholds_by_slot(train_paths):
    slot_speeds=[[] for _ in range(6)]
    for p in train_paths:
        _, hour, _, _ = parse_name(p)
        slot=hour_to_slot6(hour)
        slot_speeds[slot].extend(collect_speed_from_json(p))
    thresholds=[]
    for s in range(6):
        arr=np.array(slot_speeds[s], float)
        if arr.size==0:
            thresholds.append((20.0,40.0,60.0))
        else:
            q25,q50,q75=np.percentile(arr,[25,50,75])
            thresholds.append((float(q25), float(q50), float(q75)))
    return thresholds

def speed_to_class(speed, q25,q50,q75):
    if speed < q25: return 3
    elif speed < q50: return 2
    elif speed < q75: return 1
    else: return 0

def speed_to_class_batch(speeds, q25, q50, q75):
    cls = np.zeros_like(speeds, dtype=np.int64)
    cls[speeds <  q25] = 3
    cls[(speeds >= q25) & (speeds < q50)] = 2
    cls[(speeds >= q50) & (speeds < q75)] = 1
    cls[speeds >= q75] = 0
    return cls

def soft_label_from_speed(speed, q25, q50, q75, m=MARGIN_KMH):
    probs = [0.0, 0.0, 0.0, 0.0]
    def lin_ratio(dist):  # dist=0→1, dist>=m→0
        return max(0.0, min(1.0, 1.0 - dist/m))
    if speed < q25:
        t = lin_ratio(q25 - speed)
        if t>0:  probs[3] += (1.0 - t); probs[2] += t
        else:    probs[3] = 1.0
    elif speed < q50:
        t_low  = lin_ratio(speed - q25)
        t_high = lin_ratio(q50 - speed)
        if t_low==0 and t_high==0:
            probs[2] = 1.0
        else:
            probs[2] += (1.0 - t_low);  probs[3] += t_low
            probs[2] += (1.0 - t_high); probs[1] += t_high
            s=sum(probs); probs=[p/s for p in probs] if s>0 else [0,0,1,0]
    elif speed < q75:
        t_low  = lin_ratio(speed - q50)
        t_high = lin_ratio(q75 - speed)
        if t_low==0 and t_high==0:
            probs[1]=1.0
        else:
            probs[1]+=(1.0 - t_low); probs[2]+=t_low
            probs[1]+=(1.0 - t_high); probs[0]+=t_high
            s=sum(probs); probs=[p/s for p in probs] if s>0 else [0,1,0,0]
    else:
        t = lin_ratio(speed - q75)
        if t>0:  probs[0] += (1.0 - t); probs[1] += t
        else:    probs[0] = 1.0
    s=sum(probs)
    if s==0:
        hard = 0 if speed>=q75 else 1 if speed>=q50 else 2 if speed>=q25 else 3
        probs[hard]=1.0
    else:
        probs=[p/s for p in probs]
    return probs

def soft_ce_loss(logits, target_probs):
    return (-F.log_softmax(logits, dim=1) * target_probs).sum(dim=1).mean()

def apply_dropedge(edge_index: torch.Tensor, edge_attr: torch.Tensor, p: float):
    if p <= 0 or edge_index.numel() == 0:
        return edge_index, edge_attr
    E = edge_index.size(1)
    mask = (torch.rand(E, device=edge_index.device) > p)
    return edge_index[:, mask], edge_attr[mask]

def snap_coord(pt, grid=SNAP_GRID_M):
    # 좌표계가 미터 단위(EPSG:3857 등)라고 가정
    return (round(pt[0]/grid)*grid, round(pt[1]/grid)*grid)

# ---------------- Graph Loader (per-file) ----------------
def load_graph_from_json(path, thresholds_by_slot, k_neighbors=K_NEIGHBORS, knn_weight=KNN_WEIGHT, snap_grid=SNAP_GRID_M):
    with open(path,"r",encoding="utf-8") as f:
        data=json.load(f)
    feats=data["features"] if isinstance(data,dict) else data

    _, hour, wk, is_wkend = parse_name(path)
    slot = hour_to_slot6(hour)
    sin_h, cos_h = math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24)
    q25,q50,q75 = thresholds_by_slot[slot]

    # 1) 노드 사전(스냅 적용)
    coord_to_idx, idx_to_coord, node_payload = {}, [], {}
    real_edges=set()

    for f in feats:
        props=f["properties"]; coords=f["geometry"]["coordinates"]
        if isinstance(coords[0], (int,float)):
            raw_s=tuple(coords); raw_t=tuple(coords)
        else:
            raw_s=tuple(coords[0]); raw_t=tuple(coords[-1])
        spt = snap_coord(raw_s, snap_grid); tpt = snap_coord(raw_t, snap_grid)

        for p in (spt,tpt):
            if p not in coord_to_idx:
                i=len(idx_to_coord); coord_to_idx[p]=i; idx_to_coord.append(p)
                node_payload[i]={"spd_wsum":0,"dist_wsum":0,"time_wsum":0,"w_sum":0,"rt_list":[]}
        s,t=coord_to_idx[spt], coord_to_idx[tpt]
        if s!=t:
            real_edges.add((s,t)); real_edges.add((t,s))

        # 속도 추정 (없으면 distance/time*3.6)
        sp=props.get("speed", None)
        if sp is None:
            dist=props.get("distance"); tm=props.get("time")
            if dist is not None and tm not in (None,0):
                sp=(float(dist)/float(tm))*3.6
        sp=float(sp) if sp is not None else 0.0
        w=float(props.get("distance",1.0))  # 거리 기반 가중치

        for node in (s,t):
            i=node
            node_payload[i]["spd_wsum"]  += sp*w
            node_payload[i]["time_wsum"] += float(props.get("time",0))*w
            node_payload[i]["dist_wsum"] += w
            node_payload[i]["w_sum"]     += w
            rt=props.get("roadType")
            if rt is not None: node_payload[i]["rt_list"].append(rt)

    n=len(idx_to_coord)
    if n==0:
        raise RuntimeError(f"Empty graph after snapping: {path}")

    spd=np.zeros(n); dist=np.zeros(n); tm=np.zeros(n); rt_idx=np.zeros(n)
    for i in range(n):
        w=max(1e-6, node_payload[i]["w_sum"])
        spd[i]= node_payload[i]["spd_wsum"]/w
        dist[i]=node_payload[i]["dist_wsum"]/w
        tm[i]=  node_payload[i]["time_wsum"]/w
        rt_idx[i]=_roadtype_to_idx(_mode(node_payload[i]["rt_list"]))

    # 2) kNN + real edges
    coords_arr=np.array(idx_to_coord)
    # n_neighbors+1 (자기 자신 포함)
    nbrs=NearestNeighbors(n_neighbors=min(k_neighbors+1, n)).fit(coords_arr)
    _, idxs=nbrs.kneighbors(coords_arr)
    edges=set()
    for i in range(n):
        for j in idxs[i][1:]:
            if i!=j:
                edges.add((i,j)); edges.add((j,i))
    edges |= real_edges
    edges=sorted(list(edges))

    # 3) 노드 피처
    deg=np.zeros(n)
    for s,t in edges: deg[s]+=1
    nbrs_of=[[] for _ in range(n)]
    for s,t in edges: nbrs_of[s].append(t)
    nbr_mean=np.array([np.mean(spd[nbrs_of[i]]) if nbrs_of[i] else spd[i] for i in range(n)])
    delta_spd=np.abs(spd - nbr_mean)
    onehots=np.array([_one_hot(int(rt_idx[i]), 10) for i in range(n)])

    time_oh=[0]*6; time_oh[slot]=1
    time_feats = np.array([*time_oh, sin_h, cos_h, is_wkend], dtype=float)  # 6+2+1=9
    time_mat = np.tile(time_feats, (n,1))

    X=np.stack([[dist[i], spd[i], tm[i], deg[i], nbr_mean[i], delta_spd[i]] for i in range(n)], axis=0)
    X=np.concatenate([X, onehots, time_mat], axis=1)  # 6 + 10 + 9 = 25

    # 4) 타겟
    y_cls = np.array([speed_to_class(spd[i], q25,q50,q75) for i in range(n)], dtype=np.int64)
    y_reg = spd.astype(np.float32)
    y_soft= np.vstack([soft_label_from_speed(spd[i], q25,q50,q75, m=MARGIN_KMH) for i in range(n)]).astype(np.float32)

    # 5) 엣지 속성
    edge_attr=[]
    for s,t in edges:
        p,q=idx_to_coord[s], idx_to_coord[t]
        base_w = 1.0 if (s,t) in real_edges else knn_weight
        dval=_euclid(p,q)
        theta=_bearing(p,q); sin_b,cos_b=math.sin(theta), math.cos(theta)
        spd_diff=abs(spd[s]-spd[t])
        rt_diff=0.0 if rt_idx[s]==rt_idx[t] else 1.0
        edge_attr.append([dval*base_w, spd_diff*base_w, sin_b, cos_b, rt_diff*base_w])
    edge_attr=np.array(edge_attr, float)

    data=Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y_cls, dtype=torch.long),
        y_speed=torch.tensor(y_reg, dtype=torch.float32),
        y_soft=torch.tensor(y_soft, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
    )
    data.meta = {"path": path, "hour": hour, "slot": slot, "thresholds": (q25,q50,q75)}
    return data

# ---------------- Model ----------------
class GINE_MultiTask(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, p=0.35, num_classes=NUM_CLASSES):
        super().__init__()
        self.mlp1=nn.Sequential(nn.Linear(in_dim,hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden,hidden))
        self.conv1=GINEConv(self.mlp1, edge_dim=edge_dim); self.bn1=BatchNorm(hidden)
        self.mlp2=nn.Sequential(nn.Linear(hidden,hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden,hidden))
        self.conv2=GINEConv(self.mlp2, edge_dim=edge_dim); self.bn2=BatchNorm(hidden)
        self.cls_head = nn.Linear(hidden, num_classes)
        self.reg_head = nn.Linear(hidden, 1)
        self.register_buffer("logit_bias", torch.zeros(num_classes))

    def forward(self, x, edge_index, edge_attr):
        x=F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x=F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        logits = self.cls_head(x)
        if hasattr(self, "logit_bias") and self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits, self.reg_head(x).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma=gamma
    def forward(self, logits, target):
        logp=F.log_softmax(logits, dim=1)
        p=torch.exp(logp)
        pt=p.gather(1, target.unsqueeze(1)).squeeze(1)
        logpt=logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss=-(1-pt)**self.gamma * logpt
        if self.alpha is not None:
            at=self.alpha.gather(0, target)
            loss=loss*at
        return loss.mean()

# ---------------- Train / Eval ----------------
def compute_global_norm_stats(train_graphs):
    # 학습 그래프 전체 노드/엣지 기준 평균, 표준편차
    Xs=[]; Es=[]; Ys=[]
    for g in train_graphs:
        Xs.append(g.x)
        Es.append(g.edge_attr)
        Ys.append(g.y_speed)
    X_all=torch.cat(Xs, dim=0)
    E_all=torch.cat(Es, dim=0) if len(Es)>0 else None
    Y_all=torch.cat(Ys, dim=0)

    mu_x=X_all.mean(0,keepdim=True); sd_x=X_all.std(0,keepdim=True).clamp_min(1e-6)
    if E_all is not None:
        mu_e=E_all.mean(0,keepdim=True); sd_e=E_all.std(0,keepdim=True).clamp_min(1e-6)
    else:
        mu_e=torch.zeros(1,5); sd_e=torch.ones(1,5)
    mu_y=Y_all.mean(); sd_y=Y_all.std().clamp_min(1e-6)
    return mu_x, sd_x, mu_e, sd_e, mu_y, sd_y

def apply_norm_inplace(graphs, mu_x, sd_x, mu_e, sd_e, mu_y, sd_y):
    for g in graphs:
        g.x = (g.x - mu_x) / sd_x
        g.edge_attr = (g.edge_attr - mu_e) / sd_e
        g.y_speed = (g.y_speed - mu_y) / sd_y

def train_one_epoch(model, loader, opt, focal, drope_p=DROPE_P, device=torch.device("cpu")):
    model.train()
    total, ce_sum, sce_sum, mse_sum = 0.0, 0.0, 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        ei_do, ea_do = apply_dropedge(batch.edge_index, batch.edge_attr, drope_p)
        logits, yhat = model(batch.x, ei_do, ea_do)
        loss_ce  = focal(logits, batch.y)
        loss_sce = soft_ce_loss(logits, batch.y_soft)
        loss_mse = F.mse_loss(yhat, batch.y_speed)
        loss = loss_ce + SOFT_LAMBDA*loss_sce + REG_LAMBDA*loss_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        bs = batch.num_nodes
        total += bs
        ce_sum  += loss_ce.item()*bs
        sce_sum += loss_sce.item()*bs
        mse_sum += loss_mse.item()*bs
    return ce_sum/total, sce_sum/total, mse_sum/total

@torch.no_grad()
def validate(model, loader, focal,device=torch.device("cpu")):
    model.eval()
    total, ce_sum, sce_sum, mse_sum = 0.0, 0.0, 0.0, 0.0
    for batch in loader:
        batch = batch.to(device)
        logits, yhat = model(batch.x, batch.edge_index, batch.edge_attr)
        loss_ce  = focal(logits, batch.y)
        loss_sce = soft_ce_loss(logits, batch.y_soft)
        loss_mse = F.mse_loss(yhat, batch.y_speed)
        bs = batch.num_nodes
        total += bs
        ce_sum  += loss_ce.item()*bs
        sce_sum += loss_sce.item()*bs
        mse_sum += loss_mse.item()*bs
    vloss = ce_sum/total + SOFT_LAMBDA*(sce_sum/total) + REG_LAMBDA*(mse_sum/total)
    return vloss, ce_sum/total, sce_sum/total, mse_sum/total

@torch.no_grad()
def fit_prior_bias(model, loader):
    # 검증 세트의 true prior / pred prior 비교 → logit_bias 갱신
    # 안전하게 디바이스를 맞추고, loader가 None이거나 비어있으면 아무 작업도 하지 않음
    if loader is None:
        return
    model.eval()
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    pred_counts = torch.zeros(NUM_CLASSES, device=device)
    true_counts = torch.zeros(NUM_CLASSES, device=device)
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.edge_attr)
        pred = logits.argmax(1)
        for c in range(NUM_CLASSES):
            pred_counts[c] += (pred==c).sum()
            true_counts[c] += (batch.y==c).sum()
    pred_prior = (pred_counts+1e-6) / (pred_counts.sum()+1e-6)
    true_prior = (true_counts+1e-6) / (true_counts.sum()+1e-6)
    bias_vec = torch.log(true_prior) - torch.log(pred_prior)
    bias_vec = torch.clamp(bias_vec, -0.5, 0.5)
    # model.logit_bias may be on CPU or CUDA; ensure same device
    model.logit_bias.copy_(bias_vec.to(model.logit_bias.device))

@torch.no_grad()
def test_per_graph(model, graphs, mu_y, sd_y, thresholds_by_slot, device=torch.device("cpu")):
    model.eval()
    # mu_y / sd_y may be tensors or scalars; move to device for safe arithmetic
    if isinstance(mu_y, torch.Tensor):
        mu_y_dev = mu_y.to(device)
    else:
        mu_y_dev = torch.tensor(mu_y, device=device, dtype=torch.float32)
    if isinstance(sd_y, torch.Tensor):
        sd_y_dev = sd_y.to(device)
    else:
        sd_y_dev = torch.tensor(sd_y, device=device, dtype=torch.float32)

    for g in graphs:
        g = g.to(device)
        logits, yhat_std = model(g.x, g.edge_index, g.edge_attr)
        # 회귀 힌트: yhat_std는 표준화된 출력
        yhat_kmh = (yhat_std * sd_y_dev + mu_y_dev).cpu().numpy()
        slot = g.meta["slot"]
        q25,q50,q75 = thresholds_by_slot[slot]
        reg_cls = speed_to_class_batch(yhat_kmh, q25,q50,q75)
        hint = torch.zeros_like(logits)
        hint[torch.arange(hint.size(0)), torch.tensor(reg_cls, device=hint.device)] = ETA_SPEED_HINT
        logits = logits + hint

        probs = F.softmax(logits / TEMP_SCALE, dim=1)
        pred_cls = probs.argmax(dim=1).cpu().numpy()
        true_cls = g.y.cpu().numpy()

        acc = accuracy_score(true_cls, pred_cls)
        cm  = confusion_matrix(true_cls, pred_cls, labels=[0,1,2,3])
        print(f"📅 {os.path.basename(g.meta['path'])} → ACC={acc*100:.2f}%")
        print("Confusion Matrix (rows=true, cols=pred):\n", cm)
        print(classification_report(true_cls, pred_cls, labels=[0,1,2,3], digits=3))
        print("Macro-F1:", f1_score(true_cls, pred_cls, labels=[0,1,2,3], average='macro'))

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--k", type=int, default=K_NEIGHBORS)
    ap.add_argument("--knn_weight", type=float, default=KNN_WEIGHT)
    ap.add_argument("--snap_grid", type=float, default=SNAP_GRID_M)
    ap.add_argument("--device", type=str, default="auto")
    args=ap.parse_args()

    device = get_device(args.device)
    print(f"[INFO] Device = {device}")
    
    files=sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not files:
        raise FileNotFoundError("data 폴더에 JSON이 없습니다.")

    # 마지막 4개 테스트
    train_paths=files[:-4]; test_paths=files[-4:]
    print(f"학습 파일 {len(train_paths)}개, 테스트 파일 {len(test_paths)}개")

    thresholds_by_slot=compute_global_thresholds_by_slot(train_paths)
    for s,(q25,q50,q75) in enumerate(thresholds_by_slot):
        print(f"시간대[{s}] 속도 분위수 km/h: 25%={q25:.2f}, 50%={q50:.2f}, 75%={q75:.2f}")

    # 그래프 로드
    train_graphs=[load_graph_from_json(p, thresholds_by_slot, k_neighbors=args.k, knn_weight=args.knn_weight, snap_grid=args.snap_grid) for p in train_paths]
    test_graphs =[load_graph_from_json(p, thresholds_by_slot, k_neighbors=args.k, knn_weight=args.knn_weight, snap_grid=args.snap_grid) for p in test_paths]

    # 학습 세트 내 라벨 분포 참고
    Y_train = torch.cat([g.y for g in train_graphs], dim=0)
    uniq, cnts = torch.unique(Y_train, return_counts=True)
    print("학습 라벨 분포:", {int(u): int(c) for u,c in zip(uniq, cnts)})
    class_weights=torch.zeros(NUM_CLASSES, dtype=torch.float32)
    total=float(Y_train.numel())
    for c in range(NUM_CLASSES):
        nc=float((Y_train==c).sum())
        class_weights[c]= total/(nc+1e-6)
    print("클래스 가중치(정규화 전):", class_weights.tolist())
    alpha=(class_weights/class_weights.sum()).to(dtype=torch.float32).to(device)

    # 학습/검증 분할(그래프 기준) — 안전하게 분할하여 val와 train이 겹치지 않도록 처리
    n_tr = len(train_graphs)
    if n_tr <= 1:
        # 검증을 사용할 수 없음 (그래프가 너무 적음)
        tr_graphs = train_graphs
        val_graphs = []
    else:
        # 최소 1개 이상, 그러나 n_val는 n_tr-1 이하로 제한
        n_val = max(1, int(round(n_tr * VAL_RATIO)))
        n_val = min(n_val, n_tr - 1)
        val_graphs = train_graphs[-n_val:]
        tr_graphs  = train_graphs[:-n_val]

    # Train-only 표준화(학습 그래프 전체 통계)
    mu_x, sd_x, mu_e, sd_e, mu_y, sd_y = compute_global_norm_stats(tr_graphs)
    apply_norm_inplace(tr_graphs, mu_x, sd_x, mu_e, sd_e, mu_y, sd_y)
    apply_norm_inplace(val_graphs, mu_x, sd_x, mu_e, sd_e, mu_y, sd_y)
    apply_norm_inplace(test_graphs, mu_x, sd_x, mu_e, sd_e, mu_y, sd_y)

    # DataLoader
    train_loader = DataLoader(tr_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False) if len(val_graphs) > 0 else None

    # 모델
    in_dim  = tr_graphs[0].x.shape[1]
    e_dim   = tr_graphs[0].edge_attr.shape[1]
    model   = GINE_MultiTask(in_dim=in_dim, edge_dim=e_dim, hidden=64, p=0.35).to(device)
    focal   = FocalLoss(alpha=alpha, gamma=2.0)

    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    best=float("inf"); bad=0; best_state=None
    for ep in range(1, args.epochs+1):
        tr_ce, tr_sce, tr_mse = train_one_epoch(model, train_loader, opt, focal, drope_p=DROPE_P, device=device)
        # 검증 데이터가 있는 경우 검증을 사용, 없으면 학습 손실을 대체 지표로 사용
        if val_loader is not None:
            vloss, v_ce, v_sce, v_mse = validate(model, val_loader, focal, device=device)
        else:
            v_ce, v_sce, v_mse = tr_ce, tr_sce, tr_mse
            vloss = v_ce + SOFT_LAMBDA * v_sce + REG_LAMBDA * v_mse
        sched.step(vloss)
        print(f"[{ep:03d}] train: CE={tr_ce:.4f}, SCE={tr_sce:.4f}, MSE={tr_mse:.4f} | "
              f"val={vloss:.4f} (CE={v_ce:.4f}, SCE={v_sce:.4f}, MSE={v_mse:.4f})")

        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print(f"⏹ Early stop {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # prior-bias 학습(검증 세트 기반) — val_loader가 있는 경우에만 수행
    if val_loader is not None:
        fit_prior_bias(model, val_loader)

    # 1) 가중치 저장
    torch.save(model.state_dict(), "best_model.pt")

    # 2) 전처리 통계 + 슬롯 분위수 저장
    np.savez(
        "train_stats_and_thresholds.npz",
        mu_x=mu_x.cpu().numpy(), sd_x=sd_x.cpu().numpy(),
        mu_e=mu_e.cpu().numpy(), sd_e=sd_e.cpu().numpy(),
        mu_y=np.array(mu_y.cpu()), sd_y=np.array(sd_y.cpu()),
        thresholds_by_slot=np.array(thresholds_by_slot, dtype=np.float32)
    )
    print("[SAVE] best_model.pt, train_stats_and_thresholds.npz 저장 완료")

    # 테스트
    print("\n🧪 Testing on next-day datasets...")
    test_per_graph(model, test_graphs, mu_y, sd_y, thresholds_by_slot, device=device)

if __name__=="__main__":
    main()
