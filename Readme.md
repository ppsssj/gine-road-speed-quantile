````md
# GINE Road Speed Quantile (Classification + Regression)

도로 네트워크를 **그래프(Graph)** 로 구성한 뒤,  
**(1) 시간대별(6-slot) 속도 분위수 기반 혼잡도 4분류** + **(2) 속도(km/h) 회귀** 를 동시에 학습하는 **GINE(PyTorch Geometric) 멀티태스크 베이스라인**입니다.

> 핵심 포인트: mini-batch 학습(DataLoader), 좌표 스냅(Grid Snap), kNN 보조 엣지, DropEdge, Train-only 표준화, Soft Label, Prior-bias, 회귀 힌트, Temperature scaling

---

## Pipeline (Mermaid)

```mermaid
flowchart TD
  A["Input JSON files\ndata/YYYY-MM-DD_HH.json"] --> B["Parse geometry + properties\n(speed, distance, time, roadType)"]
  B --> C["Map hour(HH) to 6 time-slots"]

  subgraph G["Graph Construction"]
    D1["Extract endpoints\n(LineString first/last point)"] --> D2["Grid Snap\n(merge nearby nodes, e.g., 8m)"]
    D2 --> D3["Build real edges\n(road connectivity)"]
    D3 --> D4["Add kNN auxiliary edges\n(k=3, weight=0.25)"]
    D4 --> D5["Assemble PyG Data\nx, edge_index, edge_attr, y(speed), slot"]
  end

  C --> D1
  D5 --> E["Dataset Split\nTrain / Val(optional) / Test(last 4 files)"]

  subgraph N["Train-only Statistics"]
    F1["Compute normalization stats\n(mu/sd for node, edge, speed)"] --> F2["Compute quantile thresholds per slot\n(q25, q50, q75)"]
    F2 --> F3["Save artifacts\ntrain_stats_and_thresholds.npz"]
  end

  E --> F1

  subgraph M["Model & Training"]
    H1["GINE Encoder"] --> H2["Heads\n4-class logits + speed regression"]
    H3["DropEdge (train)"] --> H4["Loss\nSoft-Label CE + Focal + Reg MSE"]
    H2 --> H4
    H4 --> H5["Optimize\n(epochs, mini-batch)"]
    H5 --> H6["Save\nbest_model.pt"]
  end

  F3 --> H1
  E --> H1

  subgraph Cb["Calibration & Inference"]
    I1["Prior-bias fit\n(val if exists)"] --> I2["Regression-hint to logits"]
    I2 --> I3["Temperature scaling"]
    I3 --> I4["Test evaluation\nACC, Macro-F1, Confusion Matrix, Report"]
  end

  H6 --> I1
````

---

## What this repo does

### Task

* **Classification (4 classes)**: 각 시간대(slot)별 학습 데이터 속도 분포의 **25/50/75 분위수(q25/q50/q75)** 를 기준으로 4단계 클래스로 변환
* **Regression**: 속도(km/h) 예측(연속값)

### Graph construction

* 입력 JSON의 도로 구간(geometry)에서 **양 끝점을 노드 후보**로 만들고,
* **Grid Snap(기본 8m)** 으로 근접 노드를 동일 노드로 합쳐 “끊긴 연결”을 복원
* 실제 연결(real edge) + **kNN 기반 보조 엣지**를 추가하여 연결성을 보강

---

## Requirements

* Python **3.10+** 권장
* 주요 패키지:

  * `torch`
  * `torch-geometric`
  * `scikit-learn`
  * `numpy`

> `torch-geometric`는 PyTorch/CUDA 조합에 따라 설치 명령이 달라질 수 있습니다.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` 예시:

```txt
numpy
scikit-learn
torch
torch-geometric
```

---

## Data format

### File naming convention (필수)

데이터 디렉터리(`data/`)에 아래 형식의 JSON 파일을 넣습니다.

* `YYYY-MM-DD_HH.json`
  예: `2025-08-01_07.json` (HH는 0~23)

코드는 파일명에서 **hour(시각)** 를 파싱하여 6개 시간대(slot)로 매핑합니다.

### JSON structure (권장: GeoJSON 유사)

최상단이 dict이면 `features`를, list이면 그대로 순회합니다.

* `features[*].geometry.coordinates`

  * Point 형태면 `[x, y]`
  * LineString 형태면 `[[x1, y1], ..., [xN, yN]]` (첫점/끝점 사용)
* `features[*].properties`

  * `speed` (optional) : km/h
  * `distance` (optional) : (단위 일관 필요)
  * `time` (optional) : (단위 일관 필요)
  * `roadType` (optional) : 정수 권장 (one-hot 10차원으로 변환)

> 좌표는 유클리드 거리로 계산하므로, **가능하면 평면 좌표계(미터 단위)** 사용을 권장합니다.

---

## Quickstart

### 1) 기본 실행

```bash
python gine_v7.py --data_dir data --device auto
```

### 2) 주요 옵션

```bash
python gine_v7.py \
  --data_dir data \
  --epochs 300 \
  --batch_size 2 \
  --k 3 \
  --knn_weight 0.25 \
  --snap_grid 8.0 \
  --device auto
```

---

## Outputs

실행 후 생성 파일(학습 산출물):

* `best_model.pt` : 최적 모델 가중치
* `train_stats_and_thresholds.npz` :

  * train-only 표준화 통계(mu/sd)
  * 시간대(slot)별 분위수 임계값(q25/q50/q75)

테스트:

* 마지막 **4개 JSON을 테스트셋**으로 사용하며,
* 파일별로 ACC / Confusion Matrix / classification report / Macro-F1를 출력합니다.

> 현재 운영 방식: `.gitignore`에서 `*.pt`, `*.npz`를 무시하도록 설정하면, 레포에는 코드/데이터/문서만 올라가고 모델 산출물은 제외됩니다.

---

## Key techniques implemented

* **Mini-batch training** (PyG DataLoader)
* **Grid Snap** (`snap_grid` meters)
* **kNN auxiliary edges** + weighted edge attributes (`knn_weight`)
* **DropEdge** (학습 시 랜덤 엣지 드롭)
* **Train-only normalization** (node/edge/speed)
* **Soft Label CE** + **Focal Loss**
* **Prior-bias calibration** (검증 세트 기반 logit_bias)
* **Regression hint to logits** + **Temperature scaling**

---

## Repository structure

```txt
.
├─ gine_v7.py
├─ data/
│  ├─ YYYY-MM-DD_HH.json
│  ├─ ...
├─ requirements.txt
└─ README.md
```

---

## Roadmap (optional)

* [ ] 샘플 JSON / 스키마 문서 추가
* [ ] 결과 리포트 자동 저장(csv/json)
* [ ] 실험 설정(config) YAML화
* [ ] 도로 신설 시뮬레이터 파이프라인과 통합(후처리/NSGA-II 등)

---

## License

연구/학습 목적이라면 MIT 권장 (상황에 맞게 선택)

```
```
