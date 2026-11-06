"""
KNN on CIFAR-10
- KNN 직접 구현 (NumPy / SciPy 거리 계산만 사용, sklearn 분류기 사용 안 함)
- 다양한 k 값 x 거리 함수(L1, L2, Cosine) 실험
- Stratified 5-fold cross-validation
- Accuracy / Precision / Recall / F1 (macro) 직접 계산
- 최적 (k, metric)으로 test set 최종 평가 + 혼동행렬
- 결과: results/ 폴더에 CSV, 그래프, report.md 생성

사용법:
    python knn_cifar10.py --data-dir ./cifar-10-batches-py
    python knn_cifar10.py --data-dir ./cifar-10-batches-py --n-train 10000 --n-test 2000
"""
import argparse
import os
import pickle
import time

import numpy as np
from scipy.spatial.distance import cdist

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10


# ----------------------------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------------------------
def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].astype(np.float32) / 255.0          # (N, 3072)
    y = np.array(d[b"labels"], dtype=np.int64)
    return X, y


def load_cifar10(data_dir):
    Xs, ys = [], []
    for i in range(1, 6):
        p = os.path.join(data_dir, f"data_batch_{i}")
        if os.path.exists(p):
            X, y = _load_batch(p)
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise FileNotFoundError(f"data_batch_* 파일이 {data_dir}에 없습니다.")
    X_train, y_train = np.concatenate(Xs), np.concatenate(ys)
    X_test, y_test = _load_batch(os.path.join(data_dir, "test_batch"))
    return X_train, y_train, X_test, y_test


def stratified_subsample(X, y, n, rng):
    """클래스 비율을 유지하며 n개 샘플링"""
    if n is None or n >= len(y):
        return X, y
    per_class = n // NUM_CLASSES
    idx = np.concatenate([rng.choice(np.where(y == c)[0], per_class, replace=False)
                          for c in range(NUM_CLASSES)])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ----------------------------------------------------------------------------
# 2. KNN 구현
# ----------------------------------------------------------------------------
def pairwise_distance(A, B, metric):
    """A: (m, D) 질의, B: (n, D) 학습 데이터 -> (m, n) 거리 행렬"""
    if metric == "L2":
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b  (벡터화)
        a2 = np.sum(A ** 2, axis=1, keepdims=True)
        b2 = np.sum(B ** 2, axis=1)
        d2 = np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)
        return np.sqrt(d2)
    if metric == "L1":
        return cdist(A, B, metric="cityblock").astype(np.float32)
    if metric == "cosine":
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return 1.0 - An @ Bn.T
    raise ValueError(metric)


class KNNClassifier:
    def __init__(self, k=5, metric="L2"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        # KNN은 lazy learner: 학습 데이터를 저장만 함
        self.X_train = X
        self.y_train = y
        return self

    def neighbors(self, X, max_k, batch=500):
        """각 질의점에 대해 가까운 순서대로 max_k개 이웃 레이블 반환"""
        out = np.empty((len(X), max_k), dtype=np.int64)
        for s in range(0, len(X), batch):
            D = pairwise_distance(X[s:s + batch], self.X_train, self.metric)
            part = np.argpartition(D, max_k - 1, axis=1)[:, :max_k]
            order = np.argsort(np.take_along_axis(D, part, axis=1), axis=1)
            nn = np.take_along_axis(part, order, axis=1)
            out[s:s + batch] = self.y_train[nn]
        return out

    @staticmethod
    def vote(neighbor_labels, k):
        """다수결 투표. 동점이면 더 가까운 이웃이 있는 클래스 선택
        (순위 r의 표에 1 - r*1e-6 가중치)"""
        lab = neighbor_labels[:, :k]
        w = 1.0 - np.arange(k) * 1e-6
        scores = np.zeros((len(lab), NUM_CLASSES))
        for r in range(k):
            scores[np.arange(len(lab)), lab[:, r]] += w[r]
        return scores.argmax(axis=1)

    def predict(self, X):
        return self.vote(self.neighbors(X, self.k), self.k)


# ----------------------------------------------------------------------------
# 3. 평가 지표 (직접 구현)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0)
    true_pos = cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": tp.sum() / cm.sum(),
        "precision": precision.mean(),   # macro average
        "recall": recall.mean(),
        "f1": f1.mean(),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "cm": cm,
    }


# ----------------------------------------------------------------------------
# 4. 5-fold cross-validation
# ----------------------------------------------------------------------------
def stratified_kfold(y, n_folds, rng):
    folds = [[] for _ in range(n_folds)]
    for c in range(NUM_CLASSES):
        idx = rng.permutation(np.where(y == c)[0])
        for f, part in enumerate(np.array_split(idx, n_folds)):
            folds[f].extend(part.tolist())
    return [np.array(f) for f in folds]


def cross_validate(X, y, ks, metrics, n_folds, rng):
    """fold마다 거리/정렬은 한 번만 계산하고 모든 k를 그 결과로 평가"""
    folds = stratified_kfold(y, n_folds, rng)
    max_k = max(ks)
    rows = []  # (metric, k, fold, acc, prec, rec, f1)
    times = {}
    for metric in metrics:
        t0 = time.time()
        for f in range(n_folds):
            val_idx = folds[f]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
            knn = KNNClassifier(metric=metric).fit(X[tr_idx], y[tr_idx])
            nl = knn.neighbors(X[val_idx], max_k)
            for k in ks:
                m = evaluate(y[val_idx], KNNClassifier.vote(nl, k))
                rows.append((metric, k, f, m["accuracy"], m["precision"], m["recall"], m["f1"]))
            print(f"  [{metric}] fold {f + 1}/{n_folds} 완료 ({time.time() - t0:.1f}s)")
        times[metric] = time.time() - t0
    return rows, times


# ----------------------------------------------------------------------------
# 5. 결과 정리 / 시각화 / 보고서
# ----------------------------------------------------------------------------
def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for metric, k, f, a, p, r, f1 in rows:
        g[(metric, k)].append((a, p, r, f1))
    summ = {}
    for key, v in g.items():
        v = np.array(v)
        summ[key] = {"mean": v.mean(axis=0), "std": v.std(axis=0)}
    return summ


def save_csv(rows, summ, out):
    with open(os.path.join(out, "cv_folds.csv"), "w") as f:
        f.write("metric,k,fold,accuracy,precision,recall,f1\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2] + 1},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")
    with open(os.path.join(out, "cv_summary.csv"), "w") as f:
        f.write("metric,k,acc_mean,acc_std,precision_mean,precision_std,"
                "recall_mean,recall_std,f1_mean,f1_std\n")
        for (metric, k), s in sorted(summ.items()):
            m, sd = s["mean"], s["std"]
            f.write(f"{metric},{k}," + ",".join(f"{m[i]:.4f},{sd[i]:.4f}" for i in range(4)) + "\n")


def plot_results(summ, metrics, ks, test_res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, ax in enumerate(axes.ravel()):
        for metric in metrics:
            mean = [summ[(metric, k)]["mean"][i] for k in ks]
            std = [summ[(metric, k)]["std"][i] for k in ks]
            ax.errorbar(ks, mean, yerr=std, marker="o", capsize=3, label=metric)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k")
        ax.set_title(names[i])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("KNN on CIFAR-10: 5-fold CV (mean ± std)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cv_metrics_vs_k.png"), dpi=150)
    plt.close(fig)

    cm = test_res["cm"]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "test_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def write_report(args, summ, metrics, ks, times, best, test_res, out):
    L = []
    L.append("# KNN on CIFAR-10 — 실험 결과 보고서\n")
    L.append("## 1. 실험 설정\n")
    L.append(f"- 데이터: CIFAR-10 (32×32×3 → 3072차원 벡터, 픽셀값 /255 정규화)")
    L.append(f"- 교차검증용 학습 데이터: {args.n_train}장 (클래스당 {args.n_train // 10}장, 층화 추출)")
    L.append(f"- 최종 테스트 데이터: {args.n_test}장 (test_batch에서 층화 추출)")
    L.append(f"- k 후보: {', '.join(map(str, ks))}")
    L.append(f"- 거리 함수: {', '.join(metrics)}")
    L.append("  - L1 (Manhattan): Σ|x_i − y_i|")
    L.append("  - L2 (Euclidean): √Σ(x_i − y_i)²")
    L.append("  - Cosine: 1 − (x·y)/(‖x‖‖y‖)")
    L.append("- 검증: Stratified 5-fold cross-validation (각 fold의 클래스 비율 동일)")
    L.append("- 지표: Accuracy, Precision/Recall/F1 (10개 클래스 macro 평균)")
    L.append("- 동점 처리: 득표가 같으면 더 가까운 이웃이 속한 클래스 선택\n")

    L.append("## 2. 5-fold CV 결과 (평균 ± 표준편차)\n")
    for metric in metrics:
        L.append(f"### {metric}\n")
        L.append("| k | Accuracy | Precision | Recall | F1 |")
        L.append("|---:|---|---|---|---|")
        for k in ks:
            m, s = summ[(metric, k)]["mean"], summ[(metric, k)]["std"]
            L.append(f"| {k} | " + " | ".join(f"{m[i]:.4f} ± {s[i]:.4f}" for i in range(4)) + " |")
        L.append("")

    L.append("### 거리 함수별 최고 성능\n")
    L.append("| 거리 함수 | 최적 k | Accuracy | F1 | CV 소요 시간 |")
    L.append("|---|---:|---|---|---:|")
    best_per = {}
    for metric in metrics:
        k_b = max(ks, key=lambda k: summ[(metric, k)]["mean"][0])
        m = summ[(metric, k_b)]["mean"]
        best_per[metric] = (k_b, m)
        L.append(f"| {metric} | {k_b} | {m[0]:.4f} | {m[3]:.4f} | {times[metric]:.1f}s |")
    L.append("\n![CV metrics vs k](cv_metrics_vs_k.png)\n")

    L.append("## 3. 최종 테스트 결과\n")
    L.append(f"CV 정확도가 가장 높았던 **{best[0]}, k = {best[1]}** 설정으로 "
             f"학습 데이터 {args.n_train}장 전체를 사용해 test set {args.n_test}장을 평가했다.\n")
    L.append("| Accuracy | Precision | Recall | F1 |")
    L.append("|---|---|---|---|")
    L.append(f"| {test_res['accuracy']:.4f} | {test_res['precision']:.4f} | "
             f"{test_res['recall']:.4f} | {test_res['f1']:.4f} |\n")
    L.append("| 클래스 | Precision | Recall | F1 |")
    L.append("|---|---|---|---|")
    for c in range(10):
        L.append(f"| {CLASSES[c]} | {test_res['per_class_precision'][c]:.3f} | "
                 f"{test_res['per_class_recall'][c]:.3f} | {test_res['per_class_f1'][c]:.3f} |")
    L.append("\n![Confusion matrix](test_confusion_matrix.png)\n")

    # ---- 결과 기반 자동 분석 ----
    L.append("## 4. 비교 및 분석\n")
    order = sorted(metrics, key=lambda mt: -best_per[mt][1][0])
    gap = best_per[order[0]][1][0] - best_per[order[-1]][1][0]
    L.append(f"**거리 함수 비교.** 최고 정확도 기준 순위는 "
             + " > ".join(f"{mt}({best_per[mt][1][0]:.3f})" for mt in order)
             + f"이며, 1위와 최하위의 차이는 {gap * 100:.1f}%p이다. "
             "L1은 픽셀 차이를 선형으로 더하기 때문에, 일부 픽셀에서 큰 차이가 나는 경우"
             "(배경 변화, 위치 이동 등)에 L2보다 덜 민감하다. L2는 차이를 제곱하므로 소수의 큰 "
             "픽셀 차이가 거리를 지배한다. Cosine은 벡터의 크기(전체 밝기)를 무시하고 방향(패턴)만 "
             "비교하므로 조명·밝기 차이에 강하다.\n")

    for metric in metrics:
        accs = [summ[(metric, k)]["mean"][0] for k in ks]
        k_b = best_per[metric][0]
        L.append(f"- {metric}: k=1에서 {accs[0]:.3f}, 최적 k={k_b}에서 {max(accs):.3f}, "
                 f"k={ks[-1]}에서 {accs[-1]:.3f}")
    L.append("")
    L.append("**k 값의 영향.** k가 너무 작으면(k=1) 가장 가까운 한 장의 이미지에만 의존해 "
             "노이즈에 민감하고(분산↑, 과적합), k가 너무 크면 멀리 있는 다른 클래스 이미지까지 "
             "투표에 들어와 결정 경계가 지나치게 매끄러워진다(편향↑, 과소적합). 따라서 정확도는 "
             "중간 k에서 최대가 되는 형태를 보인다. 표준편차가 작아 fold 간 결과는 안정적이다.\n")

    pr = test_res["per_class_recall"]
    pp = test_res["per_class_precision"]
    hi = np.argsort(-pr)[:2]
    lo = np.argsort(pr)[:2]
    L.append(f"**클래스별 성능.** Recall이 높은 클래스는 {CLASSES[hi[0]]}({pr[hi[0]]:.2f}), "
             f"{CLASSES[hi[1]]}({pr[hi[1]]:.2f})이고 낮은 클래스는 {CLASSES[lo[0]]}({pr[lo[0]]:.2f}), "
             f"{CLASSES[lo[1]]}({pr[lo[1]]:.2f})이다. Precision과 Recall이 클래스마다 크게 다르기 "
             f"때문에(Precision 범위 {pp.min():.2f}~{pp.max():.2f}) macro F1이 accuracy보다 "
             "모델의 불균형한 성능을 더 잘 드러낸다. 픽셀 거리 기반 KNN은 물체의 '의미'보다 "
             "배경 색·전체 색 분포가 비슷한 이미지를 이웃으로 찾는 경향이 있어, 단색 배경"
             "(하늘·바다)을 가진 클래스로 예측이 몰리고 배경이 다양한 동물 클래스는 서로 혼동된다.\n")

    L.append("**한계.** KNN은 학습이 없지만 예측 시 모든 학습 샘플과의 거리를 계산해야 하므로 "
             "O(N·D)의 비용이 들고, 3072차원 원시 픽셀 공간에서는 차원의 저주 때문에 거리의 "
             "의미가 약해진다. 성능 향상을 위해서는 PCA·HOG·CNN 특징 등 의미 있는 특징 공간에서 "
             "KNN을 적용하는 것이 효과적이다.\n")

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./cifar-10-batches-py")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--n-train", type=int, default=5000, help="CV/학습에 쓸 샘플 수 (전체: 50000)")
    ap.add_argument("--n-test", type=int, default=1000, help="최종 테스트 샘플 수 (전체: 10000)")
    ap.add_argument("--ks", default="1,3,5,7,10,15,20,50,100")
    ap.add_argument("--metrics", default="L1,L2,cosine")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    metrics = args.metrics.split(",")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("데이터 로드 중...")
    Xtr, ytr, Xte, yte = load_cifar10(args.data_dir)
    Xtr, ytr = stratified_subsample(Xtr, ytr, args.n_train, rng)
    Xte, yte = stratified_subsample(Xte, yte, args.n_test, rng)
    args.n_train, args.n_test = len(ytr), len(yte)
    print(f"train {Xtr.shape}, test {Xte.shape}")

    print(f"{args.folds}-fold CV 시작...")
    rows, times = cross_validate(Xtr, ytr, ks, metrics, args.folds, rng)
    summ = summarize(rows)
    save_csv(rows, summ, args.out_dir)

    best = max(summ, key=lambda key: summ[key]["mean"][0])
    print(f"최적 설정: metric={best[0]}, k={best[1]}, CV acc={summ[best]['mean'][0]:.4f}")

    knn = KNNClassifier(k=best[1], metric=best[0]).fit(Xtr, ytr)
    test_res = evaluate(yte, knn.predict(Xte))
    print(f"Test  acc={test_res['accuracy']:.4f}  P={test_res['precision']:.4f}  "
          f"R={test_res['recall']:.4f}  F1={test_res['f1']:.4f}")

    plot_results(summ, metrics, ks, test_res, args.out_dir)
    write_report(args, summ, metrics, ks, times, best, test_res, args.out_dir)
    print(f"결과 저장: {args.out_dir}/ (report.md, cv_summary.csv, cv_folds.csv, *.png)")


if __name__ == "__main__":
    main()
