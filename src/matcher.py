"""LightGBM-based entity matching model."""
import os
import numpy as np
import lightgbm as lgb
from . import config


class EntityMatcher:
    """Binary classifier: does a (S1, S2/S3) pair refer to the same entity?"""

    def __init__(self):
        self.model = None
        self.threshold = config.DEFAULT_THRESHOLD

    # ─── Training ─────────────────────────────────────────────────────────

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM with optional early stopping on a validation set."""
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        scale = n_neg / n_pos if n_pos else 1.0
        print(f"    Positives: {n_pos:,}  Negatives: {n_neg:,}  scale_pos_weight: {scale:.1f}")

        params = config.LGBM_PARAMS.copy()
        params['scale_pos_weight'] = scale

        dtrain = lgb.Dataset(X_train, label=y_train)

        callbacks = [lgb.log_evaluation(50)]
        valid_sets = [dtrain]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('val')
            callbacks.append(lgb.early_stopping(config.LGBM_EARLY_STOPPING))

        self.model = lgb.train(
            params, dtrain,
            num_boost_round=config.LGBM_NUM_ROUNDS,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        print(f"    Best iteration: {self.model.best_iteration}")

        # Feature importance
        from .features import FEATURE_COLUMNS
        importance = self.model.feature_importance(importance_type='gain')
        ranked = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1])
        print("    Feature importance (gain):")
        for name, imp in ranked[:10]:
            print(f"      {name:30s} {imp:,.0f}")

    # ─── Prediction ───────────────────────────────────────────────────────

    def predict(self, X):
        """Return match probabilities for feature matrix X."""
        if self.model is None:
            raise RuntimeError("Model not trained / loaded.")
        return self.model.predict(X)

    # ─── Threshold tuning ─────────────────────────────────────────────────

    def tune_threshold(self, X_val, y_val, pair_s1_ids, pair_s23_ids, val_gt):
        """Sweep thresholds to maximise macro-averaged F_0.5 on validation pairs.

        Args:
            X_val:         feature matrix for validation pairs
            y_val:         labels (unused, kept for API symmetry)
            pair_s1_ids:   S1 ID for each row in X_val
            pair_s23_ids:  S23 ID for each row in X_val
            val_gt:        {s1_id: set(true_ids)} ground truth for val split
        """
        from .evaluate import evaluate_predictions

        probs = self.predict(X_val)

        best_thr, best_f05 = 0.5, 0.0
        for thr in np.arange(0.10, 0.96, 0.05):
            preds = {}
            for i, (s1, s23) in enumerate(zip(pair_s1_ids, pair_s23_ids)):
                if probs[i] >= thr:
                    preds.setdefault(s1, set()).add(s23)
            # Fill singletons
            for s1_id in val_gt:
                preds.setdefault(s1_id, set())

            f05 = evaluate_predictions(preds, val_gt)
            tag = " ◀ best" if f05 > best_f05 else ""
            print(f"      threshold {thr:.2f}  →  F_0.5 = {f05:.4f}{tag}")
            if f05 > best_f05:
                best_f05, best_thr = f05, thr

        self.threshold = best_thr
        print(f"\n    Selected threshold: {best_thr:.2f}  (F_0.5 = {best_f05:.4f})")

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, model_path=None, threshold_path=None):
        model_path = model_path or config.MODEL_PATH
        threshold_path = threshold_path or config.THRESHOLD_PATH
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        with open(threshold_path, 'w') as f:
            f.write(str(self.threshold))
        print(f"    Model     → {model_path}")
        print(f"    Threshold → {threshold_path}")

    def load(self, model_path=None, threshold_path=None):
        model_path = model_path or config.MODEL_PATH
        threshold_path = threshold_path or config.THRESHOLD_PATH
        self.model = lgb.Booster(model_file=model_path)
        with open(threshold_path) as f:
            self.threshold = float(f.read().strip())
        print(f"    Model loaded (threshold={self.threshold:.2f})")
