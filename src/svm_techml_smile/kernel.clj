(ns svm-techml-smile.kernel
  "RBF kernel estimation, fitting, and std-scaling helpers for Smile SVM/SVR."
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.math :as dsm]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as dfn]
            [svm-techml-smile.bridge :refer [dataset->array]])
  (:import (smile.math.kernel GaussianKernel)
           (smile.classification SVM SVM$Options)
           (smile.model.svm SVR)
           (smile.regression KernelMachine Regression)))

(set! *warn-on-reflection* true)

;;; Standard scaling

(defn fit-scale
  "Fit a standard scaler on ds. Returns an opaque scaler map (tech.ml.dataset.math format)."
  [ds]
  (dsm/fit-std-scale ds))

(defn transform-scale
  "Apply a previously fitted scaler to ds."
  [ds scaler]
  (dsm/transform-std-scale ds scaler))

;;; Sigma (RBF bandwidth) estimation

(defn estimate-sigma
  "Estimate the RBF Gaussian kernel bandwidth from a scaled feature matrix.

  Follows the sklearn 'scale' convention:
    sigma = sqrt(0.5 * n_features * variance(flatten(X)))
  i.e. gamma = 1 / (n_features * X.var()). Note `variance` here is the sample
  variance (n-1 denominator); sklearn's X.var() uses n, so sigma runs a factor
  sqrt(n/(n-1)) high — under 0.01% for any matrix over 5k values.

  Missing values are skipped, not propagated — bridge/dataset->array is what
  rejects them. Throws if sigma is not positive, since a zero-variance feature
  matrix would otherwise build a degenerate GaussianKernel."
  [scaled-ds]
  (let [sigma (Math/sqrt (* 0.5
                            (ds/column-count scaled-ds)
                            (dfn/variance
                             (dtype/concat-buffers
                              (map #(ds/column scaled-ds %) (ds/column-names scaled-ds))))))]
    (when-not (pos? sigma)
      (throw (ex-info "RBF sigma must be positive — feature matrix has zero variance or missing/NaN values"
                      {:sigma sigma :columns (ds/column-names scaled-ds)})))
    sigma))

;;; SVM (classifier, labels {-1, +1})

(defn fit-RBF-SVM
  "Train a Gaussian-kernel SVM classifier via Smile's SVM/fit.
   x — double[][], y — int[], sigma — RBF bandwidth, tol — tolerance, C — regularisation.
   SVM$Options(C tol epochs=2) preserves 2-epoch convergence of the old LASVM-based API."
  [^"[[D" x ^"[I" y sigma tol C]
  (SVM/fit x y (GaussianKernel. sigma) (SVM$Options. C tol 2)))

;;; SVR (regressor, continuous labels)

(defn fit-RBF-SVR
  "Train a Gaussian-kernel SVR via Smile's SVR.fit.
   x — double[][], y — double[], sigma — RBF bandwidth, eps — epsilon tube, C — regularisation.
   tol=0.01 is the Smile SVR default; irrelevant for the RBF kernel but required by the API."
  [^"[[D" x ^"[D" y sigma eps C]
  (.fit (SVR. (GaussianKernel. sigma) eps C 0.01) x y))

;;; Prediction + scoring
;;
;; These exist so callers never have to import or type-hint a Smile class.
;; Both are hinted to smile.regression.Regression, the interface KernelMachine
;; implements, so an unhinted caller pays no reflection cost.

(defn predict
  "Predict one observation. features is a double[] in training column order.
  Reflection-free replacement for (.predict ^KernelMachine km ^doubles row)."
  ^double [^Regression model features]
  (.predict model ^Object features))

(defn predict-all
  "Predict a whole feature matrix. features is a double[][] (see
  bridge/dataset->array); returns a double[] of predictions."
  ^"[D" [^Regression model features]
  (.predict model ^objects features))

(defn rsq
  "In-sample R^2 against an origin baseline: 1 - sum((y - yhat)^2) / sum(y^2).

  Note this is the UNCENTERED R^2 — the denominator is sum(y^2), not
  sum((y - ybar)^2), so it measures fit against zero rather than against the
  mean of y. That is what the jasmine quant-screen models have always reported;
  it reads higher than the textbook R^2 whenever y has a non-zero mean.

  sum((y - yhat)^2) is exactly what smile.validation.metric.RSS/of computes, so
  this needs no Smile call at all."
  ^double [y predictions]
  (- 1.0 (/ (dfn/distance-squared y predictions)
            (dfn/distance-squared y (repeat (count y) 0.0)))))

;;; Convenience: scale + extract arrays in one call

(defn prepare-features
  "Fit a standard scaler on x-ds, scale it, and return a map with:
    :scaler         — fitted scaler (pass to transform-scale for new data)
    :sigma          — estimated RBF bandwidth
    :features-array — double[][] ready for Smile fit/predict"
  [x-ds]
  (let [scaler (fit-scale x-ds)
        scaled (transform-scale x-ds scaler)]
    {:scaler         scaler
     :sigma          (estimate-sigma scaled)
     :features-array (dataset->array scaled)}))

(defn scale-new-features
  "Scale a new feature dataset with a previously fitted scaler and return double[][]."
  [x-ds scaler]
  (dataset->array (transform-scale x-ds scaler)))