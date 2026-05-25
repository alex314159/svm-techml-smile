(ns svm-techml-smile.kernel
  "RBF kernel estimation, fitting, and std-scaling helpers for Smile SVM/SVR."
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.math :as dsm]
            [tech.v3.datatype.functional :as dfn]
            [svm-techml-smile.bridge :refer [dataset->dataframe]])
  (:import (smile.data DataFrame)
           (smile.math.kernel GaussianKernel)
           (smile.classification SVM SVM$Options)
           (smile.model.svm SVR)
           (smile.regression KernelMachine)))

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

  Uses the sklearn 'scale' convention:
    sigma = sqrt(0.5 * n_features * variance(flatten(X)))
  which corresponds to gamma = 1 / (n_features * X.var()) in sklearn notation."
  [scaled-ds]
  (Math/sqrt (* 0.5
                (ds/column-count scaled-ds)
                (dfn/variance (vec (flatten (ds/value-reader scaled-ds)))))))

;;; SVM (classifier, labels {-1, +1})

(defn fit-RBF-SVM
  "Train a Gaussian-kernel SVM classifier via Smile 5's SVM/fit.
   x — double[][], y — int[], sigma — RBF bandwidth, tol — tolerance, C — regularisation.
   SVM$Options(C tol epochs=2) preserves 2-epoch convergence of the old LASVM-based API."
  [^"[[D" x ^"[I" y sigma tol C]
  (SVM/fit x y (GaussianKernel. sigma) (SVM$Options. C tol 2)))

;;; SVR (regressor, continuous labels)

(defn fit-RBF-SVR
  "Train a Gaussian-kernel SVR via Smile 5's SVR.fit.
   x — double[][], y — double[], sigma — RBF bandwidth, eps — epsilon tube, C — regularisation.
   tol=0.01 is the Smile SVR default; irrelevant for the RBF kernel but required by the API."
  [^"[[D" x ^"[D" y sigma eps C]
  (.fit (SVR. (GaussianKernel. sigma) eps C 0.01) x y))

;;; Convenience: scale + extract arrays in one call

(defn prepare-features
  "Fit a standard scaler on x-ds, scale it, and return a map with:
    :scaler         — fitted scaler (pass to transform-scale for new data)
    :sigma          — estimated RBF bandwidth
    :features-array — double[][] ready for Smile fit/predict"
  [x-ds]
  (let [scaler      (fit-scale x-ds)
        scaled      (transform-scale x-ds scaler)
        sigma       (estimate-sigma scaled)
        feat-array  (.toArray ^DataFrame (dataset->dataframe scaled) (into-array String []))]
    {:scaler         scaler
     :sigma          sigma
     :features-array feat-array}))

(defn scale-new-features
  "Scale a new feature dataset with a previously fitted scaler and return double[][]."
  [x-ds scaler]
  (.toArray ^DataFrame (dataset->dataframe (transform-scale x-ds scaler)) (into-array String [])))