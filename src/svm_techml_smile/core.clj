(ns svm-techml-smile.core
  "Public API — re-exports the three sub-namespaces for convenience."
  (:require [svm-techml-smile.bridge :as bridge]
            [svm-techml-smile.features :as features]
            [svm-techml-smile.kernel :as kernel]))

;;; bridge
(def dscol->smilecol  bridge/dscol->smilecol)
(def dataset->dataframe bridge/dataset->dataframe)
(def dataset->array   bridge/dataset->array)
(def cast-to-float    bridge/cast-to-float)

;;; features
(def build-column      features/build-column)
(def build-many-columns features/build-many-columns)

;;; kernel
(def fit-scale         kernel/fit-scale)
(def transform-scale   kernel/transform-scale)
(def estimate-sigma    kernel/estimate-sigma)
(def fit-RBF-SVM       kernel/fit-RBF-SVM)
(def fit-RBF-SVR       kernel/fit-RBF-SVR)
(def prepare-features  kernel/prepare-features)
(def scale-new-features kernel/scale-new-features)