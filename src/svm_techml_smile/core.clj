(ns svm-techml-smile.core
  "Public API — re-exports all sub-namespaces for single-ns consumers."
  (:require [svm-techml-smile.bridge :as bridge]
            [svm-techml-smile.features :as features]
            [svm-techml-smile.kernel :as kernel]
            [svm-techml-smile.preprocessing :as pre]
            [svm-techml-smile.classification :as cls]))

;;; bridge
(def dscol->smilecol     bridge/dscol->smilecol)
(def dataset->dataframe  bridge/dataset->dataframe)
(def dataset->array      bridge/dataset->array)
(def cast-to-float       bridge/cast-to-float)

;;; features
(def build-column        features/build-column)
(def build-many-columns  features/build-many-columns)

;;; kernel
(def fit-scale           kernel/fit-scale)
(def transform-scale     kernel/transform-scale)
(def estimate-sigma      kernel/estimate-sigma)
(def fit-RBF-SVM         kernel/fit-RBF-SVM)
(def fit-RBF-SVR         kernel/fit-RBF-SVR)
(def prepare-features    kernel/prepare-features)
(def scale-new-features  kernel/scale-new-features)

;;; preprocessing (cross-sectional SVR)
(def svr-model-spec      pre/model-spec)
(def one-hot-reducer     pre/one-hot-reducer)
(def prepare-svr-data    pre/prepare-svr-data)

;;; classification (time-series SVM)
(def ts-model-spec       cls/model-spec)
(def labels->01          cls/labels->01)
(def evaluate-classifier cls/evaluate-classifier)