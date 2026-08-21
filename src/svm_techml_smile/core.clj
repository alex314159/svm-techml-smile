(ns svm-techml-smile.core
  "Public API — re-exports all sub-namespaces for single-ns consumers.

  Aliases here carry the defining var's :doc, :arglists, :file and :line, so
  (doc svm-techml-smile.core/foo) shows the real documentation and REPL-driven
  editor hover (Calva, CIDER) works. Static-only analysis (clj-kondo, Cursive's
  offline index) does not expand this macro and will show less — require the
  defining namespace directly if that is how your editor resolves symbols."
  (:require [svm-techml-smile.bridge :as bridge]
            [svm-techml-smile.features :as features]
            [svm-techml-smile.kernel :as kernel]
            [svm-techml-smile.preprocessing :as pre]
            [svm-techml-smile.classification :as cls]))

(defmacro ^:private defalias
  "def sym as an alias for another namespace's var, copying the metadata that
  documentation tooling reads. Without this a plain (def a b) alias has no
  :doc and no :arglists, so callers see nothing useful."
  [sym target]
  `(let [target# (var ~target)]
     (alter-meta! (def ~sym @target#)
                  merge
                  (select-keys (meta target#) [:doc :arglists :file :line :column]))
     (var ~sym)))

;;; bridge
(defalias dscol->smilecol     bridge/dscol->smilecol)
(defalias dataset->dataframe  bridge/dataset->dataframe)
(defalias dataset->array      bridge/dataset->array)
(defalias cast-to-float       bridge/cast-to-float)

;;; features
(defalias build-column        features/build-column)
(defalias build-many-columns  features/build-many-columns)

;;; kernel
(defalias fit-scale           kernel/fit-scale)
(defalias transform-scale     kernel/transform-scale)
(defalias estimate-sigma      kernel/estimate-sigma)
(defalias fit-RBF-SVM         kernel/fit-RBF-SVM)
(defalias fit-RBF-SVR         kernel/fit-RBF-SVR)
(defalias prepare-features    kernel/prepare-features)
(defalias scale-new-features  kernel/scale-new-features)
(defalias predict             kernel/predict)
(defalias predict-all         kernel/predict-all)
(defalias rsq                 kernel/rsq)

;;; preprocessing (cross-sectional SVR)
(defalias svr-model-spec      pre/svr-model-spec)
(defalias one-hot-reducer     pre/one-hot-reducer)
(defalias prepare-svr-data    pre/prepare-svr-data)

;;; classification (time-series SVM)
(defalias ts-model-spec       cls/svm-model-spec)
(defalias labels->01          cls/labels->01)
(defalias evaluate-classifier cls/evaluate-classifier)
