(ns svm-techml-smile.features
  "Generic column-building DSL for tech.ml.dataset.

  A feature spec is a map with:
    :column   — keyword naming the output column
    :assoc-fn — (fn [dataset]) → column-values  (for columns derived from the whole dataset)
    OR
    :map-fn   — (fn [& col-vals]) → scalar       (applied element-wise)
    :arg-cols — vector of column keywords fed to :map-fn

  Either :assoc-fn or (:map-fn + :arg-cols) must be present."
  (:require [tech.v3.dataset :as ds]))

(defn build-column
  "Add a single column to dataset according to feature-spec."
  [dataset spec]
  (if (contains? spec :assoc-fn)
    (assoc dataset (:column spec) ((:assoc-fn spec) dataset))
    (ds/column-map dataset (:column spec) (:map-fn spec) nil (:arg-cols spec))))

(defn build-many-columns
  "Reduce over a sequence of feature specs, adding each column in order."
  [dataset specs]
  (reduce build-column dataset specs))