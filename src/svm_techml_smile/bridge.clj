(ns svm-techml-smile.bridge
  "Conversions between tech.ml.dataset columns/datasets and Smile data structures."
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype :as dtype])
  (:import (smile.data DataFrame)
           (smile.data.vector DoubleVector)))

(set! *warn-on-reflection* true)

(defn dscol->smilecol
  "Convert a tech.ml.dataset column to a Smile DoubleVector, preserving the column name."
  [col]
  (let [n            (:name (meta col))
        ^String sname (if (string? n) n (name n))]
    (DoubleVector. sname ^"[D" (double-array col))))

(defn dataset->dataframe
  "Convert a tech.ml.dataset to a Smile DataFrame (one DoubleVector per column)."
  [ds]
  (DataFrame. (into-array DoubleVector (map dscol->smilecol (vals ds)))))

(defn- assert-no-missing
  "Smile has no missing-value representation: NaNs propagate through the kernel and
  yield a silently useless model. Fail here instead, on the dataset's missing-value
  metadata (no scan)."
  [ds]
  (when (pos? (dtype/ecount (ds/missing ds)))
    (throw (ex-info "Cannot build a Smile feature matrix from a dataset with missing values — drop or impute those rows first"
                    {:missing-by-column (into {} (for [c (ds/column-names ds)
                                                       :let [n (dtype/ecount (ds/missing (ds/column ds c)))]
                                                       :when (pos? n)]
                                                   [c n]))}))))

(defn dataset->array
  "Convert a tech.ml.dataset to a row-major double[][]: one row per observation,
  one column per feature, in (ds/column-names ds) order — the layout Smile's
  fit/predict expect. Same result as (.toArray (dataset->dataframe ds)) but
  without building the intermediate DataFrame.

  Throws if any column has missing values (see assert-no-missing)."
  ^"[[D" [ds]
  (assert-no-missing ds)
  (let [names (ds/column-names ds)
        ncol  (count names)
        nrow  (ds/row-count ds)
        cols  (into-array (map #(dtype/->double-array (ds/column ds %)) names))
        out   (make-array Double/TYPE nrow ncol)]
    (dotimes [i nrow]
      (let [^doubles row (aget ^objects out i)]
        (dotimes [j ncol]
          (aset row j (aget ^doubles (aget ^objects cols j) i)))))
    out))

(defn cast-to-float
  "Cast every column in ds to :float32. Useful before feeding into Smile SVM/SVR."
  [ds]
  (reduce #(ds/column-cast %1 %2 :float32) ds (ds/column-names ds)))