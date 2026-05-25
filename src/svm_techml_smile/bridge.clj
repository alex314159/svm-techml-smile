(ns svm-techml-smile.bridge
  "Conversions between tech.ml.dataset columns/datasets and Smile data structures."
  (:require [tech.v3.dataset :as ds])
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

(defn dataset->array
  "Convert a tech.ml.dataset to a double[][] (one row per observation, one column per feature).
  Alternative to (dataset->dataframe ds) + .toArray when you want to skip the Smile DataFrame."
  [ds]
  (->> (ds/column-names ds)
       (map #(double-array (ds/column ds %)))
       (into-array)))

(defn cast-to-float
  "Cast every column in ds to :float32. Useful before feeding into Smile SVM/SVR."
  [ds]
  (reduce #(ds/column-cast %1 %2 :float32) ds (ds/column-names ds)))