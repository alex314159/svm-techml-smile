(ns svm-techml-smile.classification
  "Time-series SVM classification helpers: label conversion, evaluation metrics,
  and a generic model spec constructor (clibtrader-style).

  A time-series classification model spec is a map with:
    :dataset   — tech.ml.dataset (rows = trading days, columns = OHLCV + derived)
    :seed      — long, for deterministic train/test shuffle
    :start-idx — skip this many rows at the start (warm-up period for indicators)
    :last-idx  — reserve this many rows at the end for out-of-sample / live prediction
    :training  — fraction of middle rows used for training (rest = test)
    :id-name   — column keyword for row identifiers (e.g. :date)
    :signal    — {:column <kw> :map-fn <fn> :arg-cols [<kw> ...]}  target label spec
    :transform — seq of feature specs applied to build derived columns
    :features  — seq of feature specs (with :lag) used as the feature matrix"
  (:require [tech.v3.dataset :as ds])
  (:import (smile.classification SVM)
           (smile.validation.metric Accuracy Precision Sensitivity Fallout ConfusionMatrix)))

(set! *warn-on-reflection* true)

;;; Label conversion

(defn labels->01
  "Convert SVM {-1, +1} integer labels to {0, 1} for Smile classification metrics.
  Smile's Accuracy/Precision/Sensitivity/Fallout/ConfusionMatrix expect 0-based class
  indices — passing -1 causes ArrayIndexOutOfBoundsException."
  [coll]
  (int-array (map #(if (pos? %) 1 0) coll)))

;;; Generic evaluation

(defn evaluate-classifier
  "Run a trained Smile SVM on features-array and return predictions + metrics.

  kernel-machine — smile.classification.SVM (or compatible .predict implementor)
  features-array — double[][]
  y-array        — int[] of true labels in {-1, +1}

  Returns a map with :predictions (seq), :y-array, :accuracy, :precision,
  :sensitivity, :fallout, :confusion-matrix."
  [^SVM kernel-machine ^"[[D" features-array ^"[I" y-array]
  (let [raw-predictions (.predict kernel-machine features-array)
        yarray01        (labels->01 y-array)
        rawpred01       (labels->01 raw-predictions)]
    {:predictions      (seq raw-predictions)
     :y-array          y-array
     :accuracy         (Accuracy/of yarray01 rawpred01)
     :precision        (Precision/of yarray01 rawpred01)
     :sensitivity      (Sensitivity/of yarray01 rawpred01)
     :fallout          (Fallout/of yarray01 rawpred01)
     :confusion-matrix (ConfusionMatrix/of yarray01 rawpred01)}))

;;; Model spec constructor

(defn model-spec
  "Build a base time-series SVM classification spec.

  Required args:
    dataset  — tech.ml.dataset (one row per time step)

  Optional kwargs (all have defaults):
    :seed       — random seed for deterministic shuffle (default 42)
    :start-idx  — warm-up rows to skip at the start (default 100)
    :last-idx   — rows reserved at the end for live prediction (default 250)
    :training   — train fraction of middle rows (default 0.7)
    :id-name    — row-ID column keyword (default :date)

  Callers must assoc :signal, :transform, and :features before passing to the
  processing pipeline (process-model / split-data / svm-prepare-data in the
  host project)."
  [dataset & {:keys [seed start-idx last-idx training id-name]
              :or   {seed      42
                     start-idx 100
                     last-idx  250
                     training  0.7
                     id-name   :date}}]
  {:dataset   dataset
   :seed      seed
   :start-idx start-idx
   :last-idx  last-idx
   :training  training
   :id-name   id-name})