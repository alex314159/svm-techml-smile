(ns svm-techml-smile.preprocessing
  "Cross-sectional model spec + data preparation pipeline (jasmine-style SVR).

  A cross-sectional model spec is a map with:
    :type            — :svr (SVR regressor) or :ols (OLS — host project must supply OLS training)
    :dataset         — tech.ml.dataset holding all observations
    :id-name         — column keyword used as row identifier (excluded from feature matrix)
    :y               — {:column <kw> :predict-fn <fn>} — target column + inverse transform
    :removals        — seq of column keywords to drop before building the feature matrix
    :transformations — seq of feature specs (build-column DSL) applied first
    :one-hot         — {:columns [<kw> ...] :removals [<kw> ...]} — categorical encoding config
    :std-scale       — {:columns [<kw> ...] :scaler nil} — columns to standardise; scaler is
                        filled in by prepare-svr-data

  After prepare-svr-data, the map gains:
    :std-scale       — with :scaler filled in (use transform-scale for new data)
    :features-cols   — ordered column names of the feature matrix
    :sigma           — estimated RBF bandwidth
    :features-array  — double[][] ready for Smile SVR fit / predict
    :y-array         — double[]  target values (raw, before :predict-fn)"
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.categorical :as categorical]
            [tech.v3.dataset.math :as dsm]
            [svm-techml-smile.bridge :refer [dscol->smilecol dataset->dataframe]]
            [svm-techml-smile.features :refer [build-many-columns]]
            [svm-techml-smile.kernel :refer [estimate-sigma]])
  (:import (smile.data DataFrame)
           (smile.data.vector DoubleVector)))

(set! *warn-on-reflection* true)

;;; Model spec constructor

(defn svr-model-spec
  "Build a base cross-sectional SVR model spec.

  Required args:
    dataset          — tech.ml.dataset
    id-col           — keyword identifying the row-ID column (e.g. :Bond, :ISIN)
    y-col            — keyword naming the target column
    std-scale-cols   — seq of columns to standardise

  Optional kwargs:
    :type            — :svr (default) or :ols
    :predict-fn      — inverse-transform for y (default identity)
    :removals        — extra columns to drop before the feature matrix (default [])
    :transformations — feature-spec DSL steps to apply first (default nil)
    :one-hot-cols    — columns to one-hot encode (default [])
    :one-hot-removals— one-hot columns to drop after encoding (default [])"
  [dataset id-col y-col std-scale-cols
   & {:keys [type predict-fn removals transformations one-hot-cols one-hot-removals]
      :or   {type             :svr
             predict-fn       identity
             removals         []
             transformations  nil
             one-hot-cols     []
             one-hot-removals []}}]
  {:type            type
   :dataset         dataset
   :id-name         id-col
   :y               {:column y-col :predict-fn predict-fn}
   :removals        removals
   :transformations transformations
   :one-hot         {:columns one-hot-cols :removals one-hot-removals}
   :std-scale       {:columns std-scale-cols :scaler nil}})

;;; One-hot encoding

(defn one-hot-reducer
  "One-hot encode each column in cols, then drop any columns listed in removals.
  removals is only needed for OLS models (to avoid perfect collinearity);
  SVR models should pass [] for removals."
  [dataset cols removals]
  (ds/remove-columns
   (reduce (fn [ds col]
             (categorical/transform-one-hot ds (categorical/fit-one-hot ds col)))
           dataset cols)
   removals))

;;; Full SVR data preparation pipeline

(defn prepare-svr-data
  "Apply the model spec pipeline and return the spec augmented with Smile-ready arrays.

  Steps:
    1. apply :transformations (build-many-columns DSL)
    2. fit std-scaler on :std-scale columns
    3. drop :removals columns
    4. one-hot encode :one-hot columns (dropping :one-hot :removals)
    5. apply the std-scaler
    6. split into feature matrix (everything except :id-name and :y column)
    7. compute sigma, features-array (double[][]), y-array (double[])

  The returned spec has :std-scale :scaler filled in — pass it to
  svm-techml-smile.kernel/transform-scale to normalise new observations."
  [spec]
  (let [transformed  (if (seq (:transformations spec))
                       (build-many-columns (:dataset spec) (:transformations spec))
                       (:dataset spec))
        scaler       (dsm/fit-std-scale
                      (ds/select-columns transformed (get-in spec [:std-scale :columns])))
        clean-data   (-> transformed
                         (ds/remove-columns (:removals spec))
                         (one-hot-reducer
                          (get-in spec [:one-hot :columns])
                          (get-in spec [:one-hot :removals]))
                         (dsm/transform-std-scale scaler))
        features-ds  (ds/remove-columns clean-data
                                        [(:id-name spec)
                                         (get-in spec [:y :column])])
        sigma        (estimate-sigma features-ds)
        feat-array   (.toArray ^DataFrame (dataset->dataframe features-ds)
                               (into-array String []))
        y-array      (.toDoubleArray ^DoubleVector
                      (dscol->smilecol
                       (clean-data (get-in spec [:y :column]))))]
    (assoc spec
           :std-scale      (assoc (:std-scale spec) :scaler scaler)
           :features-cols  (ds/column-names features-ds)
           :sigma          sigma
           :features-array feat-array
           :y-array        y-array)))