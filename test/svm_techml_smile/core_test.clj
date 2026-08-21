(ns svm-techml-smile.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [svm-techml-smile.bridge :refer [dscol->smilecol dataset->dataframe dataset->array cast-to-float]]
            [svm-techml-smile.features :refer [build-column build-many-columns]]
            [svm-techml-smile.kernel :refer [estimate-sigma prepare-features scale-new-features predict predict-all rsq
                                             fit-scale transform-scale fit-RBF-SVM fit-RBF-SVR]]
            [svm-techml-smile.preprocessing :refer [svr-model-spec one-hot-reducer prepare-svr-data]]
            [svm-techml-smile.classification :refer [labels->01 evaluate-classifier svm-model-spec]]
            [svm-techml-smile.core :as core]
            [svm-techml-smile.bridge]
            [svm-techml-smile.features]
            [svm-techml-smile.kernel]
            [svm-techml-smile.preprocessing]
            [svm-techml-smile.classification]))

(def sample-ds
  (ds/->dataset {:a [1.0 2.0 3.0 4.0 5.0]
                 :b [2.0 4.0 1.0 3.0 5.0]}))

(defn close? [a b] (< (Math/abs (- (double a) (double b))) 1e-9))

;;; bridge

(deftest bridge-tests
  (testing "dscol->smilecol preserves name and values"
    (let [smile-col (dscol->smilecol (sample-ds :a))]
      (is (= "a" (.name smile-col)))
      (is (= [1.0 2.0 3.0 4.0 5.0] (vec (.toDoubleArray smile-col))))))

  (testing "dataset->dataframe shape"
    (let [df (dataset->dataframe sample-ds)]
      (is (= 5 (.nrow df)))
      (is (= 2 (.ncol df)))))

  (testing "dataset->array is row-major: rows = observations, cols = features"
    (let [arr (dataset->array sample-ds)]
      (is (= 5 (alength arr)) "one row per observation")
      (is (= 2 (alength (aget arr 0))) "one column per feature")
      (is (= [[1.0 2.0] [2.0 4.0] [3.0 1.0] [4.0 3.0] [5.0 5.0]] (mapv vec arr)))))

  (testing "dataset->array agrees with the Smile DataFrame round-trip it replaces"
    (is (= (mapv vec (.toArray (dataset->dataframe sample-ds) (into-array String [])))
           (mapv vec (dataset->array sample-ds)))))

  (testing "dataset->array follows column-names order, not insertion luck"
    (let [reordered (ds/select-columns sample-ds [:b :a])]
      (is (= [:b :a] (vec (ds/column-names reordered))))
      (is (= [2.0 1.0] (vec (aget (dataset->array reordered) 0))))))

  (testing "dataset->array coerces non-double columns"
    (is (= [[1.0] [2.0]] (mapv vec (dataset->array (ds/->dataset {:i (int-array [1 2])}))))))

  (testing "dataset->array refuses missing values rather than feeding NaN to Smile"
    (is (thrown? clojure.lang.ExceptionInfo
                 (dataset->array (ds/->dataset {:a [1.0 2.0 ##NaN]}))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (dataset->array (ds/->dataset {:a [1.0 2.0 nil]}))))
    (is (= {:a 1} (-> (try (dataset->array (ds/->dataset {:a [1.0 nil] :b [1.0 2.0]}))
                           (catch clojure.lang.ExceptionInfo e (ex-data e)))
                      :missing-by-column))
        "error names the offending column"))

  (testing "cast-to-float"
    (is (= :float32 (-> (cast-to-float sample-ds) :a meta :datatype)))
    (is (= [[1.0 2.0] [2.0 4.0] [3.0 1.0] [4.0 3.0] [5.0 5.0]]
           (mapv vec (dataset->array (cast-to-float sample-ds))))
        "float32 columns still widen to double[][]")))

;;; features

(deftest features-tests
  (testing "build-column with assoc-fn"
    (let [result (build-column sample-ds {:column :c :assoc-fn #(map + (:a %) (:b %))})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-column with map-fn"
    (let [result (build-column sample-ds {:column :c :map-fn + :arg-cols [:a :b]})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-many-columns chains, and later specs see earlier columns"
    (let [result (build-many-columns sample-ds
                                     [{:column :c :assoc-fn #(map + (:a %) (:b %))}
                                      {:column :d :map-fn * :arg-cols [:a :c]}])]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))
      (is (= [3.0 12.0 12.0 28.0 50.0] (vec (result :d)))))))

;;; kernel

(deftest kernel-tests
  (testing "estimate-sigma matches sqrt(0.5 * ncol * var(flattened))"
    (let [flat (concat (sample-ds :a) (sample-ds :b))]
      (is (close? (Math/sqrt (* 0.5 2 (dfn/variance flat)))
                  (estimate-sigma sample-ds)))))

  (testing "estimate-sigma throws on a zero-variance feature matrix"
    (is (thrown? clojure.lang.ExceptionInfo
                 (estimate-sigma (ds/->dataset {:a [1.0 1.0 1.0]})))))

  (testing "fit-scale / transform-scale standardise to mean 0, sd 1"
    (let [scaled (transform-scale sample-ds (fit-scale sample-ds))]
      (doseq [c [:a :b]]
        (is (close? 0.0 (dfn/mean (scaled c))))
        (is (close? 1.0 (dfn/standard-deviation (scaled c)))))))

  (testing "prepare-features returns a scaled row-major matrix and reusable scaler"
    (let [{:keys [scaler sigma features-array]} (prepare-features sample-ds)]
      (is (pos? sigma))
      (is (= 5 (alength features-array)))
      (is (= 2 (alength (aget features-array 0))))
      (is (close? 0.0 (dfn/mean (map #(aget ^doubles % 0) features-array))))
      (testing "scale-new-features reuses the training scaler, not a refit"
        (is (= (mapv vec features-array)
               (mapv vec (scale-new-features sample-ds scaler))))
        (let [one-row (scale-new-features (ds/select-rows sample-ds [0]) scaler)]
          (is (= 1 (alength one-row)))
          (is (= (vec (aget features-array 0)) (vec (aget one-row 0))))))))

  (testing "fit-RBF-SVM separates a linearly separable problem"
    (let [x (into-array (concat (for [i (range 20)] (double-array [(+ 5.0 (* 0.1 i)) 5.0]))
                                (for [i (range 20)] (double-array [(+ -5.0 (* 0.1 i)) -5.0]))))
          y (int-array (concat (repeat 20 1) (repeat 20 -1)))
          model (fit-RBF-SVM x y 1.0 0.001 10.0)]
      (is (= (vec y) (vec (.predict model x))) "should fit separable data exactly")))

  (testing "fit-RBF-SVR tracks a monotone target"
    (let [x (into-array (for [i (range 40)] (double-array [(* 0.25 i)])))
          y (double-array (for [i (range 40)] (* 2.0 (* 0.25 i))))
          km (fit-RBF-SVR x y 2.0 0.01 100.0)
          preds (map #(.predict km ^doubles %) x)]
      (is (pos? (dfn/pearsons-correlation (vec preds) (vec y))))
      (is (> 0.99 (/ (dfn/mean (dfn/abs (dfn/- (vec preds) (vec y))))
                     (dfn/standard-deviation (vec y))))
          "mean abs error well under the spread of y"))))

;;; preprocessing — cross-sectional SVR pipeline

(def cs-ds
  (ds/->dataset {:id   ["A" "B" "C" "D" "E" "F" "G" "H" "I" "J"]
                 :dur  [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
                 :rat  [1.0 2.0 3.0 1.0 2.0 3.0 1.0 2.0 3.0 1.0]
                 :ccy  ["USD" "USD" "EUR" "EUR" "USD" "EUR" "USD" "EUR" "USD" "EUR"]
                 :ztw  [100.0 200.0 150.0 180.0 130.0 160.0 110.0 190.0 140.0 170.0]}))

(deftest preprocessing-tests
  (testing "model-spec constructor"
    (let [spec (svr-model-spec cs-ds :id :ztw [:dur :rat] :one-hot-cols [:ccy])]
      (is (= :svr (:type spec)))
      (is (= :id (:id-name spec)))
      (is (= :ztw (get-in spec [:y :column])))
      (is (= [:ccy] (get-in spec [:one-hot :columns])))
      (is (= identity (get-in spec [:y :predict-fn])))))

  (testing "one-hot-reducer encodes and honours removals"
    (let [encoded (one-hot-reducer cs-ds [:ccy] [])
          names   (set (map name (ds/column-names encoded)))]
      (is (contains? names "ccy-USD"))
      (is (contains? names "ccy-EUR"))
      (is (not (contains? names "ccy")) "source column is consumed")
      (is (= [1.0 1.0 0.0] (take 3 (map double (encoded :ccy-USD)))))
      (is (not (contains? (set (ds/column-names (one-hot-reducer cs-ds [:ccy] [:ccy-EUR])))
                          :ccy-EUR))
          "removals drop a dummy to avoid collinearity for OLS")))

  (testing "prepare-svr-data output shape, ordering and contents"
    (let [spec   (svr-model-spec cs-ds :id :ztw [:dur :rat] :one-hot-cols [:ccy])
          result (prepare-svr-data spec)
          {:keys [features-cols sigma features-array y-array]} result]
      (is (pos? sigma))
      (is (some? (get-in result [:std-scale :scaler])))
      (is (not-any? #{:id :ztw} features-cols) "id and target excluded from features")
      (is (= #{:dur :rat :ccy-USD :ccy-EUR} (set features-cols)))
      (is (= 10 (alength features-array)) "one row per bond")
      (is (= (count features-cols) (alength (aget features-array 0))))
      (is (= (vec (cs-ds :ztw)) (vec y-array)) "y is the raw target, untransformed")
      (testing "std-scale columns are standardised, one-hot columns are left alone"
        (let [col-idx (zipmap features-cols (range))
              col-of  #(map (fn [^doubles r] (aget r (col-idx %))) features-array)]
          (is (close? 0.0 (dfn/mean (col-of :dur))))
          (is (close? 1.0 (dfn/standard-deviation (col-of :dur))))
          (is (= #{0.0 1.0} (set (col-of :ccy-USD))))))))

  (testing "prepare-svr-data applies :transformations before scaling"
    (let [spec   (svr-model-spec cs-ds :id :ztw [:dur :logdur]
                                 :transformations [{:column :logdur :map-fn #(Math/log %) :arg-cols [:dur]}]
                                 :one-hot-cols [:ccy] :removals [:rat])
          result (prepare-svr-data spec)]
      (is (some #{:logdur} (:features-cols result)))
      (is (not-any? #{:rat} (:features-cols result)) ":removals honoured")))

  (testing "prepare-svr-data + fit-RBF-SVR roundtrip predicts in target units"
    (let [prepped (prepare-svr-data (svr-model-spec cs-ds :id :ztw [:dur :rat] :one-hot-cols [:ccy]))
          km      (fit-RBF-SVR (:features-array prepped) (:y-array prepped) (:sigma prepped) 0.1 100.0)
          preds   (map #(.predict km ^doubles %) (:features-array prepped))]
      (is (= 10 (count preds)))
      (is (every? #(< 50.0 % 250.0) preds) "predictions land in the ztw range"))))

;;; classification

(deftest classification-tests
  (testing "labels->01 converts from seqs and from int arrays"
    (is (= [1 0 1 0 1] (vec (labels->01 [1 -1 1 -1 1]))))
    (is (= [1 0 1] (vec (labels->01 (int-array [1 -1 1])))))
    (is (= [0 0] (vec (labels->01 [0 -5]))) "only strictly positive maps to 1"))

  (testing "svm-model-spec defaults and overrides"
    (let [spec (svm-model-spec sample-ds)]
      (is (= [42 100 250 0.7 :date]
             ((juxt :seed :start-idx :last-idx :training :id-name) spec))))
    (is (= 7 (:seed (svm-model-spec sample-ds :seed 7)))))

  (testing "evaluate-classifier metrics match a known confusion matrix"
    ;; separable data → perfect in-sample fit → every metric is known up front
    (let [x (into-array (concat (for [i (range 15)] (double-array [(+ 5.0 (* 0.1 i)) 5.0]))
                                (for [i (range 15)] (double-array [(+ -5.0 (* 0.1 i)) -5.0]))))
          y (int-array (concat (repeat 15 1) (repeat 15 -1)))
          km (fit-RBF-SVM x y 1.0 0.001 10.0)
          {:keys [predictions accuracy precision sensitivity fallout confusion-matrix]}
          (evaluate-classifier km x y)]
      (is (= (vec y) (vec predictions)) "predictions stay in {-1,+1}, not {0,1}")
      (is (close? 1.0 accuracy))
      (is (close? 1.0 precision))
      (is (close? 1.0 sensitivity))
      (is (close? 0.0 fallout) "no false positives on separable data")
      (is (str/includes? (str confusion-matrix) "15"))))

  (testing "evaluate-classifier survives all-one-class predictions"
    (let [x (into-array (for [i (range 10)] (double-array [(* 0.1 i) 1.0])))
          y (int-array (repeat 10 -1))
          km (fit-RBF-SVM x (int-array (concat (repeat 9 -1) [1])) 1.0 0.001 1.0)
          result (evaluate-classifier km x y)]
      (is (every? #(contains? result %)
                  [:predictions :y-array :accuracy :precision :sensitivity :fallout :confusion-matrix])))))

;;; core — the single-ns re-export facade

(def impl-namespaces
  '[svm-techml-smile.bridge svm-techml-smile.features svm-techml-smile.kernel
    svm-techml-smile.preprocessing svm-techml-smile.classification])

(deftest core-api-tests
  (testing "every re-exported var carries a docstring and arglists"
    (let [publics (ns-publics 'svm-techml-smile.core)]
      (is (seq publics))
      (doseq [[sym v] publics]
        (is (string? (:doc (meta v))) (str sym " lost its docstring"))
        (is (seq (:arglists (meta v))) (str sym " lost its arglists")))))

  (testing "alias metadata is the defining var's, not a paraphrase"
    (is (= (:doc (meta #'svm-techml-smile.kernel/estimate-sigma))
           (:doc (meta #'core/estimate-sigma))))
    (is (= (:arglists (meta #'svm-techml-smile.preprocessing/prepare-svr-data))
           (:arglists (meta #'core/prepare-svr-data))))
    (is (= (:doc (meta #'svm-techml-smile.classification/svm-model-spec))
           (:doc (meta #'core/ts-model-spec)))
        "renamed alias still carries the source docstring"))

  (testing "core re-exports every public var of every impl namespace"
    ;; compared by identity of the underlying fn, so renames (svm-model-spec ->
    ;; ts-model-spec) count, and a new impl fn that nobody re-exported fails here
    (let [exported (set (map deref (vals (ns-publics 'svm-techml-smile.core))))]
      (doseq [ns-sym impl-namespaces
              [sym v] (ns-publics ns-sym)]
        (is (contains? exported @v)
            (str ns-sym "/" sym " is not re-exported by svm-techml-smile.core")))))

  (testing "aliases call through to the same implementation"
    (is (= (estimate-sigma sample-ds) (core/estimate-sigma sample-ds)))
    (is (= (mapv vec (dataset->array sample-ds))
           (mapv vec (core/dataset->array sample-ds))))))

;;; prediction wrappers — must match the raw Smile interop they replace

(deftest predict-tests
  (let [x  (into-array (for [i (range 40)] (double-array [(* 0.25 i)])))
        y  (double-array (for [i (range 40)] (* 2.0 (* 0.25 i))))
        km (fit-RBF-SVR x y 2.0 0.01 100.0)]

    (testing "predict on one row equals (.predict ^KernelMachine km ^doubles row)"
      (doseq [row (take 5 x)]
        (is (= (.predict ^smile.regression.KernelMachine km ^doubles row)
               (predict km row)))))

    (testing "predict-all on a matrix equals (.predict ^KernelMachine km ^\"[[D\" x)"
      (is (= (vec (.predict ^smile.regression.KernelMachine km ^"[[D" x))
             (vec (predict-all km x)))))

    (testing "predict-all agrees row by row with predict"
      (is (= (mapv #(predict km %) x) (vec (predict-all km x)))))

    (testing "rsq reproduces jasmine's 1 - RSS/sum(y^2)"
      (let [preds (predict-all km x)
            smile-rss (smile.validation.metric.RSS/of y preds)
            jasmine   (- 1 (/ smile-rss (dfn/distance-squared y (repeat (count y) 0.))))]
        (is (close? jasmine (rsq y preds)))))

    (testing "rsq is 1.0 for a perfect fit and 0.0 when predicting all zeros"
      (is (close? 1.0 (rsq y y)))
      (is (close? 0.0 (rsq y (double-array (repeat (alength y) 0.0))))))))
