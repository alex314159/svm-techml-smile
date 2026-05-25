(ns svm-techml-smile.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [tech.v3.dataset :as ds]
            [svm-techml-smile.bridge :refer [dscol->smilecol dataset->dataframe dataset->array cast-to-float]]
            [svm-techml-smile.features :refer [build-column build-many-columns]]
            [svm-techml-smile.kernel :refer [estimate-sigma prepare-features fit-RBF-SVM fit-RBF-SVR]]
            [svm-techml-smile.preprocessing :refer [svr-model-spec one-hot-reducer prepare-svr-data]]
            [svm-techml-smile.classification :refer [labels->01 evaluate-classifier]]))

(def sample-ds
  (ds/->dataset {:a [1.0 2.0 3.0 4.0 5.0]
                 :b [2.0 4.0 1.0 3.0 5.0]}))

;;; bridge

(deftest bridge-tests
  (testing "dscol->smilecol preserves name"
    (let [smile-col (dscol->smilecol (sample-ds :a))]
      (is (= "a" (.name smile-col)))))

  (testing "dataset->dataframe shape"
    (let [df (dataset->dataframe sample-ds)]
      (is (= 5 (.nrow df)))
      (is (= 2 (.ncol df)))))

  (testing "dataset->array shape"
    (let [arr (dataset->array sample-ds)]
      (is (= 2 (alength arr)))
      (is (= 5 (alength (aget arr 0))))))

  (testing "cast-to-float"
    (is (= :float32 (-> (cast-to-float sample-ds) :a meta :datatype)))))

;;; features

(deftest features-tests
  (testing "build-column with assoc-fn"
    (let [result (build-column sample-ds {:column :c :assoc-fn #(map + (:a %) (:b %))})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-column with map-fn"
    (let [result (build-column sample-ds {:column :c :map-fn + :arg-cols [:a :b]})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-many-columns chains"
    (let [result (build-many-columns sample-ds
                                     [{:column :c :assoc-fn #(map + (:a %) (:b %))}
                                      {:column :d :map-fn * :arg-cols [:a :b]}])]
      (is (contains? result :c))
      (is (contains? result :d)))))

;;; kernel

(deftest kernel-tests
  (testing "estimate-sigma positive"
    (is (pos? (estimate-sigma sample-ds))))

  (testing "prepare-features keys"
    (let [result (prepare-features sample-ds)]
      (is (every? #(contains? result %) [:scaler :sigma :features-array]))
      (is (pos? (:sigma result))))))

;;; preprocessing — cross-sectional SVR pipeline

(def cs-ds
  (ds/->dataset {:id   ["A" "B" "C" "D" "E" "F" "G" "H" "I" "J"]
                 :dur  [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
                 :rat  [1.0 2.0 3.0 1.0 2.0 3.0 1.0 2.0 3.0 1.0]
                 :ccy  ["USD" "USD" "EUR" "EUR" "USD" "EUR" "USD" "EUR" "USD" "EUR"]
                 :ztw  [100.0 200.0 150.0 180.0 130.0 160.0 110.0 190.0 140.0 170.0]}))

(deftest preprocessing-tests
  (testing "model-spec constructor"
    (let [spec (svr-model-spec cs-ds :id :ztw [:dur :rat]
                               :one-hot-cols [:ccy])]
      (is (= :svr (:type spec)))
      (is (= :id (:id-name spec)))
      (is (= :ztw (get-in spec [:y :column])))
      (is (= [:ccy] (get-in spec [:one-hot :columns])))))

  (testing "one-hot-reducer adds columns"
    (let [result (one-hot-reducer cs-ds [:ccy] [])]
      (is (some #(clojure.string/starts-with? (name %) "ccy") (ds/column-names result)))))

  (testing "prepare-svr-data returns required keys"
    (let [spec   (svr-model-spec cs-ds :id :ztw [:dur :rat] :one-hot-cols [:ccy])
          result (prepare-svr-data spec)]
      (is (every? #(contains? result %) [:features-cols :sigma :features-array :y-array]))
      (is (pos? (:sigma result)))
      (is (some? (get-in result [:std-scale :scaler])))))

  (testing "prepare-svr-data + fit-RBF-SVR roundtrip"
    (let [spec    (svr-model-spec cs-ds :id :ztw [:dur :rat] :one-hot-cols [:ccy])
          prepped (prepare-svr-data spec)
          km      (fit-RBF-SVR (:features-array prepped) (:y-array prepped) (:sigma prepped) 0.1 10.0)]
      (is (some? km)))))

;;; classification

(deftest classification-tests
  (testing "labels->01 converts correctly"
    (let [arr (labels->01 [1 -1 1 -1 1])]
      (is (= [1 0 1 0 1] (vec arr)))))

  (testing "evaluate-classifier returns all metric keys"
    (let [;; tiny synthetic dataset: 2 features, 20 samples, labels ±1
          n    20
          x-ds (ds/->dataset {:f1 (repeatedly n #(- (rand) 0.5))
                              :f2 (repeatedly n #(- (rand) 0.5))})
          y-raw (int-array (repeatedly n #(if (pos? (- (rand) 0.5)) 1 -1)))
          prep (prepare-features x-ds)
          km   (fit-RBF-SVM (:features-array prep) y-raw (:sigma prep) 0.001 1.0)
          result (evaluate-classifier km (:features-array prep) y-raw)]
      (is (every? #(contains? result %)
                  [:predictions :y-array :accuracy :precision :sensitivity :fallout :confusion-matrix])))))