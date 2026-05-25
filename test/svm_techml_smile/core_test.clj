(ns svm-techml-smile.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [tech.v3.dataset :as ds]
            [svm-techml-smile.bridge :refer [dscol->smilecol dataset->dataframe dataset->array cast-to-float]]
            [svm-techml-smile.features :refer [build-column build-many-columns]]
            [svm-techml-smile.kernel :refer [estimate-sigma fit-RBF-SVM fit-RBF-SVR prepare-features]]))

(def sample-ds
  (ds/->dataset {:a [1.0 2.0 3.0 4.0 5.0]
                 :b [2.0 4.0 1.0 3.0 5.0]}))

(deftest bridge-tests
  (testing "dscol->smilecol preserves name"
    (let [col   (sample-ds :a)
          smile-col (dscol->smilecol col)]
      (is (= "a" (.name smile-col)))))

  (testing "dataset->dataframe has correct shape"
    (let [df (dataset->dataframe sample-ds)]
      (is (= 5 (.nrow df)))
      (is (= 2 (.ncol df)))))

  (testing "dataset->array has correct shape"
    (let [arr (dataset->array sample-ds)]
      (is (= 2 (alength arr)))
      (is (= 5 (alength (aget arr 0))))))

  (testing "cast-to-float changes column type"
    (let [floated (cast-to-float sample-ds)]
      (is (= :float32 (-> floated :a meta :datatype))))))

(deftest features-tests
  (testing "build-column with assoc-fn"
    (let [result (build-column sample-ds {:column :c :assoc-fn #(map + (:a %) (:b %))})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-column with map-fn"
    (let [result (build-column sample-ds {:column :c :map-fn + :arg-cols [:a :b]})]
      (is (= [3.0 6.0 4.0 7.0 10.0] (vec (result :c))))))

  (testing "build-many-columns chains correctly"
    (let [result (build-many-columns sample-ds [{:column :c :assoc-fn #(map + (:a %) (:b %))}
                                                {:column :d :map-fn * :arg-cols [:a :b]}])]
      (is (contains? result :c))
      (is (contains? result :d)))))

(deftest kernel-tests
  (testing "estimate-sigma returns positive double"
    (let [sigma (estimate-sigma sample-ds)]
      (is (pos? sigma))
      (is (double? sigma))))

  (testing "prepare-features returns expected keys"
    (let [result (prepare-features sample-ds)]
      (is (contains? result :scaler))
      (is (contains? result :sigma))
      (is (contains? result :features-array))
      (is (pos? (:sigma result))))))