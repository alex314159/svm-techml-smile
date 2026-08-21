(ns svm-techml-smile.bench
  "Benchmarks for the hot paths. Not part of the released jar — lives in the
  :bench profile only. Run with:  lein bench

  Criterium reports time; it does not report allocation, and allocation is half
  of what matters here (the whole feature matrix passes through these calls),
  so `alloc-per-call` covers that separately."
  (:require [criterium.core :as crit]
            [tech.v3.dataset :as ds]
            [svm-techml-smile.bridge :as bridge]
            [svm-techml-smile.kernel :as kernel]
            [svm-techml-smile.preprocessing :as pre])
  (:import (java.lang.management ManagementFactory)
           (com.sun.management ThreadMXBean)))

(set! *warn-on-reflection* true)

(def ^ThreadMXBean thread-mx (ManagementFactory/getThreadMXBean))

(defn alloc-per-call
  "Mean bytes allocated per call of f, in MB."
  [n f]
  (dotimes [_ 3] (f))
  (System/gc)
  (let [before (.getCurrentThreadAllocatedBytes thread-mx)]
    (dotimes [_ n] (f))
    (/ (- (.getCurrentThreadAllocatedBytes thread-mx) before) 1048576.0 n)))

(defn report [label f]
  (println (format "\n=== %s ===" label))
  (println (format "allocation: %.1f MB/call" (alloc-per-call 10 f)))
  (crit/quick-bench (f)))

(defn feature-ds
  "Dense numeric dataset, nrow x ncol, deterministic so runs are comparable."
  [nrow ncol]
  (ds/->dataset (into {} (for [j (range ncol)]
                           [(keyword (str "c" j))
                            (double-array (map #(Math/sin (+ % (* 1000 j))) (range nrow)))]))))

(defn bond-ds
  "Cross-sectional dataset shaped like a jasmine SVR input."
  [nrow]
  (ds/->dataset {:id  (mapv str (range nrow))
                 :dur (double-array (map #(inc (mod % 30)) (range nrow)))
                 :rat (double-array (map #(inc (mod % 10)) (range nrow)))
                 :ccy (mapv #(if (even? %) "USD" "EUR") (range nrow))
                 :ztw (double-array (map #(+ 100 (mod % 200)) (range nrow)))}))

(defn -main [& _]
  (let [big  (feature-ds 20000 20)
        bond (bond-ds 20000)]
    (println "feature matrix:" (ds/row-count big) "rows x" (ds/column-count big) "cols")
    (report "bridge/dataset->array"   #(bridge/dataset->array big))
    (report "kernel/estimate-sigma"   #(kernel/estimate-sigma big))
    (report "kernel/prepare-features" #(kernel/prepare-features big))
    (report "preprocessing/prepare-svr-data"
            #(pre/prepare-svr-data
              (pre/svr-model-spec bond :id :ztw [:dur :rat] :one-hot-cols [:ccy])))
    (shutdown-agents)))
