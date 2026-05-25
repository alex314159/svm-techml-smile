(defproject svm-techml-smile "0.1.0-SNAPSHOT"
  :description "Shared utilities for tech.ml.dataset + Smile SVM/SVR workflows"
  :license {:name "MIT"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [techascent/tech.ml.dataset "8.021"]
                 [com.github.haifengl/smile-core "6.1.0"]]
  :source-paths ["src"]
  :test-paths ["test"]
  :repl-options {:timeout 240000})