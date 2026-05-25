(defproject svm-techml-smile "0.1.0-SNAPSHOT"
  :description "Shared utilities for tech.ml.dataset + Smile SVM/SVR workflows"
  :license {:name "MIT"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [techascent/tech.ml.dataset "8.021"]
                 [com.github.haifengl/smile-core "6.1.0"]]
  :source-paths ["src"]
  :test-paths ["test"]
  ;; smile-core 6.1.0 is compiled with Java 25 — run lein with:
  ;;   JAVA_HOME=$(/usr/libexec/java_home -v 26) lein <command>
  :jvm-opts ["--enable-native-access=ALL-UNNAMED"]
  :repl-options {:timeout 240000})