(defproject svm-techml-smile "0.1.0-SNAPSHOT"
  :description "Shared utilities for tech.ml.dataset + Smile SVM/SVR workflows. Requires Java 25+ (smile-core 6.3.0)."
  :url "https://github.com/alex314159/svm-techml-smile"
  :license {:name "MIT"
            :url  "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.12.5"]
                 [techascent/tech.ml.dataset "8.024"]
                 [com.github.haifengl/smile-core "6.3.0"]]
  :source-paths ["src"]
  :test-paths ["test"]
  ;; smile-core 6.3.0 is compiled for Java 25+ — point JAVA_HOME at a 25/26 JDK
  ;; (macOS: JAVA_HOME=$(/usr/libexec/java_home -v 26) lein <command>)
  :jvm-opts ["--enable-native-access=ALL-UNNAMED"]
  :repl-options {:timeout 240000}
  ;; Benchmarks only — :bench is outside Leiningen's default profile chain, so
  ;; criterium and dev/ never reach the released jar. Run with: lein bench
  :profiles {:bench {:dependencies [[criterium "0.4.6"]]
                     :source-paths ["dev"]}}
  :aliases {"bench" ["with-profile" "+bench" "run" "-m" "svm-techml-smile.bench"]})