# svm-techml-smile

Bridging tools between [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) and
[Smile](https://haifengl.github.io/), for SVM/SVR use cases.

## Why

tech.ml used to ship this interop itself, but dropped it when Smile changed its licence.
This library carries the pieces back for the workflows we actually use: dataset ↔ `DataFrame`
/ double-array conversion, RBF kernel estimation, std-scaling, and SVM/SVR fit + predict.

It is also the only place Smile is imported. Consumers depend on this library and get the
Smile version pinned here (`smile-core 6.3.0`) — no Smile coordinate in their own `deps.edn`
/ `project.clj`, no version skew across projects.

## Install

```clojure
[net.clojars.alex314159/svm-techml-smile "0.1.0"]
```

`smile-core 6.3.0` is compiled for **Java 25+**, so point `JAVA_HOME` at a 25/26 JDK and pass
`--enable-native-access=ALL-UNNAMED` (already in this project's `:jvm-opts`).

## Namespaces

| ns | what's in it |
|----|--------------|
| `svm-techml-smile.core` | re-exports everything below — require this one ns if you don't care about the split |
| `svm-techml-smile.bridge` | dataset/column → Smile `DataFrame`, `DoubleVector`, double arrays |
| `svm-techml-smile.kernel` | RBF sigma estimation, std-scaling, `fit-RBF-SVM` / `fit-RBF-SVR`, `predict`, `rsq` |
| `svm-techml-smile.features` | small spec DSL for deriving dataset columns |
| `svm-techml-smile.preprocessing` | cross-sectional SVR model spec + data prep (one-hot, scaling, splits) |
| `svm-techml-smile.classification` | time-series SVM model spec, label conversion, evaluation |

Note `core` aliases via a macro: REPL-driven tooling (Calva, CIDER) resolves the docs, purely
static analysis (clj-kondo, offline Cursive index) doesn't — require the defining ns there.

## Development

```
lein test
lein bench    # criterium, :bench profile only — never in the released jar
```
