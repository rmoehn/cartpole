(ns user
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.repl :refer [pst doc find-doc]]
            [clojure.string :as string]
            [clojure.tools.namespace.repl :refer [refresh]]
            [cheshire.core :as cheshire]
            [com.rpl.specter :as s]
            ))

;; Credits: https://github.com/nathanmarz/specter/blob/master/src/clj/com/rpl/specter/impl.cljc
(s/defnav s-butlast []
  (select* [this structure next-fn] (next-fn (butlast structure)))
  (transform* [this structure next-fn]
              (let [structurev (vec structure)
                    newpart (next-fn (vec (butlast structure)))
                    res (concat newpart (list (last structurev)))]
                (if (vector? structure)
                  (vec res)
                  res))))

(defn md-line-seq [md-dir cmd]
  (let [[_ md-name] (re-matches #"%load_md ([^.].*\.md)" (first cmd))]
    (->> (io/file md-dir md-name)
         slurp
         string/split-lines
         (s/transform [s-butlast s/ALL]
                      #(str % \newline)))))

;; p … path
(defn subst-md [nb-p md-dir-p out-p]
  (let [orig (cheshire/parse-string (slurp nb-p) true)
        proc (->> orig
                  (s/transform
                    [:cells s/ALL :source #(re-matches #"%load_md [^.].*\.md"
                                                       (or (first %) ""))]
                    (partial md-line-seq md-dir-p)))]
    (spit out-p (cheshire/generate-string proc {:pretty true}))))



(comment

  (subst-md "../notebooks/OffSwitchCartPole.ipynb" "../interruptibility"
            "../notebooks/ProcessedOSCP.ipynb")

  (s/select [s-butlast s/ALL] [1 2 3 4 5])

  (s/transform [s-butlast s/ALL] inc [1 2 3 4])

  (md-line-seq "../interruptibility" "%load_md results.md")





       (->> [[1 45 43 3]]
            (s/select [s/ALL s/FIRST #(> 4 %)]))

  )
