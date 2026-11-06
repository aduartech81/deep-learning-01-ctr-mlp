"""Pipeline de PySpark con la misma lógica de preprocesamiento que scikit-learn.

- Agrupación de raras con vocabulario contado SOLO en entrenamiento (join broadcast).
- 'hash': FeatureHasher (alta cardinalidad) + StringIndexer/OneHotEncoder (baja cardinalidad).
- 'freq': log1p(frecuencia de entrenamiento) + StandardScaler.
- Modelos: LogisticRegression, GBTClassifier, MultilayerPerceptronClassifier.
- Tuning: ParamGridBuilder evaluado sobre el día 29. No se usa TrainValidationSplit ni
  CrossValidator porque ambos parten al azar y romperían el protocolo temporal.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,
                                       MultilayerPerceptronClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (FeatureHasher, OneHotEncoder, StandardScaler, StringIndexer,
                                VectorAssembler)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

RARE = "__RARE__"


def get_spark(cfg, cores: int | str = "*", app="ctr-avazu") -> SparkSession:
    s = cfg["spark"]
    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()
    os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
    spark = (SparkSession.builder.appName(app).master(f"local[{cores}]")
             .config("spark.driver.memory", s["driver_memory"])
             .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.driver.maxResultSize", "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def read_train(spark, cfg, n, sample_path_fn):
    """n=None -> entrenamiento completo desde el Parquet particionado; si no, la misma muestra
    materializada que usa scikit-learn."""
    if n is None:
        df = spark.read.parquet(str(cfg.path("parquet_dir")))
        return df.filter(F.col("day").isin(cfg["split"]["train_days"]))
    return spark.read.parquet(str(sample_path_fn(cfg, "train", n)))


class SparkFeaturizer:
    def __init__(self, high_card, low_card, min_count):
        self.high_card, self.low_card, self.min_count = list(high_card), list(low_card), min_count
        self.cols = self.high_card + self.low_card

    def fit(self, train: DataFrame):
        self.vocab = {}
        self.rare_cnt = {}
        n = train.count()
        for c in self.cols:
            vc = train.groupBy(F.col(c).cast("string").alias("v")).count()
            kept = vc.filter(F.col("count") >= self.min_count).withColumnRenamed("count", f"{c}__cnt")
            kept = kept.withColumnRenamed("v", f"{c}__v").cache()
            s = kept.agg(F.sum(f"{c}__cnt")).first()[0] or 0
            self.vocab[c] = kept
            self.rare_cnt[c] = max(n - s, 1)
        self.n_train_ = n
        return self

    def apply(self, df: DataFrame) -> DataFrame:
        """Agrupa raras (y no vistas) en __RARE__ y añade log1p(frecuencia) por columna."""
        for c in self.cols:
            v = self.vocab[c]
            df = (df.join(F.broadcast(v), F.col(c).cast("string") == F.col(f"{c}__v"), "left")
                    .withColumn(f"{c}__g", F.when(F.col(f"{c}__cnt").isNull(), F.lit(RARE))
                                .otherwise(F.col(c).cast("string")))
                    .withColumn(f"{c}__f", F.log1p(F.coalesce(F.col(f"{c}__cnt"), F.lit(self.rare_cnt[c])).cast("double")))
                    .drop(f"{c}__v", f"{c}__cnt"))
        return df

    def build_pipeline(self, encoding: str, hash_dim: int | None = None) -> Pipeline:
        if encoding == "hash":
            hasher = FeatureHasher(inputCols=[f"{c}__g" for c in self.high_card], outputCol="h",
                                   numFeatures=int(hash_dim))
            idx = StringIndexer(inputCols=[f"{c}__g" for c in self.low_card],
                                outputCols=[f"{c}__i" for c in self.low_card], handleInvalid="keep")
            ohe = OneHotEncoder(inputCols=[f"{c}__i" for c in self.low_card],
                                outputCols=[f"{c}__o" for c in self.low_card], dropLast=False,
                                handleInvalid="keep")
            asm = VectorAssembler(inputCols=["h"] + [f"{c}__o" for c in self.low_card], outputCol="features")
            return Pipeline(stages=[hasher, idx, ohe, asm])
        if encoding == "freq":
            asm = VectorAssembler(inputCols=[f"{c}__f" for c in self.cols], outputCol="raw")
            sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
            return Pipeline(stages=[asm, sc])
        raise ValueError(f"Codificación no disponible en Spark: {encoding}")


def prepare(feat: SparkFeaturizer, train, valid, test, encoding, hash_dim=None):
    """Ajusta el pipeline SOLO con entrenamiento y transforma las tres particiones."""
    keep = ["id", "click", "features"]
    tr = feat.apply(train)
    pipe = feat.build_pipeline(encoding, hash_dim).fit(tr)
    out = []
    for d in (tr, feat.apply(valid), feat.apply(test)):
        out.append(pipe.transform(d).select(*keep).withColumn("label", F.col("click").cast("double")))
    dim = out[0].select("features").first()[0].size
    return out[0], out[1], out[2], dim


def make_model(kind: str, dim: int, params: dict, seed: int):
    if kind == "mlp":
        return MultilayerPerceptronClassifier(layers=[dim] + list(params["hidden"]) + [2],
                                              stepSize=params["stepSize"], maxIter=params["maxIter"],
                                              solver=params["solver"], seed=seed, blockSize=128,
                                              labelCol="label", featuresCol="features")
    if kind == "lr":
        return LogisticRegression(maxIter=100, regParam=params.get("regParam", 1e-4),
                                  elasticNetParam=0.0, labelCol="label", featuresCol="features")
    if kind == "gbt":
        return GBTClassifier(maxIter=params.get("maxIter", 50), maxDepth=params.get("maxDepth", 5),
                             stepSize=0.1, seed=seed, labelCol="label", featuresCol="features")
    raise ValueError(kind)


def predict_proba(model, df) -> pd.DataFrame:
    """Devuelve id y probabilidad de clic, ordenado por id."""
    pr = model.transform(df).select("id", vector_to_array("probability")[1].alias("p"))
    return pr.toPandas().sort_values("id").reset_index(drop=True)


def auc_spark(model, df) -> float:
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction",
                                       metricName="areaUnderROC")
    return ev.evaluate(model.transform(df))


def fit_timed(est, train):
    t0 = time.perf_counter()
    m = est.fit(train)
    return m, time.perf_counter() - t0


def run_spark_model(kind, params, trainP, validP, testP, dim, seed, yva, yte, va_ids, te_ids):
    """Entrena, elige umbral en validación, evalúa en prueba. Devuelve (registro, p_test, modelo)."""
    from . import metrics as M
    from .bench import measure
    rec = {"framework": "spark", "model": kind, "seed": seed, "class_weight": "none", "dim_entrada": dim}
    est = make_model(kind, dim, params, seed)
    with measure(rec, "train"):
        model = est.fit(trainP)
    pv = predict_proba(model, validP)
    assert np.array_equal(pv["id"].to_numpy(), va_ids), "validación desalineada"
    with measure(rec, "predict", len(te_ids)):
        pt = predict_proba(model, testP)
    assert np.array_equal(pt["id"].to_numpy(), te_ids), "prueba desalineada"
    p_va, p_te = pv["p"].to_numpy(), pt["p"].to_numpy()
    thr = M.best_threshold(yva, p_va)
    rec.update({f"val_{k}": v for k, v in M.all_metrics(yva, p_va, thr).items()
                if k in ("roc_auc", "pr_auc", "log_loss")})
    rec.update(M.all_metrics(yte, p_te, thr))
    return rec, p_te, model


# --------------------------------------------------------------- evidencia Spark UI
def ui_stage_metrics(spark) -> pd.DataFrame:
    """Lee la API REST de la Spark UI: shuffle, spill a disco y sesgo entre tareas por etapa."""
    sc = spark.sparkContext
    base = sc.uiWebUrl
    if not base:
        return pd.DataFrame()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"{base}/api/v1/applications/{sc.applicationId}/stages?status=complete"
    try:
        stages = json.loads(opener.open(url, timeout=10).read())
    except Exception as e:  # la UI puede estar deshabilitada
        return pd.DataFrame([{"error": str(e)}])
    rows = []
    for s in stages:
        row = {k: s.get(k) for k in ["stageId", "attemptId", "name", "numTasks", "executorRunTime",
                                       "shuffleReadBytes", "shuffleWriteBytes", "memoryBytesSpilled",
                                       "diskBytesSpilled", "inputBytes"]}
        try:
            q = json.loads(opener.open(
                f"{url.split('?')[0]}/{s['stageId']}/{s['attemptId']}/taskSummary?quantiles=0.5,0.95,1.0",
                timeout=10).read())
            rt = q.get("executorRunTime", [np.nan] * 3)
            row["tarea_mediana_ms"], row["tarea_p95_ms"], row["tarea_max_ms"] = rt
            row["sesgo_max_sobre_mediana"] = rt[2] / rt[0] if rt[0] else np.nan
        except Exception:
            pass
        rows.append(row)
    return pd.DataFrame(rows)
