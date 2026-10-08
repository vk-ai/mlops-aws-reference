"""The same feature transform as ``features.add_features``, as a local PySpark job.

Optional: needs ``pip install .[spark]`` and a Java 17+ runtime. Used by the DAG when
``features.engine: spark``; tests check it matches the pandas output exactly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def spark_session(app: str = "mlops-ref-features"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[2]").appName(app)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def add_features_spark(sdf):
    from pyspark.sql import functions as F

    return (
        sdf.withColumn("amount_per_month", F.col("credit_amount") / F.col("duration_months"))
        .withColumn("log_credit_amount", F.log1p(F.col("credit_amount")))
        .withColumn("young_applicant", F.when(F.col("age_years") < 25, F.lit(1.0)).otherwise(F.lit(0.0)))
    )


def run(input_parquet: Path, output_parquet: Path) -> int:
    spark = spark_session()
    try:
        sdf = spark.read.parquet(str(input_parquet))
        out = add_features_spark(sdf)
        out.coalesce(1).write.mode("overwrite").parquet(str(output_parquet))
        return out.count()
    finally:
        spark.stop()


def featurize_pandas_via_spark(df: pd.DataFrame) -> pd.DataFrame:
    spark = spark_session()
    try:
        return add_features_spark(spark.createDataFrame(df)).toPandas()
    finally:
        spark.stop()
