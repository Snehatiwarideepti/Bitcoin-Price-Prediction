import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Loading dataset
df = pd.read_csv("data/Bitcoin_History_Yahoo.csv")

# Cleaned price columns
for col in ["Price", "Open", "High", "Low"]:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# Cleaned volume column
def clean_volume(val):
    val = str(val)
    if "K" in val:
        return float(val.replace("K", "")) * 1000
    elif "M" in val:
        return float(val.replace("M", "")) * 1000000
    elif "B" in val:
        return float(val.replace("B", "")) * 1000000000
    return np.nan

df["Volume"] = df["Volume"]
df["Date"] = pd.to_datetime(df["Date"])

# Removed duplicate dates and sort chronologically
df = df.drop_duplicates(subset=["Date"])
df = df[(df["Price"] > 0) & (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0)]
df = df.sort_values("Date").reset_index(drop=True)

# Replaced missing volume with average
df["Volume"] = df["Volume"].fillna(df["Volume"].mean())

# 2. Feature engineering
df["prev_close"]    = df["Price"].shift(1)
df["ma_7"]          = df["Price"].rolling(7).mean()
df["ma_14"]         = df["Price"].rolling(14).mean()
df["daily_return"]  = (df["Price"] - df["prev_close"]) / df["prev_close"]
df["volatility_7d"] = df["daily_return"].rolling(7).std()
df["next_close"]    = df["Price"].shift(-1)

# Target: predict next-day RETURN (%) instead of raw price.
# Returns are always in range ~[-20%, +20%] — trees can handle this.
# We convert predictions back to price after inference.
df["next_return"] = (df["next_close"] - df["Price"]) / df["Price"]

df = df.dropna().reset_index(drop=True)
print("Total clean records:", len(df))

# 3. Train-Test split (80% train, 20% test)
split_point = int(len(df) * 0.8)
train_df = df.iloc[:split_point]
test_df  = df.iloc[split_point:].copy()

print(f"Training rows (older 80%): {len(train_df)}")
print(f"Testing rows  (recent 20%): {len(test_df)}")

# 4. PySpark setup and VectorAssembler
spark = SparkSession.builder.appName("CryptoPrediction").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

train_data = spark.createDataFrame(train_df)
test_data  = spark.createDataFrame(test_df)

features = ["Open", "High", "Low", "Price", "Volume",
            "prev_close", "ma_7", "ma_14", "daily_return", "volatility_7d"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

train_data = assembler.transform(train_data)
test_data  = assembler.transform(test_data)

# 5. Train Linear Regression model  (target = next_return)
lr = LinearRegression(featuresCol="features", labelCol="next_return")
lr_model = lr.fit(train_data)
lr_pred  = lr_model.transform(test_data)

# 6. Train Decision Tree model (tuned hyperparameters)
dt = DecisionTreeRegressor(
    featuresCol="features", labelCol="next_return",
    maxDepth=10, minInstancesPerNode=5
)
dt_model = dt.fit(train_data)
dt_pred = dt_model.transform(test_data)

# 6b. Random Forest
rf = RandomForestRegressor(
    featuresCol="features", labelCol="next_return",
    numTrees=100, maxDepth=10, minInstancesPerNode=5, seed=42
)
rf_model = rf.fit(train_data)
rf_pred = rf_model.transform(test_data)

# 6c. Gradient Boosted Trees
gbt = GBTRegressor(
    featuresCol="features", labelCol="next_return",
    maxIter=100, maxDepth=5, stepSize=0.1, seed=42
)
gbt_model = gbt.fit(train_data)
gbt_pred = gbt_model.transform(test_data)

# 7. Convert predicted returns → predicted prices for evaluation
# predicted_price = current_price * (1 + predicted_return)
from pyspark.sql.functions import col as scol

def add_price_pred(pred_df):
    """Convert next_return prediction back to next_close price."""
    return pred_df.withColumn(
        "pred_price", scol("Price") * (1 + scol("prediction"))
    )

lr_pred_p  = add_price_pred(lr_pred)
dt_pred_p  = add_price_pred(dt_pred)
rf_pred_p  = add_price_pred(rf_pred)
gbt_pred_p = add_price_pred(gbt_pred)

# Evaluate on price scale (RMSE/MAE in USD, R² and accuracy on actual prices)
evaluator_rmse = RegressionEvaluator(labelCol="next_close", predictionCol="pred_price", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="next_close", predictionCol="pred_price", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="next_close", predictionCol="pred_price", metricName="r2")

def compute_accuracy(pred_df, label_col="next_close", pred_col="pred_price", tolerance=0.05):
    total  = pred_df.count()
    within = pred_df.filter(
        ((pred_df[pred_col] - pred_df[label_col]) / pred_df[label_col]).between(-tolerance, tolerance)
    ).count()
    return (within / total) * 100

lr_rmse  = evaluator_rmse.evaluate(lr_pred_p);  lr_mae  = evaluator_mae.evaluate(lr_pred_p);  lr_r2  = evaluator_r2.evaluate(lr_pred_p);  lr_acc  = compute_accuracy(lr_pred_p)
dt_rmse  = evaluator_rmse.evaluate(dt_pred_p);  dt_mae  = evaluator_mae.evaluate(dt_pred_p);  dt_r2  = evaluator_r2.evaluate(dt_pred_p);  dt_acc  = compute_accuracy(dt_pred_p)
rf_rmse  = evaluator_rmse.evaluate(rf_pred_p);  rf_mae  = evaluator_mae.evaluate(rf_pred_p);  rf_r2  = evaluator_r2.evaluate(rf_pred_p);  rf_acc  = compute_accuracy(rf_pred_p)
gbt_rmse = evaluator_rmse.evaluate(gbt_pred_p); gbt_mae = evaluator_mae.evaluate(gbt_pred_p); gbt_r2 = evaluator_r2.evaluate(gbt_pred_p); gbt_acc = compute_accuracy(gbt_pred_p)

print("\n--- Model Evaluation Results (prices in USD) ---")
print(f"{'Model':<28} {'RMSE':>10} {'MAE':>10} {'R2':>8} {'Accuracy':>12}")
print("-" * 72)
print(f"{'Linear Regression':<28} {lr_rmse:>10.2f} {lr_mae:>10.2f} {lr_r2:>8.4f} {lr_acc:>10.2f}%")
print(f"{'Decision Tree (tuned)':<28} {dt_rmse:>10.2f} {dt_mae:>10.2f} {dt_r2:>8.4f} {dt_acc:>10.2f}%")
print(f"{'Random Forest':<28} {rf_rmse:>10.2f} {rf_mae:>10.2f} {rf_r2:>8.4f} {rf_acc:>10.2f}%")
print(f"{'Gradient Boosted Trees':<28} {gbt_rmse:>10.2f} {gbt_mae:>10.2f} {gbt_r2:>8.4f} {gbt_acc:>10.2f}%")
print("-" * 72)
print("Note: Accuracy = % of predictions within ±5% of actual price")

# 8. Plot actual vs predicted prices and volume
os.makedirs("outputs", exist_ok=True)

test_plot  = test_df.copy()
lr_res     = lr_pred_p.select("pred_price").toPandas()
dt_res     = dt_pred_p.select("pred_price").toPandas()
rf_res     = rf_pred_p.select("pred_price").toPandas()
gbt_res    = gbt_pred_p.select("pred_price").toPandas()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Price plot — all 4 models
ax1.plot(test_plot["Date"], test_plot["next_close"], label="Actual Price",           color="black",  linewidth=1.5)
ax1.plot(test_plot["Date"], lr_res["pred_price"],    label="Linear Regression",      color="blue",   linestyle="--", linewidth=1.2)
ax1.plot(test_plot["Date"], dt_res["pred_price"],    label="Decision Tree (tuned)",  color="red",    linestyle=":",  linewidth=1.0)
ax1.plot(test_plot["Date"], rf_res["pred_price"],    label="Random Forest",          color="green",  linestyle="-.", linewidth=1.2)
ax1.plot(test_plot["Date"], gbt_res["pred_price"],   label="Gradient Boosted Trees", color="orange", linestyle="--", linewidth=1.2)
ax1.set_title("Bitcoin Price Prediction — All Models (Test Period)")
ax1.set_ylabel("Price in USD")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.4)

ax1.plot(test_plot["Date"], gbt_res["pred_price"],   label="Gradient Boosted Trees", color="orange", linestyle="--", linewidth=1.2)
ax1.set_title("Bitcoin Price Prediction — All Models (Test Period)")
ax1.set_ylabel("Price in USD")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.4)

# Volume plot
ax2.bar(test_plot["Date"], test_plot["Volume"], color="steelblue", alpha=0.6, label="Volume", width=1)
ax2.set_title("Trading Volume Trend")
ax2.set_xlabel("Date")
ax2.set_ylabel("Volume (USD)")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/project64_analysis.png", dpi=150)
plt.savefig("outputs/model_comparison.png",   dpi=150)
plt.close()
print("\nPlot saved to outputs/project64_analysis.png")

# 9. Save models
os.makedirs("models", exist_ok=True)
lr_model.write().overwrite().save("models/linear_regression_model")
dt_model.write().overwrite().save("models/decision_tree_model")
rf_model.write().overwrite().save("models/random_forest_model")
gbt_model.write().overwrite().save("models/gbt_model")
print("Models saved successfully.")

spark.stop()
