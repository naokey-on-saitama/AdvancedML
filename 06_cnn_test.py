import numpy as np
import tensorflow as tf
import keras_tuner as kt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from src.load_data import dataset
from src.decorator import add_print

# データセットの読み込み
DATASET = dataset()

# MLPのチューニング用関数
def mlp_bulder(hp: kt.HyperParameters) -> tf.keras.Model:
    hp_units = hp.Int("units", min_value=8, max_value=129, step=8)
    hp_layers = hp.Int("layers", min_value=2, max_value=11, step=2)
    
    model = tf.keras.Sequential()
    for _ in range(hp_layers):
        model.add(tf.keras.layers.Dense(units=hp_units, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model

def main():

    dataname = "fashion-mnist"    
    # 多層パーセプトロンの学習
    mlp_tuner = kt.Hyperband(
        mlp_bulder,
        objective="val_accuracy",
        max_epochs=10,
        directory="./logs/tuner",
        project_name=f"{dataname}_mlp_tuning"
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    mlp_tuner.search(DATASET[dataname], epochs=50, validation_split=0.2, callbacks=[stop_early])


models = {
    "mlp": tf.keras.models.Sequential(),
    "cnn": tf.keras.models.Sequential(),
    "rnn": tf.keras.models.Sequential()
}

if __name__ == "__main__":
    pass