"""
Configuration for training
"""

model_name = 'MTL_2tanhs_NoBN_dropout0'

target_names = {
    "conversion_label": "binary_classification",
    "cancellation_label": "binary_classification",
    "accceptance_label": "binary_classification",
}

params = {
    "default_target": "conversion_label",
    "target_names": target_names,
    "model_name": model_name,
    "batch_size": 256,
    "epochs": 10,
    "starter_learning_rate": 0.005,
    "decay_steps": 30000,
    "decay_rate": 0.96,
    "hidden_layers": {
        "1": dict(
            n_units=32,
            activation='tanh',
            dropout=0,
            batch_normalization=False
        ),
        "2": dict(
            n_units=32,
            activation='tanh',
            dropout=0,
            batch_normalization=False
        ),
    },
    "with_uncertainty": True,
}
