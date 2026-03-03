import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Lambda, Concatenate
)
import pandas as pd

# ═══════════════════════════════════════════════
# Constrained Encoder-Decoder LSTM
# ═══════════════════════════════════════════════
# Series 1: Historical sell prices
# Series 2: Historical contract values
# Constraint: sell ≥ contract(t - α) + margin
# α = 3 (contract-to-sell lag in timesteps)
# ═══════════════════════════════════════════════

ALPHA = 3  # Lag: contract takes α steps to affect sell
T = 24  # Encoder lookback window --> len of TS
H = 6  # Decoder forecast horizon
MARGIN = 2.0  # Minimum margin above contract floor
SOFTMAX_TEMP = 5.0  # Sharpness of soft constraint


# ─── Alpha-shift: effective contract floor ───
def shift_contract_by_alpha(contract_seq):
    """
    The contract at time t doesn't affect sell
    price until time t + alpha. So the effective
    floor at forecast step t is:
        floor(t) = contract(t - alpha)

    We shift the known future contract sequence
    backward by alpha steps.
    """
    padding = tf.repeat(
        contract_seq[:, :1, :], ALPHA, axis=1
    )
    shifted = tf.concat(
        [padding, contract_seq[:, :-ALPHA, :]],
        axis=1
    )
    return shifted


# ─── Soft floor constraint (differentiable) ───
def apply_floor_constraint(inputs):
    """
    Ensures: sell_pred >= floor + margin

    During training, uses smooth approximation:
      softmax(a, b) = log(exp(τa) + exp(τb)) / τ

    This is differentiable everywhere, so gradients
    flow through the constraint. As τ → ∞, it
    approaches hard max().
    """
    raw_pred, shifted_contract = inputs
    floor = shifted_contract + MARGIN

    # LogSumExp smooth maximum
    stacked = tf.stack(
        [raw_pred * SOFTMAX_TEMP,
         floor * SOFTMAX_TEMP], axis=-1
    )
    constrained = (
            tf.reduce_logsumexp(stacked, axis=-1,
                                keepdims=True)
            / SOFTMAX_TEMP
    )
    return constrained


# ═══════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════
def build_model():
    # ─── ENCODER ───
    encoder_input = Input(
        shape=(T, 2), name='encoder_input'
    )
    # Input features: [sell_price, contract_value]

    enc_x = LSTM(
        128, return_sequences=True,
        name='enc_lstm_1'
    )(encoder_input)

    # return_state=True → we get h(T) and c(T)
    enc_out, enc_h, enc_c = LSTM(
        64, return_sequences=False,
        return_state=True,
        name='enc_lstm_2'
    )(enc_x)

    # enc_h, enc_c carry ALL encoded knowledge
    # about the sell-contract relationship,
    # the α-lag pattern, and temporal dynamics

    # ─── DECODER ───
    decoder_input = Input(
        shape=(H, 1), name='decoder_input'
    )
    # Known future contract values (H steps ahead)

    dec_x = LSTM(
        64, return_sequences=True,
        name='dec_lstm'
    )(decoder_input, initial_state=[enc_h, enc_c])
    #                ^^^^^^^^^^^^^^^^^^^^^^^^^^
    # CRITICAL: decoder starts from encoder's
    # final states — this is the "context bridge"

    # ─── RAW PREDICTION ───
    raw_pred = Dense(
        1, activation='linear',
        name='raw_prediction'
    )(dec_x)
    # Shape: (batch, H, 1) — unconstrained sell pred

    # ─── ALPHA-SHIFTED CONTRACT FLOOR ───
    shifted_contract = Lambda(
        shift_contract_by_alpha,
        name='alpha_shift'
    )(decoder_input)
    # Shape: (batch, H, 1) — floor(t) = contract(t-α)

    # ─── APPLY CONSTRAINT ───
    constrained_pred = Lambda(
        apply_floor_constraint,
        name='constrained_output'
    )([raw_pred, shifted_contract])
    # Shape: (batch, H, 1) — final sell price forecast
    # Guarantee: pred(t) >= contract(t-α) + margin

    # ─── ASSEMBLE MODEL ───
    model = Model(
        inputs=[encoder_input, decoder_input],
        outputs=constrained_pred,
        name='constrained_enc_dec_lstm'
    )


    # ─── CUSTOM LOSS ───
    def constrained_loss(y_true, y_pred):
        """
        Composite loss:
        1. MSE for accuracy
        2. Penalty for violating constraint
        """
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Extra penalty if prediction is below floor
        # (shouldn't happen with constraint layer,
        #  but adds gradient pressure during training)
        violation = tf.maximum(0.0, floor_value - y_pred)
        penalty = tf.reduce_mean(tf.square(violation))

        return mse + 10.0 * penalty


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss='mse',  # or constrained_loss
        metrics=['mae']
    )

    model.summary()

def main():
# Total params: ~133,441

# ═══════════════════════════════════════════════
# TRAINING DATA PREPARATION
# ═══════════════════════════════════════════════
#
# X_encoder: (samples, T, 2)
#   - [:, :, 0] = historical sell prices
#   - [:, :, 1] = historical contract values
#
# X_decoder: (samples, H, 1)
#   - [:, :, 0] = KNOWN future contract values
#   - (contracts are signed ahead of time!)
#
# Y: (samples, H, 1)
#   - [:, :, 0] = actual future sell prices
#
# model.fit(
#     [X_encoder, X_decoder], Y,
#     epochs=100, batch_size=32,
#     validation_split=0.2
# )

# X_encoder: (samples, len of window training (len of the TS ) , 2)

    X_encoder = 12
    build_model()

import os

model_input =  pd.read_parquet(r'model_input_forecasting.parquet', columns= ['year_month' , 'model_part_id' , 'is_make_part' ,'material_cost' , 'impact_purchase_cost'])
model_input =  model_input.loc[model_input['is_make_part'] == False]
encw_start = 202001
encw_end = 202512
enc_input = model_input.loc[(model_input['year_month'] >= encw_start) & (model_input['year_month'] <= encw_end) ]

print(model_input.head())

main()