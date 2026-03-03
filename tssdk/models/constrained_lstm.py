"""
tssdk.models.constrained_lstm — Encoder-Decoder LSTM with differentiable floor constraint.

Architecture Decision Record:
    Decision: Constrained Encoder-Decoder LSTM
    Status: Proposed (first experiment baseline)

    Context:
        - Multivariate forecasting: target (actual_cost_paid) with known-future
          covariate (contract_value).
        - Business constraint: target ≥ contract(t - α) + margin.
        - Alpha-lag: contract at time t affects target at time t + α.
        - Short series (72 monthly steps), 5 series — limited data regime.

    Decision Drivers:
        - Hard business constraint must be respected in predictions.
        - Encoder-decoder naturally separates history encoding from constrained decoding.
        - Known-future covariates feed directly into decoder.
        - LSTM chosen over Transformer due to limited data (< 500 steps per series).

    Consequences:
        - Positive: Constraint guarantees valid predictions; interpretable architecture.
        - Negative: LSTMs may underperform on long horizons vs attention.
        - Mitigation: Use as DL baseline; compare against N-HiTS + post-hoc clipping.

    Success Metrics:
        - Primary: MAE on test split
        - Guardrail: constraint violation rate = 0%
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from typing import Dict, Any

from tssdk.config import TimeseriesConfig
from tssdk.models.base import BaseTimeseriesModel
from tssdk.utils.sdk_logging import get_logger

logger = get_logger(__name__)


class ConstrainedLSTM(BaseTimeseriesModel):
    """Encoder-Decoder LSTM with differentiable constraint layer.

    Architecture:
        Encoder: 2-layer LSTM processes [target, covariates] history.
        Decoder: 1-layer LSTM with encoder state as initial context,
                 processes known-future covariates.
        Constraint: LogSumExp smooth max ensures pred ≥ floor + margin.

    Usage:
        >>> model_builder = ConstrainedLSTM(config)
        >>> model_builder.build()
        >>> keras_model = model_builder.get_model()
        >>> keras_model.fit([X_encoder, X_decoder], Y, ...)
    """

    def __init__(self, config: TimeseriesConfig):
        super().__init__(config)
        self._alpha = config.alpha_lag
        self._margin = config.margin
        self._temp = config.constraint_temp

    def build(self) -> None:
        """Construct the constrained encoder-decoder LSTM."""
        c = self.config
        T = c.encoder_length
        H = c.decoder_length
        n_enc_feat = c.n_encoder_features
        n_dec_feat = c.n_decoder_features
        alpha = self._alpha
        margin = self._margin
        temp = self._temp

        # ── ENCODER ──
        encoder_input = Input(shape=(T, n_enc_feat), name="encoder_input")

        enc_x = LSTM(
            c.encoder_hidden_1,
            return_sequences=True,
            name="enc_lstm_1",
        )(encoder_input)

        _, enc_h, enc_c = LSTM(
            c.encoder_hidden_2,
            return_sequences=False,
            return_state=True,
            name="enc_lstm_2",
        )(enc_x)

        # ── DECODER ──
        decoder_input = Input(shape=(H, n_dec_feat), name="decoder_input")

        dec_x = LSTM(
            c.decoder_hidden,
            return_sequences=True,
            name="dec_lstm",
        )(decoder_input, initial_state=[enc_h, enc_c])

        # ── RAW PREDICTION ──
        raw_pred = Dense(1, activation="linear", name="raw_prediction")(dec_x)

        # ── ALPHA-SHIFTED CONTRACT FLOOR ──
        def shift_contract_by_alpha(contract_seq):
            """Shift contract backward by α steps: floor(t) = contract(t - α)."""
            padding = tf.repeat(contract_seq[:, :1, :], alpha, axis=1)
            shifted = tf.concat(
                [padding, contract_seq[:, :-alpha, :]], axis=1
            )
            return shifted

        def apply_floor_constraint(inputs):
            """Differentiable floor: pred ≥ shifted_contract + margin.

            Uses LogSumExp smooth max:
                softmax(a, b) = log(exp(τa) + exp(τb)) / τ
            As τ → ∞, approaches hard max().
            """
            raw, shifted_contract = inputs
            floor = shifted_contract + margin

            # Squeeze to (batch, H) for clean stacking
            raw_sq = tf.squeeze(raw, axis=-1)
            floor_sq = tf.squeeze(floor, axis=-1)

            stacked = tf.stack(
                [raw_sq * temp, floor_sq * temp], axis=-1
            )
            constrained = (
                tf.reduce_logsumexp(stacked, axis=-1) / temp
            )
            # Restore to (batch, H, 1)
            return tf.expand_dims(constrained, axis=-1)

        # Only apply constraint if alpha > 0
        if alpha > 0:
            shifted_contract = Lambda(
                shift_contract_by_alpha, name="alpha_shift"
            )(decoder_input)

            constrained_pred = Lambda(
                apply_floor_constraint, name="constrained_output"
            )([raw_pred, shifted_contract])
        else:
            # No lag constraint — just use raw predictions
            constrained_pred = raw_pred

        # ── ASSEMBLE ──
        self.model = Model(
            inputs=[encoder_input, decoder_input],
            outputs=constrained_pred,
            name="constrained_enc_dec_lstm",
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=c.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        param_count = self.model.count_params()
        logger.info(
            f"Built ConstrainedLSTM | params={param_count:,} | "
            f"encoder=({T},{n_enc_feat}) | decoder=({H},{n_dec_feat}) | "
            f"alpha={alpha} | margin={margin}"
        )

    def get_model(self) -> Model:
        """Return the compiled Keras model.

        Raises:
            RuntimeError: If build() hasn't been called.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not built. Call build() first. "
                "Example: model_builder.build(); model = model_builder.get_model()"
            )
        return self.model

    def describe(self) -> Dict[str, Any]:
        """Return architecture metadata for experiment logging."""
        return {
            "name": "ConstrainedEncoderDecoderLSTM",
            "encoder_layers": 2,
            "decoder_layers": 1,
            "encoder_hidden": [self.config.encoder_hidden_1, self.config.encoder_hidden_2],
            "decoder_hidden": self.config.decoder_hidden,
            "alpha_lag": self._alpha,
            "margin": self._margin,
            "constraint_temp": self._temp,
            "constraint_type": "logsumexp_smooth_max",
            "total_params": self.model.count_params() if self.model else None,
        }
