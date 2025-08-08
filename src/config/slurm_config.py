from dataclasses import dataclass


@dataclass
class SlurmConfig:
    "Default settings used in override_config"
    model_type: str = "gpt2-small"

    d_model: int = 768
    n_layers: int = 12
    use_pretrained_weights: bool = False


    attention_n_heads: int = 12

    activation_hidden_dim: int = 3072

    norm_eps: float = 1e-6

    position_emb_theta: float = 10000.0
