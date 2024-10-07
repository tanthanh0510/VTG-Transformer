from transformers.configuration_utils import PretrainedConfig

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
    "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
    "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/config.json",
}


class GPT2Config(PretrainedConfig):
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=512,
        hidden_size=768,
        num_hidden_layers=3,
        num_attention_heads=8,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=0,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_cross_attention = kwargs.pop("add_cross_attention", True)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }
