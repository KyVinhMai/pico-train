"""
Enhanced GPT-2 implementation with RoPE support and proper HuggingFace conversion.
This fixes the convert_to_hf_model issue by making GPT-2 models compatible with the checkpointing system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block
from src.config import ModelConfig
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

class RoPE(nn.Module):
    """Rotary Positional Embeddings (RoPE) - from Pico Decoder implementation.
    
    Implements position-dependent rotation of keys and queries in attention mechanism,
    allowing better modeling of relative positions in sequences. Uses complex number
    operations for efficient rotation.
    
    Args:
        config: Model configuration containing:
            - config.position_emb_theta: Base for frequency computation
            - config.d_model: Model dimension
            - config.attention_n_heads: Number of attention heads
            - config.max_seq_len: Maximum sequence length
            
    References:
        https://arxiv.org/abs/2104.09864
    """

    _freqs_cis_tensor: torch.Tensor | None = None

    def __init__(self, config):
        super().__init__()

        self.theta = getattr(config, 'position_emb_theta', 10000.0)
        self.dim = config.hidden_size // config.num_attention_heads

        max_seq_len = getattr(config, 'n_positions', 2048)

        # only gets set once, and then reused for all RoPE instances
        if RoPE._freqs_cis_tensor is None:
            RoPE._freqs_cis_tensor = self._setup_freqs_cis(
                max_seq_len, self.theta, self.dim
            )

        # register _freqs_cis buffer
        # can be easily recomputed so persistent=False
        self.register_buffer("_freqs_cis", self._freqs_cis_tensor, persistent=False)

    @classmethod
    def _setup_freqs_cis(cls, seq_len: int, theta: float, dim: int) -> torch.Tensor:
        """Setup Frequency Tensor for RoPE Embeddings

        Initializes the complex frequency tensor that is used to compute the RoPE embeddings.

        Note other implementations will use cos and sin directly, but using the complex
        number representation is (probably?) more efficient:

            e^(theta * i * t) = cos(theta * t) + i * sin(theta * t) [Euler's formula]
        """
        _freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        positions = torch.arange(seq_len)
        freqs = torch.outer(positions, _freqs)
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def get_freqs_cis(
        self, input_shape: torch.Size, start_pos: int, end_pos: int
    ) -> torch.Tensor:
        """Reshape Frequency Tensor for RoPE Embeddings

        Makes the frequency tensor broadcastable with the input tensor.
        """
        _freqs_cis = self._freqs_cis[start_pos:end_pos]
        ndim = len(input_shape)
        assert 0 <= 1 < ndim
        assert _freqs_cis.shape == (input_shape[1], input_shape[-1])

        # TODO: Check whether this is correct (might be able to remove this)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(input_shape)]
        return _freqs_cis.view(*shape)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE Embeddings to Queries and Keys

        Applies the rotary positional embeddings to the input tensors via complex num multiplication

        NOTE: The start_pos is used if we want to use the kv_cache in the attention mechanism.
        """
        queries_ = torch.view_as_complex(
            queries.float().reshape(*queries.shape[:-1], -1, 2)
        )
        keys_ = torch.view_as_complex(keys.float().reshape(*keys.shape[:-1], -1, 2))

        input_shape = (
            queries_.shape
        )  # same as keys: (batch_size, seq_len, n_heads, head_dim/2)
        freqs_start_pos = start_pos
        freqs_end_pos = freqs_start_pos + queries_.shape[1]

        freqs_cis = self.get_freqs_cis(input_shape, freqs_start_pos, freqs_end_pos)

        queries_rotated = torch.view_as_real(queries_ * freqs_cis).flatten(3)
        keys_rotated = torch.view_as_real(keys_ * freqs_cis).flatten(3)
        return queries_rotated.type_as(queries), keys_rotated.type_as(keys)


class StandardModelConfig(ModelConfig):
    """Configuration for standard models like GPT-2 with enhanced options."""
    
    use_pretrained_weights: bool = False
    gpt2_variant: str = "gpt2"  
    config_type: str = "standard"
    
    # Positional embedding options
    positional_embedding_type: str = "learned"  # Options: "learned", "rope"
    position_emb_theta: float = 10000.0 
    
    # Override some defaults for GPT-2 compatibility
    attention_n_kv_heads: int = None  # GPT-2 doesn't use grouped query attention
    
    # Additional GPT-2 specific settings
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02


class GPT2RoPEAttention(nn.Module):
    """GPT-2 Attention layer with RoPE (Rotary Positional Embedding) support."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.scale_attn_weights = True
        
        # Query, Key, Value projections
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # RoPE setup using Pico Decoder's implementation
        self.rope = RoPE(config)
        
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Splits hidden_size dim into attn_head_size and num_heads"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merges attn_head_size dim and num_attn_heads dim into hidden_size"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Apply RoPE to queries and keys using Pico Decoder's implementation
        start_pos = 0
        if hasattr(self, '_kv_cache_seq_len'):
            start_pos = self._kv_cache_seq_len
        
        query, key = self.rope(query, key, start_pos=start_pos)
        
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        
        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
            # Track sequence length for RoPE start_pos
            self._kv_cache_seq_len = past_key.shape[-2]
        else:
            self._kv_cache_seq_len = 0
        
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs


class GPT2RoPEBlock(GPT2Block):
    """GPT-2 transformer block with RoPE attention."""
    
    def __init__(self, config):
        super().__init__(config)
        self.attn = GPT2RoPEAttention(config)


class GPT2RoPEModel(GPT2Model):
    """GPT-2 model with RoPE positional embeddings instead of learned positions."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Remove the position embeddings since we're using RoPE
        del self.wpe  # Remove learned positional embeddings
        
        # Replace transformer blocks with RoPE blocks
        self.h = nn.ModuleList([GPT2RoPEBlock(config) for _ in range(config.num_hidden_layers)])
        
        # Re-initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,  # This will be ignored since we use RoPE
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # No position_ids needed for RoPE, but we keep this for compatibility
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        # Attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        # Head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Embedding
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # Note: No position embeddings added here since we use RoPE
        hidden_states = inputs_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        
        hidden_states = self.drop(hidden_states)
        
        output_shape = input_shape + (hidden_states.size(-1),)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            
            if output_attentions:
                all_self_attentions = all_self_attententions + (outputs[2 if use_cache else 1],)
        
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class GPT2WithConversion(GPT2LMHeadModel):
    """Standard GPT-2 Language Model with convert_to_hf_model method for compatibility."""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config  # Store config for conversion
        
        # Re-initialize weights
        self.post_init()
    
    def convert_to_hf_model(self):
        """Convert to HuggingFace model (already is one, so return self)."""
        return self


class GPT2RoPELMHeadModel(GPT2LMHeadModel):
    """GPT-2 Language Model with RoPE positional embeddings."""
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2RoPEModel(config)
        self.config = config  # Store config for conversion
        
        # Re-initialize weights
        self.post_init()
    
    def convert_to_hf_model(self):
        """Convert to HuggingFace model (already is one, so return self)."""
        return self


def initialize_gpt2_model(model_config: ModelConfig):
    """Initialize a GPT-2 model with optional RoPE support."""
    
    # Create GPT-2 configuration
    gpt2_config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.max_seq_len,
        n_embd=model_config.d_model,
        n_layer=model_config.n_layers,
        n_head=model_config.attention_n_heads,
        n_inner=model_config.activation_hidden_dim,
        activation_function="gelu_new",
        resid_pdrop=getattr(model_config, 'residual_dropout', 0.1),
        embd_pdrop=getattr(model_config, 'embedding_dropout', 0.1),
        attn_pdrop=getattr(model_config, 'attention_dropout', 0.1),
        layer_norm_epsilon=float(model_config.norm_eps),
        initializer_range=getattr(model_config, 'initializer_range', 0.02),
        use_cache=False,  # Disable for training
        bos_token_id=50256,
        eos_token_id=50256,
    )
    
    # Add RoPE-specific config if using RoPE
    positional_embedding_type = getattr(model_config, 'positional_embedding_type', 'learned')
    if positional_embedding_type == 'rope':
        gpt2_config.position_emb_theta = getattr(model_config, 'position_emb_theta', 10000.0)
    
    use_pretrained = getattr(model_config, 'use_pretrained_weights', False)
    gpt2_variant = getattr(model_config, 'gpt2_variant', 'gpt2')
    
    if use_pretrained and positional_embedding_type == 'rope':
        raise ValueError("Cannot use pretrained weights with RoPE. RoPE requires training from scratch.")
    
    if use_pretrained:
        # Load pretrained GPT-2 model (standard positional embeddings)
        model = GPT2WithConversion.from_pretrained(
            gpt2_variant,
            config=gpt2_config,
            ignore_mismatched_sizes=True
        )
        print(f"✅ Loaded pretrained GPT-2 model: {gpt2_variant}")
        
    elif positional_embedding_type == 'rope':
        # Initialize GPT-2 model with RoPE from scratch
        model = GPT2RoPELMHeadModel(gpt2_config)
        _initialize_gpt2_weights(model)
        print(f"✅ Initialized GPT-2 model with RoPE from scratch")
        print(f"   - Using Rotary Positional Embeddings")
        print(f"   - RoPE theta: {gpt2_config.position_emb_theta}")
        print(f"   - Max positions: {gpt2_config.n_positions}")
        
    else:
        # Initialize standard GPT-2 model from scratch
        model = GPT2WithConversion(gpt2_config)
        _initialize_gpt2_weights(model)
        print(f"✅ Initialized GPT-2 model from scratch")
        print(f"   - Using learned positional embeddings")
    
    print(f"   - Model size: {gpt2_variant}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Vocab size: {gpt2_config.vocab_size}")
    print(f"   - Sequence length: {gpt2_config.n_positions}")
    print(f"   - Positional embedding: {positional_embedding_type}")
    
    return model


def _initialize_gpt2_weights(model):
    """Apply GPT-2 style weight initialization to the model."""
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    # Apply initialization to all modules
    model.apply(_init_weights)
    
    # Special initialization for the output projection (following GPT-2 paper)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("c_proj" in name or "mlp.c_proj" in name):
            torch.nn.init.normal_(
                module.weight, 
                mean=0.0, 
                std=0.02 / (2 * model.config.n_layer) ** 0.5
            )