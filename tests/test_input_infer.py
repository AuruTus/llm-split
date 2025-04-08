import ast
import astunparse
from llm_split.model_meta import ForwardMeta
from llm_split.namespace import NameSpace
from llm_split.var_extract import VariableVisitor, AttrFilter, GlobalVarFilter, BasicTypeFilter

from transformers import LlamaModel

code_1 = """
output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
output_hidden_states = (
    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
)
use_cache = use_cache if use_cache is not None else self.config.use_cache
return_dict = return_dict if return_dict is not None else self.config.use_return_dict

if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError(
        "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
    )

if self.gradient_checkpointing and self.training and use_cache:
    logger.warning_once(
        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
    )
    use_cache = False

if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)

past_seen_tokens = 0
if use_cache:  # kept for BC (cache positions)
    if not isinstance(past_key_values, StaticCache):
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_seen_tokens = past_key_values.get_seq_length()

if cache_position is None:
    if isinstance(past_key_values, StaticCache):
        raise ValueError("cache_position is a required argument when using StaticCache.")
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )

if position_ids is None:
    position_ids = cache_position.unsqueeze(0)

causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

# embed positions
hidden_states = inputs_embeds

# decoder layers
all_hidden_states = () if output_hidden_states else None
all_self_attns = () if output_attentions else None
next_decoder_cache = None

for decoder_layer in self.layers:
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if self.gradient_checkpointing and self.training:
        layer_outputs = self._gradient_checkpointing_func(
            decoder_layer.__call__,
            hidden_states,
            causal_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
            cache_position,
        )
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    hidden_states = layer_outputs[0]

    if use_cache:
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    if output_attentions:
        all_self_attns += (layer_outputs[1],)

hidden_states = self.norm(hidden_states)

# add hidden states from the last decoder layer
if output_hidden_states:
    all_hidden_states += (hidden_states,)

next_cache = None
if use_cache:
    next_cache = (
        next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
    )
if not return_dict:
    return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
return BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=next_cache,
    hidden_states=all_hidden_states,
    attentions=all_self_attns,
)
"""


def _analyze_code_segments(code_1: str) -> ForwardMeta:
    # Parse both code segments

    stmt_1 = ast.parse(code_1)

    # Create visitors for both code segments
    visitor_1 = VariableVisitor()

    # Analyze both code segments
    visitor_1.visit(stmt_1)

    print(visitor_1.assigned_vars)
    print(visitor_1.read_vars)
    print(visitor_1.read_before_assigned)

    input_filter = AttrFilter(visitor_1.read_before_assigned)
    print("====================")
    print(input_filter.self_attr)
    print("====================")
    print(input_filter.var)

    global_filter = GlobalVarFilter(input_filter.var, NameSpace.namespace_of(LlamaModel))
    print("====================")
    print(global_filter.global_var)
    print("====================")
    print(global_filter.var)

    type_filter = BasicTypeFilter(global_filter.var)
    print("====================")
    print(type_filter.basic_type)
    print("====================")
    print(type_filter.var)


def test_code():
    # Test with your example inputs
    _analyze_code_segments(code_1)
