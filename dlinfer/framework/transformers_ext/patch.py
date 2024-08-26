import transformers
import inspect
import importlib

def apply_model_patches(module):
    if module.__name__ == 'transformers_modules.internlm2-chat-7b.modeling_internlm2':
        from . import internlm2
        module.InternLM2RMSNorm.forward = internlm2.modeling_internlm2_InternLM2RMSNorm_forward
        module.InternLM2Attention.forward = internlm2.modeling_internlm2_InternLM2Attention_forward
        module.InternLM2ForCausalLM.prepare_inputs_for_generation = internlm2.modeling_internlm2_InternLM2ForCausalLM_prepare_inputs_for_generation
        transformers.cache_utils.DynamicCache.update = internlm2.transformers_cache_utils_dynamiccache_update
    elif module.__name__ == 'transformers_modules.modeling_internvl_chat':
        from . import internvl
        vit_module = inspect.getmodule(module.InternVisionModel)
        vit_module.InternAttention._naive_attn = internvl.InternAttention_naive_attn
        vit_module.InternRMSNorm.forward = internvl.InternRMSNorm_forward
    elif module.__name__ == 'transformers_modules.cogvlm-chat.modeling_cogvlm':
        from . import cogvlm
        vit_module = importlib.import_module('transformers_modules.cogvlm-chat.visual')
        vit_module.Attention.forward = cogvlm.PatchedAttentionForward

