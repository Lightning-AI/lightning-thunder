import pytest
import torch
import thunder
import einops


@pytest.mark.benchmark(
    group="neva",
    warmup_iterations=1,  # takes a long time to run.
)
def g1():
    class DynamoModule(torch.nn.Module):
        def forward(
            self,
            L_vision_x_: torch.Tensor,
            L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_parameters_class_embedding_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_weight_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_bias_: torch.nn.parameter.Parameter,
            L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_buffers_position_ids_: torch.Tensor,
        ):
            l_vision_x_ = L_vision_x_
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_parameters_class_embedding_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_parameters_class_embedding_
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_weight_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_weight_
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_bias_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_bias_
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_ = L_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_
            l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_weight_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_weight_
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_bias_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_bias_
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_buffers_position_ids_ = (
                L_self_modules_vision_encoder_modules_vision_model_modules_embeddings_buffers_position_ids_
            )
            vision_x = einops.einops.rearrange(l_vision_x_, "b T F c h w -> (b T F) c h w")
            l_vision_x_ = None
            vision_x_1 = vision_x.to(torch.bfloat16)
            vision_x = None
            to_1 = vision_x_1.to(dtype=torch.bfloat16)
            vision_x_1 = None
            patch_embeds = torch.conv2d(
                to_1,
                l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_,
                None,
                (14, 14),
                (0, 0),
                (1, 1),
                1,
            )
            to_1 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_patch_embedding_parameters_weight_
            ) = None
            flatten = patch_embeds.flatten(2)
            patch_embeds = None
            patch_embeds_1 = flatten.transpose(1, 2)
            flatten = None
            class_embeds = l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_parameters_class_embedding_.expand(
                2, 1, -1
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_parameters_class_embedding_ = None
            embeddings = torch.cat([class_embeds, patch_embeds_1], dim=1)
            class_embeds = patch_embeds_1 = None
            embedding = torch.nn.functional.embedding(
                l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_buffers_position_ids_,
                l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_,
                None,
                None,
                2.0,
                False,
                False,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_buffers_position_ids_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_embeddings_modules_position_embedding_parameters_weight_
            ) = None
            embeddings_1 = embeddings + embedding
            embeddings = embedding = None
            hidden_states = torch.nn.functional.layer_norm(
                embeddings_1,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_bias_,
                1e-05,
            )
            embeddings_1 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_weight_
            ) = l_self_modules_vision_encoder_modules_vision_model_modules_pre_layrnorm_parameters_bias_ = None
            hidden_states_1 = torch.nn.functional.layer_norm(
                hidden_states,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm1_parameters_bias_
            ) = None
            linear = torch._C._nn.linear(
                hidden_states_1,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states = linear * 0.125
            linear = None
            linear_1 = torch._C._nn.linear(
                hidden_states_1,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view = linear_1.view(2, -1, 16, 64)
            linear_1 = None
            transpose_1 = view.transpose(1, 2)
            view = None
            key_states = transpose_1.contiguous()
            transpose_1 = None
            linear_2 = torch._C._nn.linear(
                hidden_states_1,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_1 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_1 = linear_2.view(2, -1, 16, 64)
            linear_2 = None
            transpose_2 = view_1.transpose(1, 2)
            view_1 = None
            value_states = transpose_2.contiguous()
            transpose_2 = None
            view_2 = query_states.view(2, 257, 16, 64)
            query_states = None
            transpose_3 = view_2.transpose(1, 2)
            view_2 = None
            contiguous_2 = transpose_3.contiguous()
            transpose_3 = None
            query_states_1 = contiguous_2.view(32, -1, 64)
            contiguous_2 = None
            key_states_1 = key_states.view(32, -1, 64)
            key_states = None
            value_states_1 = value_states.view(32, -1, 64)
            value_states = None
            transpose_4 = key_states_1.transpose(1, 2)
            key_states_1 = None
            attn_weights = torch.bmm(query_states_1, transpose_4)
            query_states_1 = transpose_4 = None
            attn_weights_1 = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = None
            attn_probs = torch.nn.functional.dropout(attn_weights_1, p=0.0, training=False)
            attn_weights_1 = None
            attn_output = torch.bmm(attn_probs, value_states_1)
            attn_probs = value_states_1 = None
            attn_output_1 = attn_output.view(2, 16, 257, 64)
            attn_output = None
            attn_output_2 = attn_output_1.transpose(1, 2)
            attn_output_1 = None
            attn_output_3 = attn_output_2.reshape(2, 257, 1024)
            attn_output_2 = None
            attn_output_4 = torch._C._nn.linear(
                attn_output_3,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_3 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_2 = hidden_states + attn_output_4
            hidden_states = attn_output_4 = None
            hidden_states_3 = torch.nn.functional.layer_norm(
                hidden_states_2,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_4 = torch._C._nn.linear(
                hidden_states_3,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_3 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_1 = 1.702 * hidden_states_4
            sigmoid = torch.sigmoid(mul_1)
            mul_1 = None
            hidden_states_5 = hidden_states_4 * sigmoid
            hidden_states_4 = sigmoid = None
            hidden_states_6 = torch._C._nn.linear(
                hidden_states_5,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_5 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_0_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_7 = hidden_states_2 + hidden_states_6
            hidden_states_2 = hidden_states_6 = None
            hidden_states_8 = torch.nn.functional.layer_norm(
                hidden_states_7,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm1_parameters_bias_
            ) = None
            linear_6 = torch._C._nn.linear(
                hidden_states_8,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_2 = linear_6 * 0.125
            linear_6 = None
            linear_7 = torch._C._nn.linear(
                hidden_states_8,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_7 = linear_7.view(2, -1, 16, 64)
            linear_7 = None
            transpose_6 = view_7.transpose(1, 2)
            view_7 = None
            key_states_2 = transpose_6.contiguous()
            transpose_6 = None
            linear_8 = torch._C._nn.linear(
                hidden_states_8,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_8 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_8 = linear_8.view(2, -1, 16, 64)
            linear_8 = None
            transpose_7 = view_8.transpose(1, 2)
            view_8 = None
            value_states_2 = transpose_7.contiguous()
            transpose_7 = None
            view_9 = query_states_2.view(2, 257, 16, 64)
            query_states_2 = None
            transpose_8 = view_9.transpose(1, 2)
            view_9 = None
            contiguous_5 = transpose_8.contiguous()
            transpose_8 = None
            query_states_3 = contiguous_5.view(32, -1, 64)
            contiguous_5 = None
            key_states_3 = key_states_2.view(32, -1, 64)
            key_states_2 = None
            value_states_3 = value_states_2.view(32, -1, 64)
            value_states_2 = None
            transpose_9 = key_states_3.transpose(1, 2)
            key_states_3 = None
            attn_weights_2 = torch.bmm(query_states_3, transpose_9)
            query_states_3 = transpose_9 = None
            attn_weights_3 = torch.nn.functional.softmax(attn_weights_2, dim=-1)
            attn_weights_2 = None
            attn_probs_1 = torch.nn.functional.dropout(attn_weights_3, p=0.0, training=False)
            attn_weights_3 = None
            attn_output_5 = torch.bmm(attn_probs_1, value_states_3)
            attn_probs_1 = value_states_3 = None
            attn_output_6 = attn_output_5.view(2, 16, 257, 64)
            attn_output_5 = None
            attn_output_7 = attn_output_6.transpose(1, 2)
            attn_output_6 = None
            attn_output_8 = attn_output_7.reshape(2, 257, 1024)
            attn_output_7 = None
            attn_output_9 = torch._C._nn.linear(
                attn_output_8,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_8 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_9 = hidden_states_7 + attn_output_9
            hidden_states_7 = attn_output_9 = None
            hidden_states_10 = torch.nn.functional.layer_norm(
                hidden_states_9,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_11 = torch._C._nn.linear(
                hidden_states_10,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_10 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_4 = 1.702 * hidden_states_11
            sigmoid_1 = torch.sigmoid(mul_4)
            mul_4 = None
            hidden_states_12 = hidden_states_11 * sigmoid_1
            hidden_states_11 = sigmoid_1 = None
            hidden_states_13 = torch._C._nn.linear(
                hidden_states_12,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_12 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_1_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_14 = hidden_states_9 + hidden_states_13
            hidden_states_9 = hidden_states_13 = None
            hidden_states_15 = torch.nn.functional.layer_norm(
                hidden_states_14,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm1_parameters_bias_
            ) = None
            linear_12 = torch._C._nn.linear(
                hidden_states_15,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_4 = linear_12 * 0.125
            linear_12 = None
            linear_13 = torch._C._nn.linear(
                hidden_states_15,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_14 = linear_13.view(2, -1, 16, 64)
            linear_13 = None
            transpose_11 = view_14.transpose(1, 2)
            view_14 = None
            key_states_4 = transpose_11.contiguous()
            transpose_11 = None
            linear_14 = torch._C._nn.linear(
                hidden_states_15,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_15 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_15 = linear_14.view(2, -1, 16, 64)
            linear_14 = None
            transpose_12 = view_15.transpose(1, 2)
            view_15 = None
            value_states_4 = transpose_12.contiguous()
            transpose_12 = None
            view_16 = query_states_4.view(2, 257, 16, 64)
            query_states_4 = None
            transpose_13 = view_16.transpose(1, 2)
            view_16 = None
            contiguous_8 = transpose_13.contiguous()
            transpose_13 = None
            query_states_5 = contiguous_8.view(32, -1, 64)
            contiguous_8 = None
            key_states_5 = key_states_4.view(32, -1, 64)
            key_states_4 = None
            value_states_5 = value_states_4.view(32, -1, 64)
            value_states_4 = None
            transpose_14 = key_states_5.transpose(1, 2)
            key_states_5 = None
            attn_weights_4 = torch.bmm(query_states_5, transpose_14)
            query_states_5 = transpose_14 = None
            attn_weights_5 = torch.nn.functional.softmax(attn_weights_4, dim=-1)
            attn_weights_4 = None
            attn_probs_2 = torch.nn.functional.dropout(attn_weights_5, p=0.0, training=False)
            attn_weights_5 = None
            attn_output_10 = torch.bmm(attn_probs_2, value_states_5)
            attn_probs_2 = value_states_5 = None
            attn_output_11 = attn_output_10.view(2, 16, 257, 64)
            attn_output_10 = None
            attn_output_12 = attn_output_11.transpose(1, 2)
            attn_output_11 = None
            attn_output_13 = attn_output_12.reshape(2, 257, 1024)
            attn_output_12 = None
            attn_output_14 = torch._C._nn.linear(
                attn_output_13,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_13 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_16 = hidden_states_14 + attn_output_14
            hidden_states_14 = attn_output_14 = None
            hidden_states_17 = torch.nn.functional.layer_norm(
                hidden_states_16,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_18 = torch._C._nn.linear(
                hidden_states_17,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_17 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_7 = 1.702 * hidden_states_18
            sigmoid_2 = torch.sigmoid(mul_7)
            mul_7 = None
            hidden_states_19 = hidden_states_18 * sigmoid_2
            hidden_states_18 = sigmoid_2 = None
            hidden_states_20 = torch._C._nn.linear(
                hidden_states_19,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_19 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_2_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_21 = hidden_states_16 + hidden_states_20
            hidden_states_16 = hidden_states_20 = None
            hidden_states_22 = torch.nn.functional.layer_norm(
                hidden_states_21,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm1_parameters_bias_
            ) = None
            linear_18 = torch._C._nn.linear(
                hidden_states_22,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_6 = linear_18 * 0.125
            linear_18 = None
            linear_19 = torch._C._nn.linear(
                hidden_states_22,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_21 = linear_19.view(2, -1, 16, 64)
            linear_19 = None
            transpose_16 = view_21.transpose(1, 2)
            view_21 = None
            key_states_6 = transpose_16.contiguous()
            transpose_16 = None
            linear_20 = torch._C._nn.linear(
                hidden_states_22,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_22 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_22 = linear_20.view(2, -1, 16, 64)
            linear_20 = None
            transpose_17 = view_22.transpose(1, 2)
            view_22 = None
            value_states_6 = transpose_17.contiguous()
            transpose_17 = None
            view_23 = query_states_6.view(2, 257, 16, 64)
            query_states_6 = None
            transpose_18 = view_23.transpose(1, 2)
            view_23 = None
            contiguous_11 = transpose_18.contiguous()
            transpose_18 = None
            query_states_7 = contiguous_11.view(32, -1, 64)
            contiguous_11 = None
            key_states_7 = key_states_6.view(32, -1, 64)
            key_states_6 = None
            value_states_7 = value_states_6.view(32, -1, 64)
            value_states_6 = None
            transpose_19 = key_states_7.transpose(1, 2)
            key_states_7 = None
            attn_weights_6 = torch.bmm(query_states_7, transpose_19)
            query_states_7 = transpose_19 = None
            attn_weights_7 = torch.nn.functional.softmax(attn_weights_6, dim=-1)
            attn_weights_6 = None
            attn_probs_3 = torch.nn.functional.dropout(attn_weights_7, p=0.0, training=False)
            attn_weights_7 = None
            attn_output_15 = torch.bmm(attn_probs_3, value_states_7)
            attn_probs_3 = value_states_7 = None
            attn_output_16 = attn_output_15.view(2, 16, 257, 64)
            attn_output_15 = None
            attn_output_17 = attn_output_16.transpose(1, 2)
            attn_output_16 = None
            attn_output_18 = attn_output_17.reshape(2, 257, 1024)
            attn_output_17 = None
            attn_output_19 = torch._C._nn.linear(
                attn_output_18,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_18 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_23 = hidden_states_21 + attn_output_19
            hidden_states_21 = attn_output_19 = None
            hidden_states_24 = torch.nn.functional.layer_norm(
                hidden_states_23,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_25 = torch._C._nn.linear(
                hidden_states_24,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_24 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_10 = 1.702 * hidden_states_25
            sigmoid_3 = torch.sigmoid(mul_10)
            mul_10 = None
            hidden_states_26 = hidden_states_25 * sigmoid_3
            hidden_states_25 = sigmoid_3 = None
            hidden_states_27 = torch._C._nn.linear(
                hidden_states_26,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_26 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_3_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_28 = hidden_states_23 + hidden_states_27
            hidden_states_23 = hidden_states_27 = None
            hidden_states_29 = torch.nn.functional.layer_norm(
                hidden_states_28,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm1_parameters_bias_
            ) = None
            linear_24 = torch._C._nn.linear(
                hidden_states_29,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_8 = linear_24 * 0.125
            linear_24 = None
            linear_25 = torch._C._nn.linear(
                hidden_states_29,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_28 = linear_25.view(2, -1, 16, 64)
            linear_25 = None
            transpose_21 = view_28.transpose(1, 2)
            view_28 = None
            key_states_8 = transpose_21.contiguous()
            transpose_21 = None
            linear_26 = torch._C._nn.linear(
                hidden_states_29,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_29 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_29 = linear_26.view(2, -1, 16, 64)
            linear_26 = None
            transpose_22 = view_29.transpose(1, 2)
            view_29 = None
            value_states_8 = transpose_22.contiguous()
            transpose_22 = None
            view_30 = query_states_8.view(2, 257, 16, 64)
            query_states_8 = None
            transpose_23 = view_30.transpose(1, 2)
            view_30 = None
            contiguous_14 = transpose_23.contiguous()
            transpose_23 = None
            query_states_9 = contiguous_14.view(32, -1, 64)
            contiguous_14 = None
            key_states_9 = key_states_8.view(32, -1, 64)
            key_states_8 = None
            value_states_9 = value_states_8.view(32, -1, 64)
            value_states_8 = None
            transpose_24 = key_states_9.transpose(1, 2)
            key_states_9 = None
            attn_weights_8 = torch.bmm(query_states_9, transpose_24)
            query_states_9 = transpose_24 = None
            attn_weights_9 = torch.nn.functional.softmax(attn_weights_8, dim=-1)
            attn_weights_8 = None
            attn_probs_4 = torch.nn.functional.dropout(attn_weights_9, p=0.0, training=False)
            attn_weights_9 = None
            attn_output_20 = torch.bmm(attn_probs_4, value_states_9)
            attn_probs_4 = value_states_9 = None
            attn_output_21 = attn_output_20.view(2, 16, 257, 64)
            attn_output_20 = None
            attn_output_22 = attn_output_21.transpose(1, 2)
            attn_output_21 = None
            attn_output_23 = attn_output_22.reshape(2, 257, 1024)
            attn_output_22 = None
            attn_output_24 = torch._C._nn.linear(
                attn_output_23,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_23 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_30 = hidden_states_28 + attn_output_24
            hidden_states_28 = attn_output_24 = None
            hidden_states_31 = torch.nn.functional.layer_norm(
                hidden_states_30,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_32 = torch._C._nn.linear(
                hidden_states_31,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_31 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_13 = 1.702 * hidden_states_32
            sigmoid_4 = torch.sigmoid(mul_13)
            mul_13 = None
            hidden_states_33 = hidden_states_32 * sigmoid_4
            hidden_states_32 = sigmoid_4 = None
            hidden_states_34 = torch._C._nn.linear(
                hidden_states_33,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_33 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_4_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_35 = hidden_states_30 + hidden_states_34
            hidden_states_30 = hidden_states_34 = None
            hidden_states_36 = torch.nn.functional.layer_norm(
                hidden_states_35,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm1_parameters_bias_
            ) = None
            linear_30 = torch._C._nn.linear(
                hidden_states_36,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_10 = linear_30 * 0.125
            linear_30 = None
            linear_31 = torch._C._nn.linear(
                hidden_states_36,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_35 = linear_31.view(2, -1, 16, 64)
            linear_31 = None
            transpose_26 = view_35.transpose(1, 2)
            view_35 = None
            key_states_10 = transpose_26.contiguous()
            transpose_26 = None
            linear_32 = torch._C._nn.linear(
                hidden_states_36,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_36 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_36 = linear_32.view(2, -1, 16, 64)
            linear_32 = None
            transpose_27 = view_36.transpose(1, 2)
            view_36 = None
            value_states_10 = transpose_27.contiguous()
            transpose_27 = None
            view_37 = query_states_10.view(2, 257, 16, 64)
            query_states_10 = None
            transpose_28 = view_37.transpose(1, 2)
            view_37 = None
            contiguous_17 = transpose_28.contiguous()
            transpose_28 = None
            query_states_11 = contiguous_17.view(32, -1, 64)
            contiguous_17 = None
            key_states_11 = key_states_10.view(32, -1, 64)
            key_states_10 = None
            value_states_11 = value_states_10.view(32, -1, 64)
            value_states_10 = None
            transpose_29 = key_states_11.transpose(1, 2)
            key_states_11 = None
            attn_weights_10 = torch.bmm(query_states_11, transpose_29)
            query_states_11 = transpose_29 = None
            attn_weights_11 = torch.nn.functional.softmax(attn_weights_10, dim=-1)
            attn_weights_10 = None
            attn_probs_5 = torch.nn.functional.dropout(attn_weights_11, p=0.0, training=False)
            attn_weights_11 = None
            attn_output_25 = torch.bmm(attn_probs_5, value_states_11)
            attn_probs_5 = value_states_11 = None
            attn_output_26 = attn_output_25.view(2, 16, 257, 64)
            attn_output_25 = None
            attn_output_27 = attn_output_26.transpose(1, 2)
            attn_output_26 = None
            attn_output_28 = attn_output_27.reshape(2, 257, 1024)
            attn_output_27 = None
            attn_output_29 = torch._C._nn.linear(
                attn_output_28,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_28 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_37 = hidden_states_35 + attn_output_29
            hidden_states_35 = attn_output_29 = None
            hidden_states_38 = torch.nn.functional.layer_norm(
                hidden_states_37,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_39 = torch._C._nn.linear(
                hidden_states_38,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_38 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_16 = 1.702 * hidden_states_39
            sigmoid_5 = torch.sigmoid(mul_16)
            mul_16 = None
            hidden_states_40 = hidden_states_39 * sigmoid_5
            hidden_states_39 = sigmoid_5 = None
            hidden_states_41 = torch._C._nn.linear(
                hidden_states_40,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_40 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_5_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_42 = hidden_states_37 + hidden_states_41
            hidden_states_37 = hidden_states_41 = None
            hidden_states_43 = torch.nn.functional.layer_norm(
                hidden_states_42,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm1_parameters_bias_
            ) = None
            linear_36 = torch._C._nn.linear(
                hidden_states_43,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_12 = linear_36 * 0.125
            linear_36 = None
            linear_37 = torch._C._nn.linear(
                hidden_states_43,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_42 = linear_37.view(2, -1, 16, 64)
            linear_37 = None
            transpose_31 = view_42.transpose(1, 2)
            view_42 = None
            key_states_12 = transpose_31.contiguous()
            transpose_31 = None
            linear_38 = torch._C._nn.linear(
                hidden_states_43,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_43 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_43 = linear_38.view(2, -1, 16, 64)
            linear_38 = None
            transpose_32 = view_43.transpose(1, 2)
            view_43 = None
            value_states_12 = transpose_32.contiguous()
            transpose_32 = None
            view_44 = query_states_12.view(2, 257, 16, 64)
            query_states_12 = None
            transpose_33 = view_44.transpose(1, 2)
            view_44 = None
            contiguous_20 = transpose_33.contiguous()
            transpose_33 = None
            query_states_13 = contiguous_20.view(32, -1, 64)
            contiguous_20 = None
            key_states_13 = key_states_12.view(32, -1, 64)
            key_states_12 = None
            value_states_13 = value_states_12.view(32, -1, 64)
            value_states_12 = None
            transpose_34 = key_states_13.transpose(1, 2)
            key_states_13 = None
            attn_weights_12 = torch.bmm(query_states_13, transpose_34)
            query_states_13 = transpose_34 = None
            attn_weights_13 = torch.nn.functional.softmax(attn_weights_12, dim=-1)
            attn_weights_12 = None
            attn_probs_6 = torch.nn.functional.dropout(attn_weights_13, p=0.0, training=False)
            attn_weights_13 = None
            attn_output_30 = torch.bmm(attn_probs_6, value_states_13)
            attn_probs_6 = value_states_13 = None
            attn_output_31 = attn_output_30.view(2, 16, 257, 64)
            attn_output_30 = None
            attn_output_32 = attn_output_31.transpose(1, 2)
            attn_output_31 = None
            attn_output_33 = attn_output_32.reshape(2, 257, 1024)
            attn_output_32 = None
            attn_output_34 = torch._C._nn.linear(
                attn_output_33,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_33 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_44 = hidden_states_42 + attn_output_34
            hidden_states_42 = attn_output_34 = None
            hidden_states_45 = torch.nn.functional.layer_norm(
                hidden_states_44,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_46 = torch._C._nn.linear(
                hidden_states_45,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_45 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_19 = 1.702 * hidden_states_46
            sigmoid_6 = torch.sigmoid(mul_19)
            mul_19 = None
            hidden_states_47 = hidden_states_46 * sigmoid_6
            hidden_states_46 = sigmoid_6 = None
            hidden_states_48 = torch._C._nn.linear(
                hidden_states_47,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_47 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_6_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_49 = hidden_states_44 + hidden_states_48
            hidden_states_44 = hidden_states_48 = None
            hidden_states_50 = torch.nn.functional.layer_norm(
                hidden_states_49,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm1_parameters_bias_
            ) = None
            linear_42 = torch._C._nn.linear(
                hidden_states_50,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_14 = linear_42 * 0.125
            linear_42 = None
            linear_43 = torch._C._nn.linear(
                hidden_states_50,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_49 = linear_43.view(2, -1, 16, 64)
            linear_43 = None
            transpose_36 = view_49.transpose(1, 2)
            view_49 = None
            key_states_14 = transpose_36.contiguous()
            transpose_36 = None
            linear_44 = torch._C._nn.linear(
                hidden_states_50,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_50 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_50 = linear_44.view(2, -1, 16, 64)
            linear_44 = None
            transpose_37 = view_50.transpose(1, 2)
            view_50 = None
            value_states_14 = transpose_37.contiguous()
            transpose_37 = None
            view_51 = query_states_14.view(2, 257, 16, 64)
            query_states_14 = None
            transpose_38 = view_51.transpose(1, 2)
            view_51 = None
            contiguous_23 = transpose_38.contiguous()
            transpose_38 = None
            query_states_15 = contiguous_23.view(32, -1, 64)
            contiguous_23 = None
            key_states_15 = key_states_14.view(32, -1, 64)
            key_states_14 = None
            value_states_15 = value_states_14.view(32, -1, 64)
            value_states_14 = None
            transpose_39 = key_states_15.transpose(1, 2)
            key_states_15 = None
            attn_weights_14 = torch.bmm(query_states_15, transpose_39)
            query_states_15 = transpose_39 = None
            attn_weights_15 = torch.nn.functional.softmax(attn_weights_14, dim=-1)
            attn_weights_14 = None
            attn_probs_7 = torch.nn.functional.dropout(attn_weights_15, p=0.0, training=False)
            attn_weights_15 = None
            attn_output_35 = torch.bmm(attn_probs_7, value_states_15)
            attn_probs_7 = value_states_15 = None
            attn_output_36 = attn_output_35.view(2, 16, 257, 64)
            attn_output_35 = None
            attn_output_37 = attn_output_36.transpose(1, 2)
            attn_output_36 = None
            attn_output_38 = attn_output_37.reshape(2, 257, 1024)
            attn_output_37 = None
            attn_output_39 = torch._C._nn.linear(
                attn_output_38,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_38 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_51 = hidden_states_49 + attn_output_39
            hidden_states_49 = attn_output_39 = None
            hidden_states_52 = torch.nn.functional.layer_norm(
                hidden_states_51,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_53 = torch._C._nn.linear(
                hidden_states_52,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_52 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_22 = 1.702 * hidden_states_53
            sigmoid_7 = torch.sigmoid(mul_22)
            mul_22 = None
            hidden_states_54 = hidden_states_53 * sigmoid_7
            hidden_states_53 = sigmoid_7 = None
            hidden_states_55 = torch._C._nn.linear(
                hidden_states_54,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_54 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_7_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_56 = hidden_states_51 + hidden_states_55
            hidden_states_51 = hidden_states_55 = None
            hidden_states_57 = torch.nn.functional.layer_norm(
                hidden_states_56,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm1_parameters_bias_
            ) = None
            linear_48 = torch._C._nn.linear(
                hidden_states_57,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_16 = linear_48 * 0.125
            linear_48 = None
            linear_49 = torch._C._nn.linear(
                hidden_states_57,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_56 = linear_49.view(2, -1, 16, 64)
            linear_49 = None
            transpose_41 = view_56.transpose(1, 2)
            view_56 = None
            key_states_16 = transpose_41.contiguous()
            transpose_41 = None
            linear_50 = torch._C._nn.linear(
                hidden_states_57,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_57 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_57 = linear_50.view(2, -1, 16, 64)
            linear_50 = None
            transpose_42 = view_57.transpose(1, 2)
            view_57 = None
            value_states_16 = transpose_42.contiguous()
            transpose_42 = None
            view_58 = query_states_16.view(2, 257, 16, 64)
            query_states_16 = None
            transpose_43 = view_58.transpose(1, 2)
            view_58 = None
            contiguous_26 = transpose_43.contiguous()
            transpose_43 = None
            query_states_17 = contiguous_26.view(32, -1, 64)
            contiguous_26 = None
            key_states_17 = key_states_16.view(32, -1, 64)
            key_states_16 = None
            value_states_17 = value_states_16.view(32, -1, 64)
            value_states_16 = None
            transpose_44 = key_states_17.transpose(1, 2)
            key_states_17 = None
            attn_weights_16 = torch.bmm(query_states_17, transpose_44)
            query_states_17 = transpose_44 = None
            attn_weights_17 = torch.nn.functional.softmax(attn_weights_16, dim=-1)
            attn_weights_16 = None
            attn_probs_8 = torch.nn.functional.dropout(attn_weights_17, p=0.0, training=False)
            attn_weights_17 = None
            attn_output_40 = torch.bmm(attn_probs_8, value_states_17)
            attn_probs_8 = value_states_17 = None
            attn_output_41 = attn_output_40.view(2, 16, 257, 64)
            attn_output_40 = None
            attn_output_42 = attn_output_41.transpose(1, 2)
            attn_output_41 = None
            attn_output_43 = attn_output_42.reshape(2, 257, 1024)
            attn_output_42 = None
            attn_output_44 = torch._C._nn.linear(
                attn_output_43,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_43 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_58 = hidden_states_56 + attn_output_44
            hidden_states_56 = attn_output_44 = None
            hidden_states_59 = torch.nn.functional.layer_norm(
                hidden_states_58,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_60 = torch._C._nn.linear(
                hidden_states_59,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_59 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_25 = 1.702 * hidden_states_60
            sigmoid_8 = torch.sigmoid(mul_25)
            mul_25 = None
            hidden_states_61 = hidden_states_60 * sigmoid_8
            hidden_states_60 = sigmoid_8 = None
            hidden_states_62 = torch._C._nn.linear(
                hidden_states_61,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_61 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_8_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_63 = hidden_states_58 + hidden_states_62
            hidden_states_58 = hidden_states_62 = None
            hidden_states_64 = torch.nn.functional.layer_norm(
                hidden_states_63,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm1_parameters_bias_
            ) = None
            linear_54 = torch._C._nn.linear(
                hidden_states_64,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_18 = linear_54 * 0.125
            linear_54 = None
            linear_55 = torch._C._nn.linear(
                hidden_states_64,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_63 = linear_55.view(2, -1, 16, 64)
            linear_55 = None
            transpose_46 = view_63.transpose(1, 2)
            view_63 = None
            key_states_18 = transpose_46.contiguous()
            transpose_46 = None
            linear_56 = torch._C._nn.linear(
                hidden_states_64,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_64 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_64 = linear_56.view(2, -1, 16, 64)
            linear_56 = None
            transpose_47 = view_64.transpose(1, 2)
            view_64 = None
            value_states_18 = transpose_47.contiguous()
            transpose_47 = None
            view_65 = query_states_18.view(2, 257, 16, 64)
            query_states_18 = None
            transpose_48 = view_65.transpose(1, 2)
            view_65 = None
            contiguous_29 = transpose_48.contiguous()
            transpose_48 = None
            query_states_19 = contiguous_29.view(32, -1, 64)
            contiguous_29 = None
            key_states_19 = key_states_18.view(32, -1, 64)
            key_states_18 = None
            value_states_19 = value_states_18.view(32, -1, 64)
            value_states_18 = None
            transpose_49 = key_states_19.transpose(1, 2)
            key_states_19 = None
            attn_weights_18 = torch.bmm(query_states_19, transpose_49)
            query_states_19 = transpose_49 = None
            attn_weights_19 = torch.nn.functional.softmax(attn_weights_18, dim=-1)
            attn_weights_18 = None
            attn_probs_9 = torch.nn.functional.dropout(attn_weights_19, p=0.0, training=False)
            attn_weights_19 = None
            attn_output_45 = torch.bmm(attn_probs_9, value_states_19)
            attn_probs_9 = value_states_19 = None
            attn_output_46 = attn_output_45.view(2, 16, 257, 64)
            attn_output_45 = None
            attn_output_47 = attn_output_46.transpose(1, 2)
            attn_output_46 = None
            attn_output_48 = attn_output_47.reshape(2, 257, 1024)
            attn_output_47 = None
            attn_output_49 = torch._C._nn.linear(
                attn_output_48,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_48 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_65 = hidden_states_63 + attn_output_49
            hidden_states_63 = attn_output_49 = None
            hidden_states_66 = torch.nn.functional.layer_norm(
                hidden_states_65,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_67 = torch._C._nn.linear(
                hidden_states_66,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_66 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_28 = 1.702 * hidden_states_67
            sigmoid_9 = torch.sigmoid(mul_28)
            mul_28 = None
            hidden_states_68 = hidden_states_67 * sigmoid_9
            hidden_states_67 = sigmoid_9 = None
            hidden_states_69 = torch._C._nn.linear(
                hidden_states_68,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_68 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_9_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_70 = hidden_states_65 + hidden_states_69
            hidden_states_65 = hidden_states_69 = None
            hidden_states_71 = torch.nn.functional.layer_norm(
                hidden_states_70,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm1_parameters_bias_
            ) = None
            linear_60 = torch._C._nn.linear(
                hidden_states_71,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_20 = linear_60 * 0.125
            linear_60 = None
            linear_61 = torch._C._nn.linear(
                hidden_states_71,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_70 = linear_61.view(2, -1, 16, 64)
            linear_61 = None
            transpose_51 = view_70.transpose(1, 2)
            view_70 = None
            key_states_20 = transpose_51.contiguous()
            transpose_51 = None
            linear_62 = torch._C._nn.linear(
                hidden_states_71,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_71 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_71 = linear_62.view(2, -1, 16, 64)
            linear_62 = None
            transpose_52 = view_71.transpose(1, 2)
            view_71 = None
            value_states_20 = transpose_52.contiguous()
            transpose_52 = None
            view_72 = query_states_20.view(2, 257, 16, 64)
            query_states_20 = None
            transpose_53 = view_72.transpose(1, 2)
            view_72 = None
            contiguous_32 = transpose_53.contiguous()
            transpose_53 = None
            query_states_21 = contiguous_32.view(32, -1, 64)
            contiguous_32 = None
            key_states_21 = key_states_20.view(32, -1, 64)
            key_states_20 = None
            value_states_21 = value_states_20.view(32, -1, 64)
            value_states_20 = None
            transpose_54 = key_states_21.transpose(1, 2)
            key_states_21 = None
            attn_weights_20 = torch.bmm(query_states_21, transpose_54)
            query_states_21 = transpose_54 = None
            attn_weights_21 = torch.nn.functional.softmax(attn_weights_20, dim=-1)
            attn_weights_20 = None
            attn_probs_10 = torch.nn.functional.dropout(attn_weights_21, p=0.0, training=False)
            attn_weights_21 = None
            attn_output_50 = torch.bmm(attn_probs_10, value_states_21)
            attn_probs_10 = value_states_21 = None
            attn_output_51 = attn_output_50.view(2, 16, 257, 64)
            attn_output_50 = None
            attn_output_52 = attn_output_51.transpose(1, 2)
            attn_output_51 = None
            attn_output_53 = attn_output_52.reshape(2, 257, 1024)
            attn_output_52 = None
            attn_output_54 = torch._C._nn.linear(
                attn_output_53,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_53 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_72 = hidden_states_70 + attn_output_54
            hidden_states_70 = attn_output_54 = None
            hidden_states_73 = torch.nn.functional.layer_norm(
                hidden_states_72,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_74 = torch._C._nn.linear(
                hidden_states_73,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_73 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_31 = 1.702 * hidden_states_74
            sigmoid_10 = torch.sigmoid(mul_31)
            mul_31 = None
            hidden_states_75 = hidden_states_74 * sigmoid_10
            hidden_states_74 = sigmoid_10 = None
            hidden_states_76 = torch._C._nn.linear(
                hidden_states_75,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_75 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_10_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_77 = hidden_states_72 + hidden_states_76
            hidden_states_72 = hidden_states_76 = None
            hidden_states_78 = torch.nn.functional.layer_norm(
                hidden_states_77,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm1_parameters_bias_
            ) = None
            linear_66 = torch._C._nn.linear(
                hidden_states_78,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_22 = linear_66 * 0.125
            linear_66 = None
            linear_67 = torch._C._nn.linear(
                hidden_states_78,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_77 = linear_67.view(2, -1, 16, 64)
            linear_67 = None
            transpose_56 = view_77.transpose(1, 2)
            view_77 = None
            key_states_22 = transpose_56.contiguous()
            transpose_56 = None
            linear_68 = torch._C._nn.linear(
                hidden_states_78,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_78 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_78 = linear_68.view(2, -1, 16, 64)
            linear_68 = None
            transpose_57 = view_78.transpose(1, 2)
            view_78 = None
            value_states_22 = transpose_57.contiguous()
            transpose_57 = None
            view_79 = query_states_22.view(2, 257, 16, 64)
            query_states_22 = None
            transpose_58 = view_79.transpose(1, 2)
            view_79 = None
            contiguous_35 = transpose_58.contiguous()
            transpose_58 = None
            query_states_23 = contiguous_35.view(32, -1, 64)
            contiguous_35 = None
            key_states_23 = key_states_22.view(32, -1, 64)
            key_states_22 = None
            value_states_23 = value_states_22.view(32, -1, 64)
            value_states_22 = None
            transpose_59 = key_states_23.transpose(1, 2)
            key_states_23 = None
            attn_weights_22 = torch.bmm(query_states_23, transpose_59)
            query_states_23 = transpose_59 = None
            attn_weights_23 = torch.nn.functional.softmax(attn_weights_22, dim=-1)
            attn_weights_22 = None
            attn_probs_11 = torch.nn.functional.dropout(attn_weights_23, p=0.0, training=False)
            attn_weights_23 = None
            attn_output_55 = torch.bmm(attn_probs_11, value_states_23)
            attn_probs_11 = value_states_23 = None
            attn_output_56 = attn_output_55.view(2, 16, 257, 64)
            attn_output_55 = None
            attn_output_57 = attn_output_56.transpose(1, 2)
            attn_output_56 = None
            attn_output_58 = attn_output_57.reshape(2, 257, 1024)
            attn_output_57 = None
            attn_output_59 = torch._C._nn.linear(
                attn_output_58,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_58 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_79 = hidden_states_77 + attn_output_59
            hidden_states_77 = attn_output_59 = None
            hidden_states_80 = torch.nn.functional.layer_norm(
                hidden_states_79,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_81 = torch._C._nn.linear(
                hidden_states_80,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_80 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_34 = 1.702 * hidden_states_81
            sigmoid_11 = torch.sigmoid(mul_34)
            mul_34 = None
            hidden_states_82 = hidden_states_81 * sigmoid_11
            hidden_states_81 = sigmoid_11 = None
            hidden_states_83 = torch._C._nn.linear(
                hidden_states_82,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_82 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_11_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_84 = hidden_states_79 + hidden_states_83
            hidden_states_79 = hidden_states_83 = None
            hidden_states_85 = torch.nn.functional.layer_norm(
                hidden_states_84,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm1_parameters_bias_
            ) = None
            linear_72 = torch._C._nn.linear(
                hidden_states_85,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_24 = linear_72 * 0.125
            linear_72 = None
            linear_73 = torch._C._nn.linear(
                hidden_states_85,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_84 = linear_73.view(2, -1, 16, 64)
            linear_73 = None
            transpose_61 = view_84.transpose(1, 2)
            view_84 = None
            key_states_24 = transpose_61.contiguous()
            transpose_61 = None
            linear_74 = torch._C._nn.linear(
                hidden_states_85,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_85 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_85 = linear_74.view(2, -1, 16, 64)
            linear_74 = None
            transpose_62 = view_85.transpose(1, 2)
            view_85 = None
            value_states_24 = transpose_62.contiguous()
            transpose_62 = None
            view_86 = query_states_24.view(2, 257, 16, 64)
            query_states_24 = None
            transpose_63 = view_86.transpose(1, 2)
            view_86 = None
            contiguous_38 = transpose_63.contiguous()
            transpose_63 = None
            query_states_25 = contiguous_38.view(32, -1, 64)
            contiguous_38 = None
            key_states_25 = key_states_24.view(32, -1, 64)
            key_states_24 = None
            value_states_25 = value_states_24.view(32, -1, 64)
            value_states_24 = None
            transpose_64 = key_states_25.transpose(1, 2)
            key_states_25 = None
            attn_weights_24 = torch.bmm(query_states_25, transpose_64)
            query_states_25 = transpose_64 = None
            attn_weights_25 = torch.nn.functional.softmax(attn_weights_24, dim=-1)
            attn_weights_24 = None
            attn_probs_12 = torch.nn.functional.dropout(attn_weights_25, p=0.0, training=False)
            attn_weights_25 = None
            attn_output_60 = torch.bmm(attn_probs_12, value_states_25)
            attn_probs_12 = value_states_25 = None
            attn_output_61 = attn_output_60.view(2, 16, 257, 64)
            attn_output_60 = None
            attn_output_62 = attn_output_61.transpose(1, 2)
            attn_output_61 = None
            attn_output_63 = attn_output_62.reshape(2, 257, 1024)
            attn_output_62 = None
            attn_output_64 = torch._C._nn.linear(
                attn_output_63,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_63 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_86 = hidden_states_84 + attn_output_64
            hidden_states_84 = attn_output_64 = None
            hidden_states_87 = torch.nn.functional.layer_norm(
                hidden_states_86,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_88 = torch._C._nn.linear(
                hidden_states_87,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_87 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_37 = 1.702 * hidden_states_88
            sigmoid_12 = torch.sigmoid(mul_37)
            mul_37 = None
            hidden_states_89 = hidden_states_88 * sigmoid_12
            hidden_states_88 = sigmoid_12 = None
            hidden_states_90 = torch._C._nn.linear(
                hidden_states_89,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_89 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_12_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_91 = hidden_states_86 + hidden_states_90
            hidden_states_86 = hidden_states_90 = None
            hidden_states_92 = torch.nn.functional.layer_norm(
                hidden_states_91,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm1_parameters_bias_
            ) = None
            linear_78 = torch._C._nn.linear(
                hidden_states_92,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_26 = linear_78 * 0.125
            linear_78 = None
            linear_79 = torch._C._nn.linear(
                hidden_states_92,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_91 = linear_79.view(2, -1, 16, 64)
            linear_79 = None
            transpose_66 = view_91.transpose(1, 2)
            view_91 = None
            key_states_26 = transpose_66.contiguous()
            transpose_66 = None
            linear_80 = torch._C._nn.linear(
                hidden_states_92,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_92 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_92 = linear_80.view(2, -1, 16, 64)
            linear_80 = None
            transpose_67 = view_92.transpose(1, 2)
            view_92 = None
            value_states_26 = transpose_67.contiguous()
            transpose_67 = None
            view_93 = query_states_26.view(2, 257, 16, 64)
            query_states_26 = None
            transpose_68 = view_93.transpose(1, 2)
            view_93 = None
            contiguous_41 = transpose_68.contiguous()
            transpose_68 = None
            query_states_27 = contiguous_41.view(32, -1, 64)
            contiguous_41 = None
            key_states_27 = key_states_26.view(32, -1, 64)
            key_states_26 = None
            value_states_27 = value_states_26.view(32, -1, 64)
            value_states_26 = None
            transpose_69 = key_states_27.transpose(1, 2)
            key_states_27 = None
            attn_weights_26 = torch.bmm(query_states_27, transpose_69)
            query_states_27 = transpose_69 = None
            attn_weights_27 = torch.nn.functional.softmax(attn_weights_26, dim=-1)
            attn_weights_26 = None
            attn_probs_13 = torch.nn.functional.dropout(attn_weights_27, p=0.0, training=False)
            attn_weights_27 = None
            attn_output_65 = torch.bmm(attn_probs_13, value_states_27)
            attn_probs_13 = value_states_27 = None
            attn_output_66 = attn_output_65.view(2, 16, 257, 64)
            attn_output_65 = None
            attn_output_67 = attn_output_66.transpose(1, 2)
            attn_output_66 = None
            attn_output_68 = attn_output_67.reshape(2, 257, 1024)
            attn_output_67 = None
            attn_output_69 = torch._C._nn.linear(
                attn_output_68,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_68 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_93 = hidden_states_91 + attn_output_69
            hidden_states_91 = attn_output_69 = None
            hidden_states_94 = torch.nn.functional.layer_norm(
                hidden_states_93,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_95 = torch._C._nn.linear(
                hidden_states_94,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_94 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_40 = 1.702 * hidden_states_95
            sigmoid_13 = torch.sigmoid(mul_40)
            mul_40 = None
            hidden_states_96 = hidden_states_95 * sigmoid_13
            hidden_states_95 = sigmoid_13 = None
            hidden_states_97 = torch._C._nn.linear(
                hidden_states_96,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_96 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_13_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_98 = hidden_states_93 + hidden_states_97
            hidden_states_93 = hidden_states_97 = None
            hidden_states_99 = torch.nn.functional.layer_norm(
                hidden_states_98,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm1_parameters_bias_
            ) = None
            linear_84 = torch._C._nn.linear(
                hidden_states_99,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_28 = linear_84 * 0.125
            linear_84 = None
            linear_85 = torch._C._nn.linear(
                hidden_states_99,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_98 = linear_85.view(2, -1, 16, 64)
            linear_85 = None
            transpose_71 = view_98.transpose(1, 2)
            view_98 = None
            key_states_28 = transpose_71.contiguous()
            transpose_71 = None
            linear_86 = torch._C._nn.linear(
                hidden_states_99,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_99 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_99 = linear_86.view(2, -1, 16, 64)
            linear_86 = None
            transpose_72 = view_99.transpose(1, 2)
            view_99 = None
            value_states_28 = transpose_72.contiguous()
            transpose_72 = None
            view_100 = query_states_28.view(2, 257, 16, 64)
            query_states_28 = None
            transpose_73 = view_100.transpose(1, 2)
            view_100 = None
            contiguous_44 = transpose_73.contiguous()
            transpose_73 = None
            query_states_29 = contiguous_44.view(32, -1, 64)
            contiguous_44 = None
            key_states_29 = key_states_28.view(32, -1, 64)
            key_states_28 = None
            value_states_29 = value_states_28.view(32, -1, 64)
            value_states_28 = None
            transpose_74 = key_states_29.transpose(1, 2)
            key_states_29 = None
            attn_weights_28 = torch.bmm(query_states_29, transpose_74)
            query_states_29 = transpose_74 = None
            attn_weights_29 = torch.nn.functional.softmax(attn_weights_28, dim=-1)
            attn_weights_28 = None
            attn_probs_14 = torch.nn.functional.dropout(attn_weights_29, p=0.0, training=False)
            attn_weights_29 = None
            attn_output_70 = torch.bmm(attn_probs_14, value_states_29)
            attn_probs_14 = value_states_29 = None
            attn_output_71 = attn_output_70.view(2, 16, 257, 64)
            attn_output_70 = None
            attn_output_72 = attn_output_71.transpose(1, 2)
            attn_output_71 = None
            attn_output_73 = attn_output_72.reshape(2, 257, 1024)
            attn_output_72 = None
            attn_output_74 = torch._C._nn.linear(
                attn_output_73,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_73 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_100 = hidden_states_98 + attn_output_74
            hidden_states_98 = attn_output_74 = None
            hidden_states_101 = torch.nn.functional.layer_norm(
                hidden_states_100,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_102 = torch._C._nn.linear(
                hidden_states_101,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_101 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_43 = 1.702 * hidden_states_102
            sigmoid_14 = torch.sigmoid(mul_43)
            mul_43 = None
            hidden_states_103 = hidden_states_102 * sigmoid_14
            hidden_states_102 = sigmoid_14 = None
            hidden_states_104 = torch._C._nn.linear(
                hidden_states_103,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_103 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_14_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_105 = hidden_states_100 + hidden_states_104
            hidden_states_100 = hidden_states_104 = None
            hidden_states_106 = torch.nn.functional.layer_norm(
                hidden_states_105,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm1_parameters_bias_
            ) = None
            linear_90 = torch._C._nn.linear(
                hidden_states_106,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_30 = linear_90 * 0.125
            linear_90 = None
            linear_91 = torch._C._nn.linear(
                hidden_states_106,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_105 = linear_91.view(2, -1, 16, 64)
            linear_91 = None
            transpose_76 = view_105.transpose(1, 2)
            view_105 = None
            key_states_30 = transpose_76.contiguous()
            transpose_76 = None
            linear_92 = torch._C._nn.linear(
                hidden_states_106,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_106 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_106 = linear_92.view(2, -1, 16, 64)
            linear_92 = None
            transpose_77 = view_106.transpose(1, 2)
            view_106 = None
            value_states_30 = transpose_77.contiguous()
            transpose_77 = None
            view_107 = query_states_30.view(2, 257, 16, 64)
            query_states_30 = None
            transpose_78 = view_107.transpose(1, 2)
            view_107 = None
            contiguous_47 = transpose_78.contiguous()
            transpose_78 = None
            query_states_31 = contiguous_47.view(32, -1, 64)
            contiguous_47 = None
            key_states_31 = key_states_30.view(32, -1, 64)
            key_states_30 = None
            value_states_31 = value_states_30.view(32, -1, 64)
            value_states_30 = None
            transpose_79 = key_states_31.transpose(1, 2)
            key_states_31 = None
            attn_weights_30 = torch.bmm(query_states_31, transpose_79)
            query_states_31 = transpose_79 = None
            attn_weights_31 = torch.nn.functional.softmax(attn_weights_30, dim=-1)
            attn_weights_30 = None
            attn_probs_15 = torch.nn.functional.dropout(attn_weights_31, p=0.0, training=False)
            attn_weights_31 = None
            attn_output_75 = torch.bmm(attn_probs_15, value_states_31)
            attn_probs_15 = value_states_31 = None
            attn_output_76 = attn_output_75.view(2, 16, 257, 64)
            attn_output_75 = None
            attn_output_77 = attn_output_76.transpose(1, 2)
            attn_output_76 = None
            attn_output_78 = attn_output_77.reshape(2, 257, 1024)
            attn_output_77 = None
            attn_output_79 = torch._C._nn.linear(
                attn_output_78,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_78 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_107 = hidden_states_105 + attn_output_79
            hidden_states_105 = attn_output_79 = None
            hidden_states_108 = torch.nn.functional.layer_norm(
                hidden_states_107,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_109 = torch._C._nn.linear(
                hidden_states_108,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_108 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_46 = 1.702 * hidden_states_109
            sigmoid_15 = torch.sigmoid(mul_46)
            mul_46 = None
            hidden_states_110 = hidden_states_109 * sigmoid_15
            hidden_states_109 = sigmoid_15 = None
            hidden_states_111 = torch._C._nn.linear(
                hidden_states_110,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_110 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_15_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_112 = hidden_states_107 + hidden_states_111
            hidden_states_107 = hidden_states_111 = None
            hidden_states_113 = torch.nn.functional.layer_norm(
                hidden_states_112,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm1_parameters_bias_
            ) = None
            linear_96 = torch._C._nn.linear(
                hidden_states_113,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_32 = linear_96 * 0.125
            linear_96 = None
            linear_97 = torch._C._nn.linear(
                hidden_states_113,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_112 = linear_97.view(2, -1, 16, 64)
            linear_97 = None
            transpose_81 = view_112.transpose(1, 2)
            view_112 = None
            key_states_32 = transpose_81.contiguous()
            transpose_81 = None
            linear_98 = torch._C._nn.linear(
                hidden_states_113,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_113 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_113 = linear_98.view(2, -1, 16, 64)
            linear_98 = None
            transpose_82 = view_113.transpose(1, 2)
            view_113 = None
            value_states_32 = transpose_82.contiguous()
            transpose_82 = None
            view_114 = query_states_32.view(2, 257, 16, 64)
            query_states_32 = None
            transpose_83 = view_114.transpose(1, 2)
            view_114 = None
            contiguous_50 = transpose_83.contiguous()
            transpose_83 = None
            query_states_33 = contiguous_50.view(32, -1, 64)
            contiguous_50 = None
            key_states_33 = key_states_32.view(32, -1, 64)
            key_states_32 = None
            value_states_33 = value_states_32.view(32, -1, 64)
            value_states_32 = None
            transpose_84 = key_states_33.transpose(1, 2)
            key_states_33 = None
            attn_weights_32 = torch.bmm(query_states_33, transpose_84)
            query_states_33 = transpose_84 = None
            attn_weights_33 = torch.nn.functional.softmax(attn_weights_32, dim=-1)
            attn_weights_32 = None
            attn_probs_16 = torch.nn.functional.dropout(attn_weights_33, p=0.0, training=False)
            attn_weights_33 = None
            attn_output_80 = torch.bmm(attn_probs_16, value_states_33)
            attn_probs_16 = value_states_33 = None
            attn_output_81 = attn_output_80.view(2, 16, 257, 64)
            attn_output_80 = None
            attn_output_82 = attn_output_81.transpose(1, 2)
            attn_output_81 = None
            attn_output_83 = attn_output_82.reshape(2, 257, 1024)
            attn_output_82 = None
            attn_output_84 = torch._C._nn.linear(
                attn_output_83,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_83 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_114 = hidden_states_112 + attn_output_84
            hidden_states_112 = attn_output_84 = None
            hidden_states_115 = torch.nn.functional.layer_norm(
                hidden_states_114,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_116 = torch._C._nn.linear(
                hidden_states_115,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_115 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_49 = 1.702 * hidden_states_116
            sigmoid_16 = torch.sigmoid(mul_49)
            mul_49 = None
            hidden_states_117 = hidden_states_116 * sigmoid_16
            hidden_states_116 = sigmoid_16 = None
            hidden_states_118 = torch._C._nn.linear(
                hidden_states_117,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_117 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_16_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_119 = hidden_states_114 + hidden_states_118
            hidden_states_114 = hidden_states_118 = None
            hidden_states_120 = torch.nn.functional.layer_norm(
                hidden_states_119,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm1_parameters_bias_
            ) = None
            linear_102 = torch._C._nn.linear(
                hidden_states_120,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_34 = linear_102 * 0.125
            linear_102 = None
            linear_103 = torch._C._nn.linear(
                hidden_states_120,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_119 = linear_103.view(2, -1, 16, 64)
            linear_103 = None
            transpose_86 = view_119.transpose(1, 2)
            view_119 = None
            key_states_34 = transpose_86.contiguous()
            transpose_86 = None
            linear_104 = torch._C._nn.linear(
                hidden_states_120,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_120 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_120 = linear_104.view(2, -1, 16, 64)
            linear_104 = None
            transpose_87 = view_120.transpose(1, 2)
            view_120 = None
            value_states_34 = transpose_87.contiguous()
            transpose_87 = None
            view_121 = query_states_34.view(2, 257, 16, 64)
            query_states_34 = None
            transpose_88 = view_121.transpose(1, 2)
            view_121 = None
            contiguous_53 = transpose_88.contiguous()
            transpose_88 = None
            query_states_35 = contiguous_53.view(32, -1, 64)
            contiguous_53 = None
            key_states_35 = key_states_34.view(32, -1, 64)
            key_states_34 = None
            value_states_35 = value_states_34.view(32, -1, 64)
            value_states_34 = None
            transpose_89 = key_states_35.transpose(1, 2)
            key_states_35 = None
            attn_weights_34 = torch.bmm(query_states_35, transpose_89)
            query_states_35 = transpose_89 = None
            attn_weights_35 = torch.nn.functional.softmax(attn_weights_34, dim=-1)
            attn_weights_34 = None
            attn_probs_17 = torch.nn.functional.dropout(attn_weights_35, p=0.0, training=False)
            attn_weights_35 = None
            attn_output_85 = torch.bmm(attn_probs_17, value_states_35)
            attn_probs_17 = value_states_35 = None
            attn_output_86 = attn_output_85.view(2, 16, 257, 64)
            attn_output_85 = None
            attn_output_87 = attn_output_86.transpose(1, 2)
            attn_output_86 = None
            attn_output_88 = attn_output_87.reshape(2, 257, 1024)
            attn_output_87 = None
            attn_output_89 = torch._C._nn.linear(
                attn_output_88,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_88 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_121 = hidden_states_119 + attn_output_89
            hidden_states_119 = attn_output_89 = None
            hidden_states_122 = torch.nn.functional.layer_norm(
                hidden_states_121,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_123 = torch._C._nn.linear(
                hidden_states_122,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_122 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_52 = 1.702 * hidden_states_123
            sigmoid_17 = torch.sigmoid(mul_52)
            mul_52 = None
            hidden_states_124 = hidden_states_123 * sigmoid_17
            hidden_states_123 = sigmoid_17 = None
            hidden_states_125 = torch._C._nn.linear(
                hidden_states_124,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_124 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_17_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_126 = hidden_states_121 + hidden_states_125
            hidden_states_121 = hidden_states_125 = None
            hidden_states_127 = torch.nn.functional.layer_norm(
                hidden_states_126,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm1_parameters_bias_
            ) = None
            linear_108 = torch._C._nn.linear(
                hidden_states_127,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_36 = linear_108 * 0.125
            linear_108 = None
            linear_109 = torch._C._nn.linear(
                hidden_states_127,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_126 = linear_109.view(2, -1, 16, 64)
            linear_109 = None
            transpose_91 = view_126.transpose(1, 2)
            view_126 = None
            key_states_36 = transpose_91.contiguous()
            transpose_91 = None
            linear_110 = torch._C._nn.linear(
                hidden_states_127,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_127 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_127 = linear_110.view(2, -1, 16, 64)
            linear_110 = None
            transpose_92 = view_127.transpose(1, 2)
            view_127 = None
            value_states_36 = transpose_92.contiguous()
            transpose_92 = None
            view_128 = query_states_36.view(2, 257, 16, 64)
            query_states_36 = None
            transpose_93 = view_128.transpose(1, 2)
            view_128 = None
            contiguous_56 = transpose_93.contiguous()
            transpose_93 = None
            query_states_37 = contiguous_56.view(32, -1, 64)
            contiguous_56 = None
            key_states_37 = key_states_36.view(32, -1, 64)
            key_states_36 = None
            value_states_37 = value_states_36.view(32, -1, 64)
            value_states_36 = None
            transpose_94 = key_states_37.transpose(1, 2)
            key_states_37 = None
            attn_weights_36 = torch.bmm(query_states_37, transpose_94)
            query_states_37 = transpose_94 = None
            attn_weights_37 = torch.nn.functional.softmax(attn_weights_36, dim=-1)
            attn_weights_36 = None
            attn_probs_18 = torch.nn.functional.dropout(attn_weights_37, p=0.0, training=False)
            attn_weights_37 = None
            attn_output_90 = torch.bmm(attn_probs_18, value_states_37)
            attn_probs_18 = value_states_37 = None
            attn_output_91 = attn_output_90.view(2, 16, 257, 64)
            attn_output_90 = None
            attn_output_92 = attn_output_91.transpose(1, 2)
            attn_output_91 = None
            attn_output_93 = attn_output_92.reshape(2, 257, 1024)
            attn_output_92 = None
            attn_output_94 = torch._C._nn.linear(
                attn_output_93,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_93 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_128 = hidden_states_126 + attn_output_94
            hidden_states_126 = attn_output_94 = None
            hidden_states_129 = torch.nn.functional.layer_norm(
                hidden_states_128,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_130 = torch._C._nn.linear(
                hidden_states_129,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_129 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_55 = 1.702 * hidden_states_130
            sigmoid_18 = torch.sigmoid(mul_55)
            mul_55 = None
            hidden_states_131 = hidden_states_130 * sigmoid_18
            hidden_states_130 = sigmoid_18 = None
            hidden_states_132 = torch._C._nn.linear(
                hidden_states_131,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_131 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_18_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_133 = hidden_states_128 + hidden_states_132
            hidden_states_128 = hidden_states_132 = None
            hidden_states_134 = torch.nn.functional.layer_norm(
                hidden_states_133,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm1_parameters_bias_
            ) = None
            linear_114 = torch._C._nn.linear(
                hidden_states_134,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_38 = linear_114 * 0.125
            linear_114 = None
            linear_115 = torch._C._nn.linear(
                hidden_states_134,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_133 = linear_115.view(2, -1, 16, 64)
            linear_115 = None
            transpose_96 = view_133.transpose(1, 2)
            view_133 = None
            key_states_38 = transpose_96.contiguous()
            transpose_96 = None
            linear_116 = torch._C._nn.linear(
                hidden_states_134,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_134 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_134 = linear_116.view(2, -1, 16, 64)
            linear_116 = None
            transpose_97 = view_134.transpose(1, 2)
            view_134 = None
            value_states_38 = transpose_97.contiguous()
            transpose_97 = None
            view_135 = query_states_38.view(2, 257, 16, 64)
            query_states_38 = None
            transpose_98 = view_135.transpose(1, 2)
            view_135 = None
            contiguous_59 = transpose_98.contiguous()
            transpose_98 = None
            query_states_39 = contiguous_59.view(32, -1, 64)
            contiguous_59 = None
            key_states_39 = key_states_38.view(32, -1, 64)
            key_states_38 = None
            value_states_39 = value_states_38.view(32, -1, 64)
            value_states_38 = None
            transpose_99 = key_states_39.transpose(1, 2)
            key_states_39 = None
            attn_weights_38 = torch.bmm(query_states_39, transpose_99)
            query_states_39 = transpose_99 = None
            attn_weights_39 = torch.nn.functional.softmax(attn_weights_38, dim=-1)
            attn_weights_38 = None
            attn_probs_19 = torch.nn.functional.dropout(attn_weights_39, p=0.0, training=False)
            attn_weights_39 = None
            attn_output_95 = torch.bmm(attn_probs_19, value_states_39)
            attn_probs_19 = value_states_39 = None
            attn_output_96 = attn_output_95.view(2, 16, 257, 64)
            attn_output_95 = None
            attn_output_97 = attn_output_96.transpose(1, 2)
            attn_output_96 = None
            attn_output_98 = attn_output_97.reshape(2, 257, 1024)
            attn_output_97 = None
            attn_output_99 = torch._C._nn.linear(
                attn_output_98,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_98 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_135 = hidden_states_133 + attn_output_99
            hidden_states_133 = attn_output_99 = None
            hidden_states_136 = torch.nn.functional.layer_norm(
                hidden_states_135,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_137 = torch._C._nn.linear(
                hidden_states_136,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_136 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_58 = 1.702 * hidden_states_137
            sigmoid_19 = torch.sigmoid(mul_58)
            mul_58 = None
            hidden_states_138 = hidden_states_137 * sigmoid_19
            hidden_states_137 = sigmoid_19 = None
            hidden_states_139 = torch._C._nn.linear(
                hidden_states_138,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_138 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_19_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_140 = hidden_states_135 + hidden_states_139
            hidden_states_135 = hidden_states_139 = None
            hidden_states_141 = torch.nn.functional.layer_norm(
                hidden_states_140,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm1_parameters_bias_
            ) = None
            linear_120 = torch._C._nn.linear(
                hidden_states_141,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_40 = linear_120 * 0.125
            linear_120 = None
            linear_121 = torch._C._nn.linear(
                hidden_states_141,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_140 = linear_121.view(2, -1, 16, 64)
            linear_121 = None
            transpose_101 = view_140.transpose(1, 2)
            view_140 = None
            key_states_40 = transpose_101.contiguous()
            transpose_101 = None
            linear_122 = torch._C._nn.linear(
                hidden_states_141,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_141 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_141 = linear_122.view(2, -1, 16, 64)
            linear_122 = None
            transpose_102 = view_141.transpose(1, 2)
            view_141 = None
            value_states_40 = transpose_102.contiguous()
            transpose_102 = None
            view_142 = query_states_40.view(2, 257, 16, 64)
            query_states_40 = None
            transpose_103 = view_142.transpose(1, 2)
            view_142 = None
            contiguous_62 = transpose_103.contiguous()
            transpose_103 = None
            query_states_41 = contiguous_62.view(32, -1, 64)
            contiguous_62 = None
            key_states_41 = key_states_40.view(32, -1, 64)
            key_states_40 = None
            value_states_41 = value_states_40.view(32, -1, 64)
            value_states_40 = None
            transpose_104 = key_states_41.transpose(1, 2)
            key_states_41 = None
            attn_weights_40 = torch.bmm(query_states_41, transpose_104)
            query_states_41 = transpose_104 = None
            attn_weights_41 = torch.nn.functional.softmax(attn_weights_40, dim=-1)
            attn_weights_40 = None
            attn_probs_20 = torch.nn.functional.dropout(attn_weights_41, p=0.0, training=False)
            attn_weights_41 = None
            attn_output_100 = torch.bmm(attn_probs_20, value_states_41)
            attn_probs_20 = value_states_41 = None
            attn_output_101 = attn_output_100.view(2, 16, 257, 64)
            attn_output_100 = None
            attn_output_102 = attn_output_101.transpose(1, 2)
            attn_output_101 = None
            attn_output_103 = attn_output_102.reshape(2, 257, 1024)
            attn_output_102 = None
            attn_output_104 = torch._C._nn.linear(
                attn_output_103,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_103 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_142 = hidden_states_140 + attn_output_104
            hidden_states_140 = attn_output_104 = None
            hidden_states_143 = torch.nn.functional.layer_norm(
                hidden_states_142,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_144 = torch._C._nn.linear(
                hidden_states_143,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_143 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_61 = 1.702 * hidden_states_144
            sigmoid_20 = torch.sigmoid(mul_61)
            mul_61 = None
            hidden_states_145 = hidden_states_144 * sigmoid_20
            hidden_states_144 = sigmoid_20 = None
            hidden_states_146 = torch._C._nn.linear(
                hidden_states_145,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_145 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_20_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_147 = hidden_states_142 + hidden_states_146
            hidden_states_142 = hidden_states_146 = None
            hidden_states_148 = torch.nn.functional.layer_norm(
                hidden_states_147,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm1_parameters_bias_
            ) = None
            linear_126 = torch._C._nn.linear(
                hidden_states_148,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_42 = linear_126 * 0.125
            linear_126 = None
            linear_127 = torch._C._nn.linear(
                hidden_states_148,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_147 = linear_127.view(2, -1, 16, 64)
            linear_127 = None
            transpose_106 = view_147.transpose(1, 2)
            view_147 = None
            key_states_42 = transpose_106.contiguous()
            transpose_106 = None
            linear_128 = torch._C._nn.linear(
                hidden_states_148,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_148 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_148 = linear_128.view(2, -1, 16, 64)
            linear_128 = None
            transpose_107 = view_148.transpose(1, 2)
            view_148 = None
            value_states_42 = transpose_107.contiguous()
            transpose_107 = None
            view_149 = query_states_42.view(2, 257, 16, 64)
            query_states_42 = None
            transpose_108 = view_149.transpose(1, 2)
            view_149 = None
            contiguous_65 = transpose_108.contiguous()
            transpose_108 = None
            query_states_43 = contiguous_65.view(32, -1, 64)
            contiguous_65 = None
            key_states_43 = key_states_42.view(32, -1, 64)
            key_states_42 = None
            value_states_43 = value_states_42.view(32, -1, 64)
            value_states_42 = None
            transpose_109 = key_states_43.transpose(1, 2)
            key_states_43 = None
            attn_weights_42 = torch.bmm(query_states_43, transpose_109)
            query_states_43 = transpose_109 = None
            attn_weights_43 = torch.nn.functional.softmax(attn_weights_42, dim=-1)
            attn_weights_42 = None
            attn_probs_21 = torch.nn.functional.dropout(attn_weights_43, p=0.0, training=False)
            attn_weights_43 = None
            attn_output_105 = torch.bmm(attn_probs_21, value_states_43)
            attn_probs_21 = value_states_43 = None
            attn_output_106 = attn_output_105.view(2, 16, 257, 64)
            attn_output_105 = None
            attn_output_107 = attn_output_106.transpose(1, 2)
            attn_output_106 = None
            attn_output_108 = attn_output_107.reshape(2, 257, 1024)
            attn_output_107 = None
            attn_output_109 = torch._C._nn.linear(
                attn_output_108,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_108 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_149 = hidden_states_147 + attn_output_109
            hidden_states_147 = attn_output_109 = None
            hidden_states_150 = torch.nn.functional.layer_norm(
                hidden_states_149,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_151 = torch._C._nn.linear(
                hidden_states_150,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_150 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_64 = 1.702 * hidden_states_151
            sigmoid_21 = torch.sigmoid(mul_64)
            mul_64 = None
            hidden_states_152 = hidden_states_151 * sigmoid_21
            hidden_states_151 = sigmoid_21 = None
            hidden_states_153 = torch._C._nn.linear(
                hidden_states_152,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_152 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_21_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_154 = hidden_states_149 + hidden_states_153
            hidden_states_149 = hidden_states_153 = None
            hidden_states_155 = torch.nn.functional.layer_norm(
                hidden_states_154,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm1_parameters_bias_
            ) = None
            linear_132 = torch._C._nn.linear(
                hidden_states_155,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_44 = linear_132 * 0.125
            linear_132 = None
            linear_133 = torch._C._nn.linear(
                hidden_states_155,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_154 = linear_133.view(2, -1, 16, 64)
            linear_133 = None
            transpose_111 = view_154.transpose(1, 2)
            view_154 = None
            key_states_44 = transpose_111.contiguous()
            transpose_111 = None
            linear_134 = torch._C._nn.linear(
                hidden_states_155,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_155 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_155 = linear_134.view(2, -1, 16, 64)
            linear_134 = None
            transpose_112 = view_155.transpose(1, 2)
            view_155 = None
            value_states_44 = transpose_112.contiguous()
            transpose_112 = None
            view_156 = query_states_44.view(2, 257, 16, 64)
            query_states_44 = None
            transpose_113 = view_156.transpose(1, 2)
            view_156 = None
            contiguous_68 = transpose_113.contiguous()
            transpose_113 = None
            query_states_45 = contiguous_68.view(32, -1, 64)
            contiguous_68 = None
            key_states_45 = key_states_44.view(32, -1, 64)
            key_states_44 = None
            value_states_45 = value_states_44.view(32, -1, 64)
            value_states_44 = None
            transpose_114 = key_states_45.transpose(1, 2)
            key_states_45 = None
            attn_weights_44 = torch.bmm(query_states_45, transpose_114)
            query_states_45 = transpose_114 = None
            attn_weights_45 = torch.nn.functional.softmax(attn_weights_44, dim=-1)
            attn_weights_44 = None
            attn_probs_22 = torch.nn.functional.dropout(attn_weights_45, p=0.0, training=False)
            attn_weights_45 = None
            attn_output_110 = torch.bmm(attn_probs_22, value_states_45)
            attn_probs_22 = value_states_45 = None
            attn_output_111 = attn_output_110.view(2, 16, 257, 64)
            attn_output_110 = None
            attn_output_112 = attn_output_111.transpose(1, 2)
            attn_output_111 = None
            attn_output_113 = attn_output_112.reshape(2, 257, 1024)
            attn_output_112 = None
            attn_output_114 = torch._C._nn.linear(
                attn_output_113,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_113 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_156 = hidden_states_154 + attn_output_114
            hidden_states_154 = attn_output_114 = None
            hidden_states_157 = torch.nn.functional.layer_norm(
                hidden_states_156,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_158 = torch._C._nn.linear(
                hidden_states_157,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_157 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_67 = 1.702 * hidden_states_158
            sigmoid_22 = torch.sigmoid(mul_67)
            mul_67 = None
            hidden_states_159 = hidden_states_158 * sigmoid_22
            hidden_states_158 = sigmoid_22 = None
            hidden_states_160 = torch._C._nn.linear(
                hidden_states_159,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_159 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_22_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_161 = hidden_states_156 + hidden_states_160
            hidden_states_156 = hidden_states_160 = None
            hidden_states_162 = torch.nn.functional.layer_norm(
                hidden_states_161,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm1_parameters_bias_
            ) = None
            linear_138 = torch._C._nn.linear(
                hidden_states_162,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_q_proj_parameters_bias_
            ) = None
            query_states_46 = linear_138 * 0.125
            linear_138 = None
            linear_139 = torch._C._nn.linear(
                hidden_states_162,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_k_proj_parameters_bias_
            ) = None
            view_161 = linear_139.view(2, -1, 16, 64)
            linear_139 = None
            transpose_116 = view_161.transpose(1, 2)
            view_161 = None
            key_states_46 = transpose_116.contiguous()
            transpose_116 = None
            linear_140 = torch._C._nn.linear(
                hidden_states_162,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_,
            )
            hidden_states_162 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_v_proj_parameters_bias_
            ) = None
            view_162 = linear_140.view(2, -1, 16, 64)
            linear_140 = None
            transpose_117 = view_162.transpose(1, 2)
            view_162 = None
            value_states_46 = transpose_117.contiguous()
            transpose_117 = None
            view_163 = query_states_46.view(2, 257, 16, 64)
            query_states_46 = None
            transpose_118 = view_163.transpose(1, 2)
            view_163 = None
            contiguous_71 = transpose_118.contiguous()
            transpose_118 = None
            query_states_47 = contiguous_71.view(32, -1, 64)
            contiguous_71 = None
            key_states_47 = key_states_46.view(32, -1, 64)
            key_states_46 = None
            value_states_47 = value_states_46.view(32, -1, 64)
            value_states_46 = None
            transpose_119 = key_states_47.transpose(1, 2)
            key_states_47 = None
            attn_weights_46 = torch.bmm(query_states_47, transpose_119)
            query_states_47 = transpose_119 = None
            attn_weights_47 = torch.nn.functional.softmax(attn_weights_46, dim=-1)
            attn_weights_46 = None
            attn_probs_23 = torch.nn.functional.dropout(attn_weights_47, p=0.0, training=False)
            attn_weights_47 = None
            attn_output_115 = torch.bmm(attn_probs_23, value_states_47)
            attn_probs_23 = value_states_47 = None
            attn_output_116 = attn_output_115.view(2, 16, 257, 64)
            attn_output_115 = None
            attn_output_117 = attn_output_116.transpose(1, 2)
            attn_output_116 = None
            attn_output_118 = attn_output_117.reshape(2, 257, 1024)
            attn_output_117 = None
            attn_output_119 = torch._C._nn.linear(
                attn_output_118,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_,
            )
            attn_output_118 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_self_attn_modules_out_proj_parameters_bias_
            ) = None
            hidden_states_163 = hidden_states_161 + attn_output_119
            attn_output_119 = None
            hidden_states_164 = torch.nn.functional.layer_norm(
                hidden_states_163,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_,
                1e-05,
            )
            l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_weight_ = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_layer_norm2_parameters_bias_
            ) = None
            hidden_states_165 = torch._C._nn.linear(
                hidden_states_164,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_,
            )
            hidden_states_164 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc1_parameters_bias_
            ) = None
            mul_70 = 1.702 * hidden_states_165
            sigmoid_23 = torch.sigmoid(mul_70)
            mul_70 = None
            hidden_states_166 = hidden_states_165 * sigmoid_23
            hidden_states_165 = sigmoid_23 = None
            hidden_states_167 = torch._C._nn.linear(
                hidden_states_166,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_,
            )
            hidden_states_166 = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_weight_
            ) = (
                l_self_modules_vision_encoder_modules_vision_model_modules_encoder_modules_layers_modules_23_modules_mlp_modules_fc2_parameters_bias_
            ) = None
            hidden_states_168 = hidden_states_163 + hidden_states_167
            hidden_states_163 = hidden_states_167 = None
            pooled_output = hidden_states_168[(slice(None, None, None), 0, slice(None, None, None))]
            hidden_states_168 = None
            pooled_output_1 = torch.nn.functional.layer_norm(
                pooled_output,
                (1024,),
                l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_weight_,
                l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_bias_,
                1e-05,
            )
            pooled_output = (
                l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_weight_
            ) = l_self_modules_vision_encoder_modules_vision_model_modules_post_layernorm_parameters_bias_ = (
                pooled_output_1
            ) = None
            vision_x_2 = einops.einops.rearrange(hidden_states_161, "(b T F) v d -> b T F v d", b=2, T=1, F=1)
            hidden_states_161 = None
            vision_x_3 = vision_x_2[
                (slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(1, None, None))
            ]
            vision_x_2 = None
            return (vision_x_3,)

    inputs = [
        torch.randn(
            size=torch.Size([2, 1, 1, 3, 224, 224]),
            dtype=torch.float32,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 3, 14, 14]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([257, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([4096, 1024]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([4096]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024, 4096]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randn(
            size=torch.Size([1024]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
        torch.randint(
            low=0,
            high=256,
            size=torch.Size([1, 257]),
            dtype=torch.int64,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g1")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g3():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_media_features_: torch.Tensor, L_input_ids_: torch.Tensor):
            l_media_features_ = L_media_features_
            l_input_ids_ = L_input_ids_
            media_features = l_media_features_.view(2, -1, 5120)
            l_media_features_ = None
            eq = l_input_ids_ == 32005
            l_input_ids_ = None
            to = eq.to(torch.int64)
            eq = None
            sort = to.sort(dim=-1, descending=True, stable=True)
            to = None
            sorted_media_end_positions_mask = sort[0]
            media_end_positions_mask_sort_idx = sort[1]
            sort = None
            sorted_media_end_positions_mask_1 = sorted_media_end_positions_mask[
                (slice(None, None, None), slice(None, 1, None))
            ]
            sorted_media_end_positions_mask = None
            media_end_positions_mask_sort_idx_1 = media_end_positions_mask_sort_idx[
                (slice(None, None, None), slice(None, 1, None))
            ]
            media_end_positions_mask_sort_idx = None
            to_1 = sorted_media_end_positions_mask_1.to(torch.bool)
            sub = media_end_positions_mask_sort_idx_1 - 256
            media_end_positions_mask_sort_idx_1 = None
            add = sub + 1
            sub = None
            padded_media_indices = torch.where(to_1, add, 384)
            to_1 = add = None
            return (media_features, sorted_media_end_positions_mask_1, padded_media_indices)

    inputs = [
        torch.randn(
            size=torch.Size([2, 1, 1, 256, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randint(
            low=0,
            high=32005,
            size=torch.Size([2, 384]),
            dtype=torch.int64,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g3")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g4():
    class DynamoModule(torch.nn.Module):
        def forward(
            self, L_padded_media_indices_: torch.Tensor, L_inputs_embeds_: torch.Tensor, L_media_features_: torch.Tensor
        ):
            l_padded_media_indices_ = L_padded_media_indices_
            l_inputs_embeds_ = L_inputs_embeds_
            l_media_features_ = L_media_features_
            unsqueeze = l_padded_media_indices_.unsqueeze(-1)
            l_padded_media_indices_ = None
            arange = torch.arange(256, device=torch.device(type="cuda", index=0))
            repeat = arange.repeat(2, 1, 1)
            arange = None
            padded_media_indices = unsqueeze + repeat
            unsqueeze = repeat = None
            padded_media_indices_1 = padded_media_indices.reshape(2, -1)
            padded_media_indices = None
            padded_media_indices_2 = einops.einops.repeat(padded_media_indices_1, "b s -> b s h", h=5120)
            padded_media_indices_1 = None
            zeros = torch.zeros((2, 256, 5120), device=torch.device(type="cuda", index=0))
            updated_input_embeds = torch.cat((l_inputs_embeds_, zeros), dim=1)
            l_inputs_embeds_ = zeros = None
            updated_input_embeds_1 = updated_input_embeds.type(torch.bfloat16)
            updated_input_embeds = None
            scatter_ = updated_input_embeds_1.scatter_(1, padded_media_indices_2, l_media_features_)
            l_media_features_ = scatter_ = None
            updated_input_embeds_2 = updated_input_embeds_1[(slice(None, None, None), slice(None, 384, None))]
            updated_input_embeds_1 = None
            return (padded_media_indices_2, updated_input_embeds_2)

    inputs = [
        torch.tensor(data=64, dtype=torch.int64, device="cuda:0", requires_grad=False).broadcast_to(torch.Size([2, 1])),
        torch.randn(
            size=torch.Size([2, 384, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([2, 256, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g4")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g5():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_stack0_: torch.Tensor):
            l_stack0_ = L_stack0_
            transpose = l_stack0_.transpose(0, 1)
            l_stack0_ = None
            embeddings = transpose.contiguous()
            transpose = None
            embeddings_1 = torch.nn.functional.dropout(embeddings, 0.0, False, False)
            embeddings = None
            return (embeddings_1,)

    inputs = [
        torch.randn(
            size=torch.Size([2, 384, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g5")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g6():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_self_modules_rotary_pos_emb_buffers_inv_freq_: torch.Tensor):
            l_self_modules_rotary_pos_emb_buffers_inv_freq_ = L_self_modules_rotary_pos_emb_buffers_inv_freq_
            arange = torch.arange(384, device=torch.device(type="cuda", index=0))
            seq = arange + 0
            arange = None
            seq_1 = seq.type_as(l_self_modules_rotary_pos_emb_buffers_inv_freq_)
            seq = None
            freqs = torch.functional.einsum("i , j -> i j", seq_1, l_self_modules_rotary_pos_emb_buffers_inv_freq_)
            seq_1 = l_self_modules_rotary_pos_emb_buffers_inv_freq_ = None
            emb = torch.cat((freqs, freqs), dim=-1)
            freqs = None
            rotary_pos_emb = einops.einops.rearrange(emb, "n d -> n 1 1 d")
            emb = None
            return (rotary_pos_emb,)

    inputs = [
        torch.randn(
            size=torch.Size([64]), dtype=torch.bfloat16, layout=torch.strided, device="cuda:0", requires_grad=False
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g6")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g12():
    class DynamoModule(torch.nn.Module):
        def forward(
            self,
            L_self_modules_query_key_value_parameters_weight_: torch.nn.parameter.Parameter,
            L_hidden_states_: torch.Tensor,
        ):
            l_self_modules_query_key_value_parameters_weight_ = L_self_modules_query_key_value_parameters_weight_
            l_hidden_states_ = L_hidden_states_
            function_ctx = torch.autograd.function.FunctionCtx()
            function_ctx = None
            function_ctx_1 = torch.autograd.function.FunctionCtx()
            function_ctx_1 = None
            t = l_self_modules_query_key_value_parameters_weight_.t()
            l_self_modules_query_key_value_parameters_weight_ = None
            output = torch.matmul(l_hidden_states_, t)
            l_hidden_states_ = t = None
            mixed_x_layer = output.view(384, 2, 40, 384)
            output = None
            split = torch.functional.split(mixed_x_layer, 128, dim=3)
            mixed_x_layer = None
            chunk = split[0]
            chunk_1 = split[1]
            chunk_2 = split[2]
            split = None
            query_layer = chunk.contiguous()
            chunk = None
            key_layer = chunk_1.contiguous()
            chunk_1 = None
            value_layer = chunk_2.contiguous()
            chunk_2 = None
            return (query_layer, key_layer, value_layer)

    inputs = [
        torch.randn(
            size=torch.Size([15360, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g12")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g13():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_query_layer_: torch.Tensor, L_key_layer_: torch.Tensor, L_rotary_pos_emb_0_: torch.Tensor):
            l_query_layer_ = L_query_layer_
            l_key_layer_ = L_key_layer_
            l_rotary_pos_emb_0_ = L_rotary_pos_emb_0_
            t = l_query_layer_[(Ellipsis, slice(None, 128, None))]
            t_pass = l_query_layer_[(Ellipsis, slice(128, None, None))]
            l_query_layer_ = None
            cos = l_rotary_pos_emb_0_.cos()
            mul = t * cos
            cos = None
            x = einops.einops.rearrange(t, "... (j d) -> ... j d", j=2)
            t = None
            unbind = x.unbind(dim=-2)
            x = None
            x1 = unbind[0]
            x2 = unbind[1]
            unbind = None
            neg = -x2
            x2 = None
            cat = torch.cat((neg, x1), dim=-1)
            neg = x1 = None
            sin = l_rotary_pos_emb_0_.sin()
            mul_1 = cat * sin
            cat = sin = None
            t_1 = mul + mul_1
            mul = mul_1 = None
            query_layer = torch.cat((t_1, t_pass), dim=-1)
            t_1 = t_pass = None
            t_2 = l_key_layer_[(Ellipsis, slice(None, 128, None))]
            t_pass_1 = l_key_layer_[(Ellipsis, slice(128, None, None))]
            l_key_layer_ = None
            cos_1 = l_rotary_pos_emb_0_.cos()
            mul_2 = t_2 * cos_1
            cos_1 = None
            x_1 = einops.einops.rearrange(t_2, "... (j d) -> ... j d", j=2)
            t_2 = None
            unbind_1 = x_1.unbind(dim=-2)
            x_1 = None
            x1_1 = unbind_1[0]
            x2_1 = unbind_1[1]
            unbind_1 = None
            neg_1 = -x2_1
            x2_1 = None
            cat_2 = torch.cat((neg_1, x1_1), dim=-1)
            neg_1 = x1_1 = None
            sin_1 = l_rotary_pos_emb_0_.sin()
            l_rotary_pos_emb_0_ = None
            mul_3 = cat_2 * sin_1
            cat_2 = sin_1 = None
            t_3 = mul_2 + mul_3
            mul_2 = mul_3 = None
            key_layer = torch.cat((t_3, t_pass_1), dim=-1)
            t_3 = t_pass_1 = None
            return (query_layer, key_layer)

    inputs = [
        torch.randn(
            size=torch.Size([384, 2, 40, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 40, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 1, 1, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g13")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g14():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_query_layer_: torch.Tensor, L_key_layer_: torch.Tensor, L_value_layer_: torch.Tensor):
            l_query_layer_ = L_query_layer_
            l_key_layer_ = L_key_layer_
            l_value_layer_ = L_value_layer_
            query_layer = einops.einops.rearrange(l_query_layer_, "sq b np hn -> (b np) sq hn")
            l_query_layer_ = None
            key_layer = einops.einops.rearrange(l_key_layer_, "sk b np hn -> (b np) hn sk")
            l_key_layer_ = None
            value_layer = einops.einops.rearrange(l_value_layer_, "sv b np hn -> (b np) sv hn")
            l_value_layer_ = None
            matmul_input_buffer = torch.empty(
                80, 384, 384, dtype=torch.bfloat16, device=torch.device(type="cuda", index=0)
            )
            matmul_result = torch.baddbmm(
                matmul_input_buffer, query_layer, key_layer, beta=0.0, alpha=0.08838834764831843
            )
            matmul_input_buffer = query_layer = key_layer = None
            attention_scores = matmul_result.view(2, 40, 384, 384)
            matmul_result = None
            return (attention_scores, value_layer)

    inputs = [
        torch.randn(
            size=torch.Size([384, 2, 40, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 40, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 40, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g14")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g18():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_tensor_: torch.Tensor):
            l_tensor_ = L_tensor_
            reduce = einops.einops.reduce(l_tensor_, "b np sq sk -> (b np) sq sk", reduction="rearrange")
            l_tensor_ = None
            return (reduce,)

    inputs = [
        torch.randn(
            size=torch.Size([2, 40, 384, 384]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g18")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g19():
    class DynamoModule(torch.nn.Module):
        def forward(self, L_tensor_: torch.Tensor):
            l_tensor_ = L_tensor_
            reduce = einops.einops.reduce(l_tensor_, "(b np) sq hn -> b np sq hn", reduction="rearrange", np=40)
            l_tensor_ = None
            return (reduce,)

    inputs = [
        torch.randn(
            size=torch.Size([80, 384, 128]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g19")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g23():
    class DynamoModule(torch.nn.Module):
        def forward(
            self,
            L_self_modules_dense_h_to_4h_parameters_weight_: torch.nn.parameter.Parameter,
            L_hidden_states_: torch.Tensor,
        ):
            l_self_modules_dense_h_to_4h_parameters_weight_ = L_self_modules_dense_h_to_4h_parameters_weight_
            l_hidden_states_ = L_hidden_states_
            function_ctx = torch.autograd.function.FunctionCtx()
            function_ctx = None
            function_ctx_1 = torch.autograd.function.FunctionCtx()
            function_ctx_1 = None
            t = l_self_modules_dense_h_to_4h_parameters_weight_.t()
            l_self_modules_dense_h_to_4h_parameters_weight_ = None
            output = torch.matmul(l_hidden_states_, t)
            l_hidden_states_ = t = None
            chunk = torch.chunk(output, 2, dim=-1)
            output = None
            intermediate_parallel = chunk[0]
            intermediate_parallel_2 = chunk[1]
            chunk = None
            silu = torch.nn.functional.silu(intermediate_parallel)
            intermediate_parallel = None
            intermediate_parallel_3 = silu * intermediate_parallel_2
            silu = intermediate_parallel_2 = None
            return (intermediate_parallel_3,)

    inputs = [
        torch.randn(
            size=torch.Size([27648, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g23")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def g25():
    """Also relevant outside NeVA"""

    class DynamoModule(torch.nn.Module):
        def forward(self, L_stack0_0_: torch.Tensor, L_layernorm_input_: torch.Tensor):
            l_stack0_0_ = L_stack0_0_
            l_layernorm_input_ = L_layernorm_input_
            out = torch.nn.functional.dropout(l_stack0_0_, p=0.0, training=False)
            l_stack0_0_ = None
            out_1 = l_layernorm_input_ + out
            l_layernorm_input_ = out = None
            return (out_1,)

    inputs = [
        torch.randn(
            size=torch.Size([384, 2, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
        torch.randn(
            size=torch.Size([384, 2, 5120]),
            dtype=torch.bfloat16,
            layout=torch.strided,
            device="cuda:0",
            requires_grad=False,
        ),
    ]
    fqn = thunder.jit(DynamoModule())
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("g25")
    fqn(*inputs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


def test_g1(benchmark):
    benchmark(g1)


def test_g3(benchmark):
    benchmark(g3)


def test_g4(benchmark):
    benchmark(g4)


def test_g5(benchmark):
    benchmark(g5)


def test_g6(benchmark):
    benchmark(g6)


def test_g12(benchmark):
    benchmark(g12)


def test_g13(benchmark):
    benchmark(g13)


def test_g14(benchmark):
    benchmark(g14)


def test_g18(benchmark):
    benchmark(g18)


def test_g19(benchmark):
    benchmark(g19)


def test_g23(benchmark):
    benchmark(g23)


def test_g25(benchmark):
    benchmark(g25)
