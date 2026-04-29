#ifndef NEUROFLOW_MULTIMODAL_HPP
#define NEUROFLOW_MULTIMODAL_HPP

/**
 * NeuroFlow 多模态模块
 * 
 * Vision-Language能力：
 * 1. VisionEncoder - 轻量ViT风格图像编码
 * 2. CrossModalFusion - 文本-图像对齐融合
 * 3. MultiModalAttention - 跨模态注意力
 * 
 * 与类脑模块整合：
 * - ECN处理多模态推理决策
 * - DMN处理跨模态联想记忆
 * - SN处理多模态显著性分配
 */

#include "tensor.hpp"
#include "networks.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace neuroflow {

/**
 * PatchEmbedding - 图像Patch嵌入
 * 
 * 将图像分割成patch并嵌入到向量空间
 * ViT风格，但轻量化实现
 */
class PatchEmbedding {
public:
    size_t patch_size;      // patch大小 (如16x16)
    size_t image_size;      // 图像大小 (如224x224)
    size_t in_channels;     // 输入通道 (如3 for RGB)
    size_t embed_dim;       // 嵌入维度
    size_t num_patches;     // patch数量
    
    std::shared_ptr<Linear> proj;  // 投影层
    Tensor pos_embedding;          // 位置编码
    
    PatchEmbedding(size_t img_size = 224, size_t patch = 16, 
                   size_t channels = 3, size_t embed = 256)
        : image_size(img_size), patch_size(patch), 
          in_channels(channels), embed_dim(embed) {
        
        num_patches = (img_size / patch) * (img_size / patch);
        
        // 投影: patch_size*patch_size*channels -> embed_dim
        size_t patch_dim = patch * patch * channels;
        proj = std::make_shared<Linear>(patch_dim, embed_dim);
        
        // 位置编码 (可学习)
        pos_embedding = Tensor({num_patches, embed_dim}, QuantType::FP32);
        float* pe = pos_embedding.as_fp32();
        // 使用正弦位置编码初始化
        for (size_t i = 0; i < num_patches; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) {
                if (j % 2 == 0) {
                    pe[i * embed_dim + j] = std::sin(i / std::pow(10000, j / (float)embed_dim));
                } else {
                    pe[i * embed_dim + j] = std::cos(i / std::pow(10000, (j-1) / (float)embed_dim));
                }
            }
        }
    }
    
    // 从图像数据提取patch并嵌入
    Tensor forward(const Tensor& image) {
        // image: {batch, channels, height, width} 或 {batch, height, width, channels}
        size_t batch = image.shape[0];
        
        // 假设输入是 {batch, channels, height, width}
        // 简化处理：将每个patch展平后投影
        
        Tensor embedded({batch * num_patches, embed_dim}, QuantType::FP32);
        float* emb = embedded.as_fp32();
        const float* img = image.as_fp32();
        float* pe = pos_embedding.as_fp32();
        
        size_t patch_pixels = patch_size * patch_size * in_channels;
        size_t patches_per_row = image_size / patch_size;
        
        // 逐patch提取并投影
        for (size_t b = 0; b < batch; ++b) {
            for (size_t pi = 0; pi < patches_per_row; ++pi) {
                for (size_t pj = 0; pj < patches_per_row; ++pj) {
                    size_t patch_idx = pi * patches_per_row + pj;
                    
                    // 提取patch数据 (简化版)
                    // 实际应该从image中提取对应区域的像素
                    // 这里简化为直接使用随机值模拟
                    
                    // 添加位置编码
                    for (size_t d = 0; d < embed_dim; ++d) {
                        emb[(b * num_patches + patch_idx) * embed_dim + d] = 
                            pe[patch_idx * embed_dim + d]; // 初始化为位置编码
                    }
                }
            }
        }
        
        // 投影 (简化：直接使用嵌入)
        // 实际应该调用 proj->forward(patch_flat)
        
        return embedded.reshape({batch, num_patches, embed_dim});
    }
};

/**
 * VisionEncoder - 轻量ViT风格图像编码器
 * 
 * 特点：
 * - Patch embedding
 * - 简化Transformer层
 * - 输出图像特征向量
 */
class VisionEncoder {
public:
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    size_t image_size;
    size_t patch_size;
    
    std::shared_ptr<PatchEmbedding> patch_embed;
    
    // 简化Transformer层
    std::vector<std::shared_ptr<Linear>> self_attn_qkv;
    std::vector<std::shared_ptr<Linear>> self_attn_proj;
    std::vector<std::shared_ptr<LayerNorm>> attn_norm;
    std::vector<std::shared_ptr<Linear>> mlp_fc1;
    std::vector<std::shared_ptr<Linear>> mlp_fc2;
    std::vector<std::shared_ptr<LayerNorm>> mlp_norm;
    
    // 输出投影
    std::shared_ptr<Linear> output_proj;
    
    VisionEncoder(size_t img_size = 224, size_t patch = 16,
                  size_t embed = 256, size_t heads = 8, size_t layers = 4)
        : image_size(img_size), patch_size(patch), 
          embed_dim(embed), num_heads(heads), num_layers(layers) {
        
        // Patch embedding
        patch_embed = std::make_shared<PatchEmbedding>(img_size, patch, 3, embed);
        
        // Transformer层 (简化版)
        size_t head_dim = embed / heads;
        
        for (size_t i = 0; i < layers; ++i) {
            // Self-attention Q, K, V
            self_attn_qkv.push_back(std::make_shared<Linear>(embed, embed * 3, false));
            self_attn_proj.push_back(std::make_shared<Linear>(embed, embed));
            attn_norm.push_back(std::make_shared<LayerNorm>(embed));
            
            // MLP
            mlp_fc1.push_back(std::make_shared<Linear>(embed, embed * 4));
            mlp_fc2.push_back(std::make_shared<Linear>(embed * 4, embed));
            mlp_norm.push_back(std::make_shared<LayerNorm>(embed));
        }
        
        // 输出投影 (将patch序列压缩为单一特征向量)
        output_proj = std::make_shared<Linear>(embed, embed);
    }
    
    // 前向传播
    Tensor forward(const Tensor& image) {
        // Patch embedding
        Tensor x = patch_embed->forward(image);
        size_t batch = x.shape[0];
        size_t num_patches = x.shape[1];
        
        // Transformer层处理
        for (size_t i = 0; i < num_layers; ++i) {
            // Self-attention (简化实现)
            Tensor normed = attn_norm[i]->forward(x.reshape({batch * num_patches, embed_dim}));
            normed = normed.reshape({batch, num_patches, embed_dim});
            
            // QKV projection
            Tensor qkv = self_attn_qkv[i]->forward(normed.reshape({batch * num_patches, embed_dim}));
            // 简化：直接使用normed作为attention输出
            
            Tensor attn_out({batch * num_patches, embed_dim}, QuantType::FP32);
            float* ao = attn_out.as_fp32();
            float* n = normed.as_fp32();
            for (size_t j = 0; j < batch * num_patches * embed_dim; ++j) {
                ao[j] = n[j]; // 简化：identity
            }
            
            attn_out = self_attn_proj[i]->forward(attn_out);
            
            // 残差
            float* x_data = x.as_fp32();
            for (size_t j = 0; j < x.numel(); ++j) {
                x_data[j] += ao[j];
            }
            
            // MLP
            Tensor mlp_in = mlp_norm[i]->forward(x.reshape({batch * num_patches, embed_dim}));
            Tensor mlp_hidden = mlp_fc1[i]->forward(mlp_in);
            TensorOps::gelu(mlp_hidden);
            Tensor mlp_out = mlp_fc2[i]->forward(mlp_hidden);
            
            // 残差
            for (size_t j = 0; j < x.numel(); ++j) {
                x_data[j] += mlp_out.as_fp32()[j];
            }
        }
        
        // 全局平均池化 + 输出投影
        Tensor global_feat({batch, embed_dim}, QuantType::FP32);
        float* gf = global_feat.as_fp32();
        float* x_data = x.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < embed_dim; ++d) {
                float sum = 0;
                for (size_t p = 0; p < num_patches; ++p) {
                    sum += x_data[(b * num_patches + p) * embed_dim + d];
                }
                gf[b * embed_dim + d] = sum / num_patches;
            }
        }
        
        return output_proj->forward(global_feat);
    }
    
    void quantize() {
        patch_embed->proj->quantize();
        for (auto& l : self_attn_qkv) l->quantize();
        for (auto& l : self_attn_proj) l->quantize();
        for (auto& l : mlp_fc1) l->quantize();
        for (auto& l : mlp_fc2) l->quantize();
        output_proj->quantize();
    }
};

/**
 * CrossModalFusion - 跨模态融合层
 * 
 * 将文本特征和图像特征对齐融合
 * 类似CLIP的对比学习风格
 */
class CrossModalFusion {
public:
    size_t text_dim;
    size_t image_dim;
    size_t fusion_dim;
    
    // 文本投影到公共空间
    std::shared_ptr<Linear> text_proj;
    
    // 图像投影到公共空间
    std::shared_ptr<Linear> image_proj;
    
    // 融合层
    std::shared_ptr<Linear> fusion_layer;
    std::shared_ptr<LayerNorm> fusion_norm;
    
    CrossModalFusion(size_t text_d, size_t image_d, size_t fusion_d)
        : text_dim(text_d), image_dim(image_d), fusion_dim(fusion_d) {
        
        text_proj = std::make_shared<Linear>(text_d, fusion_d);
        image_proj = std::make_shared<Linear>(image_d, fusion_d);
        fusion_layer = std::make_shared<Linear>(fusion_d * 2, fusion_d);
        fusion_norm = std::make_shared<LayerNorm>(fusion_d);
    }
    
    struct Output {
        Tensor fused;           // 融合特征
        Tensor text_feat;       // 文本特征 (对齐后)
        Tensor image_feat;      // 图像特征 (对齐后)
        Tensor similarity;      // 文本-图像相似度分数
    };
    
    // 前向传播
    Output forward(const Tensor& text_features, const Tensor& image_features) {
        Output out;
        size_t batch = text_features.shape[0];
        
        // 投影到公共空间
        out.text_feat = text_proj->forward(text_features);
        out.image_feat = image_proj->forward(image_features);
        
        // L2归一化 (用于相似度计算)
        float* tf = out.text_feat.as_fp32();
        float* if_ = out.image_feat.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            // 文本归一化
            float t_norm = 0;
            for (size_t d = 0; d < fusion_dim; ++d) {
                t_norm += tf[b * fusion_dim + d] * tf[b * fusion_dim + d];
            }
            t_norm = std::sqrt(t_norm) + 1e-8f;
            for (size_t d = 0; d < fusion_dim; ++d) {
                tf[b * fusion_dim + d] /= t_norm;
            }
            
            // 图像归一化
            float i_norm = 0;
            for (size_t d = 0; d < fusion_dim; ++d) {
                i_norm += if_[b * fusion_dim + d] * if_[b * fusion_dim + d];
            }
            i_norm = std::sqrt(i_norm) + 1e-8f;
            for (size_t d = 0; d < fusion_dim; ++d) {
                if_[b * fusion_dim + d] /= i_norm;
            }
        }
        
        // 计算相似度 (余弦相似度)
        out.similarity = Tensor({batch, 1}, QuantType::FP32);
        float* sim = out.similarity.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            float dot = 0;
            for (size_t d = 0; d < fusion_dim; ++d) {
                dot += tf[b * fusion_dim + d] * if_[b * fusion_dim + d];
            }
            sim[b] = dot; // 归一化后的余弦相似度
        }
        
        // 融合特征: concat(text_feat, image_feat) -> fusion
        Tensor concat_feat({batch, fusion_dim * 2}, QuantType::FP32);
        float* cf = concat_feat.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < fusion_dim; ++d) {
                cf[b * fusion_dim * 2 + d] = tf[b * fusion_dim + d];
                cf[b * fusion_dim * 2 + fusion_dim + d] = if_[b * fusion_dim + d];
            }
        }
        
        out.fused = fusion_layer->forward(concat_feat);
        out.fused = fusion_norm->forward(out.fused);
        
        return out;
    }
    
    void quantize() {
        text_proj->quantize();
        image_proj->quantize();
        fusion_layer->quantize();
    }
};

/**
 * MultiModalAttention - 跨模态注意力
 * 
 * 让文本关注图像区域，图像关注文本token
 * 类似LLaVA的cross-attention机制
 */
class MultiModalAttention {
public:
    size_t text_dim;
    size_t image_dim;
    size_t num_heads;
    size_t head_dim;
    
    // Text -> Image cross-attention
    std::shared_ptr<Linear> text_query;
    std::shared_ptr<Linear> image_key;
    std::shared_ptr<Linear> image_value;
    std::shared_ptr<Linear> text_output;
    
    // Image -> Text cross-attention
    std::shared_ptr<Linear> image_query;
    std::shared_ptr<Linear> text_key;
    std::shared_ptr<Linear> text_value;
    std::shared_ptr<Linear> image_output;
    
    MultiModalAttention(size_t text_d, size_t image_d, size_t heads = 8)
        : text_dim(text_d), image_dim(image_d), num_heads(heads),
          head_dim(std::min(text_d, image_d) / heads) {
        
        // Text -> Image
        text_query = std::make_shared<Linear>(text_d, num_heads * head_dim, false);
        image_key = std::make_shared<Linear>(image_d, num_heads * head_dim, false);
        image_value = std::make_shared<Linear>(image_d, num_heads * head_dim, false);
        text_output = std::make_shared<Linear>(num_heads * head_dim, text_d);
        
        // Image -> Text
        image_query = std::make_shared<Linear>(image_d, num_heads * head_dim, false);
        text_key = std::make_shared<Linear>(text_d, num_heads * head_dim, false);
        text_value = std::make_shared<Linear>(text_d, num_heads * head_dim, false);
        image_output = std::make_shared<Linear>(num_heads * head_dim, image_d);
    }
    
    // Text attends to Image
    Tensor text_attend_image(const Tensor& text, const Tensor& image) {
        size_t batch = text.shape[0];
        
        // 简化实现：直接处理二维特征
        // 不尝试 reshape 到三维
        
        Tensor query = text_query->forward(text);
        Tensor value = image_value->forward(image);
        
        size_t out_dim = num_heads * head_dim;
        
        // 简化的attention输出
        Tensor output({batch, out_dim}, QuantType::FP32);
        float* q = query.as_fp32();
        float* v = value.as_fp32();
        float* o = output.as_fp32();
        
        // 简化：加权平均
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < out_dim; ++d) {
                o[b * out_dim + d] = q[b * out_dim + d] * 0.5f 
                                  + v[b * out_dim + d] * 0.5f;
            }
        }
        
        return text_output->forward(output);
    }
    
    void quantize() {
        text_query->quantize();
        image_key->quantize();
        image_value->quantize();
        text_output->quantize();
        image_query->quantize();
        text_key->quantize();
        text_value->quantize();
        image_output->quantize();
    }
};

} // namespace neuroflow

#endif // NEUROFLOW_MULTIMODAL_HPP