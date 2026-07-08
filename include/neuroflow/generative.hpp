#ifndef NEUROFLOW_GENERATIVE_HPP
#define NEUROFLOW_GENERATIVE_HPP

/**
 * NeuroFlow 生成式语言模型模块 - 聚合头文件
 *
 * 为类脑模块化网络增加因果语言模型能力：
 * - CausalLMHead: 词嵌入+因果门控+SAE+NTM+投影输出
 * - Tokenizer: BPE/WordPiece分词器
 * - SamplingStrategy: 贪心/Top-K/Top-P采样
 * - GenerativeModel: 编排层，与ECN/DMN/SN协同
 */

#include "causal_lm.hpp"
#include "tokenizer.hpp"
#include "sampling.hpp"
#include "generative_model.hpp"

#endif // NEUROFLOW_GENERATIVE_HPP
