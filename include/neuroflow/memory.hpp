#ifndef NEUROFLOW_MEMORY_HPP
#define NEUROFLOW_MEMORY_HPP

/**
 * NeuroFlow 记忆系统
 * 
 * 核心技术：
 * 1. MLA (Multi-head Latent Attention) - DeepSeek KV压缩
 * 2. 滑动窗口长记忆
 * 3. 记忆分页与磁盘溢出
 * 4. 记忆巩固 (LTP模拟)
 */

#include "tensor.hpp"
#include "networks.hpp"
#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <unordered_map>
#include <queue>

namespace neuroflow {

/**
 * MLA压缩KV Cache
 * 
 * DeepSeek核心技术：将KV压缩到潜在空间
 * 内存节省：87.5%+
 */
class LatentKVCache {
public:
    size_t d_model;
    size_t n_heads;
    size_t d_latent;  // 压缩维度
    size_t head_dim;
    
    // 投影矩阵
    std::shared_ptr<Linear> W_q;      // Q投影
    std::shared_ptr<Linear> W_dkv;    // KV压缩投影
    std::shared_ptr<Linear> W_uk;     // K解压
    std::shared_ptr<Linear> W_uv;     // V解压
    std::shared_ptr<Linear> W_o;      // 输出投影
    
    // Cache存储 (压缩形式)
    Tensor cache;                     // (max_seq, d_latent)
    size_t cache_len;
    size_t max_cache_len;
    
    LatentKVCache(size_t model_dim, size_t heads, size_t latent_dim, size_t max_len = 4096)
        : d_model(model_dim), n_heads(heads), d_latent(latent_dim),
          head_dim(model_dim / heads), max_cache_len(max_len), cache_len(0) {
        
        W_q = std::make_shared<Linear>(d_model, d_model, false);
        W_dkv = std::make_shared<Linear>(d_model, d_latent, false);  // 压缩！
        W_uk = std::make_shared<Linear>(d_latent, d_model, false);
        W_uv = std::make_shared<Linear>(d_latent, d_model, false);
        W_o = std::make_shared<Linear>(d_model, d_model, false);
        
        // 初始化cache
        cache = Tensor({max_len, d_latent}, QuantType::FP32);
    }
    
    // 前向传播 (带cache)
    Tensor forward(const Tensor& x, bool use_cache = true) {
        size_t batch = x.shape_[0];
        size_t seq_len = x.shape_.size() > 1 ? x.shape_[1] : 1;
        size_t input_dim = x.shape_.size() > 2 ? x.shape_[2] : (x.shape_.size() > 1 ? x.shape_[1] : d_model);
        
        // 确定实际维度
        if (x.shape_.size() == 2 && x.shape_[1] == d_model) {
            // 输入是 {batch, d_model}，seq_len=1
            seq_len = 1;
            input_dim = d_model;
        } else if (x.shape_.size() == 2) {
            // 输入可能是 {batch, seq_len} 但缺少 d_model
            // 将 seq_len 视为实际序列长度，假设每个位置是 d_model 维
            // 这需要特殊处理
            seq_len = 1;
            input_dim = x.shape_[1];
        }
        
        // Q投影 - 输入需要是 {batch * seq_len, d_model}
        size_t flat_batch = batch * seq_len;
        Tensor x_flat({flat_batch, d_model}, QuantType::FP32);
        float* xf = x_flat.as_fp32();
        const float* xd = x.as_fp32();
        
        // 如果输入维度小于 d_model，补零
        size_t copy_size = std::min(input_dim, d_model);
        for (size_t i = 0; i < flat_batch; ++i) {
            for (size_t j = 0; j < copy_size; ++j) {
                xf[i * d_model + j] = xd[i * input_dim + j];
            }
            for (size_t j = copy_size; j < d_model; ++j) {
                xf[i * d_model + j] = 0.0f;
            }
        }
        
        Tensor q = W_q->forward(x_flat);
        q = q.reshape({batch, seq_len, n_heads, head_dim});
        
        // KV压缩到潜在空间 (MLA核心!)
        Tensor c_kv = W_dkv->forward(x_flat);
        c_kv = c_kv.reshape({batch, seq_len, d_latent});
        
        // 拼接历史cache
        if (use_cache && cache_len > 0) {
            size_t new_len = cache_len + seq_len;
            Tensor new_cache({new_len, d_latent}, QuantType::FP32);
            float* nc = new_cache.as_fp32();
            float* old = cache.as_fp32();
            
            // 拷贝旧cache
            memcpy(nc, old, cache_len * d_latent * sizeof(float));
            
            // 拷贝新cache (取batch=0)
            float* new_kv = c_kv.as_fp32();
            for (size_t s = 0; s < seq_len; ++s) {
                memcpy(nc + (cache_len + s) * d_latent,
                       new_kv + s * d_latent,
                       d_latent * sizeof(float));
            }
            
            c_kv = new_cache.reshape({1, new_len, d_latent});
        }
        
        // 解压K, V
        size_t total_len = use_cache && cache_len > 0 ? (cache_len + seq_len) : seq_len;
        
        // 正确reshape c_kv到二维 - 注意batch维度处理
        // 当有历史cache时，c_kv 被 reshape 到 {1, new_len, d_latent}
        // 需要正确处理batch扩展
        size_t c_kv_batch = use_cache && cache_len > 0 ? 1 : batch;
        size_t actual_elements = c_kv_batch * total_len * d_latent;
        
        Tensor c_kv_flat({batch * total_len, d_latent}, QuantType::FP32);
        float* ckf = c_kv_flat.as_fp32();
        const float* ck = c_kv.as_fp32();
        
        // 正确拷贝：只拷贝实际存在的数据
        for (size_t b = 0; b < batch; ++b) {
            for (size_t t = 0; t < total_len; ++t) {
                for (size_t d = 0; d < d_latent; ++d) {
                    // 当有历史cache时，所有batch共享同一份cache数据
                    size_t src_idx = (c_kv_batch == 1 ? t : b * total_len + t) * d_latent + d;
                    size_t dst_idx = (b * total_len + t) * d_latent + d;
                    ckf[dst_idx] = ck[src_idx];
                }
            }
        }
        
        Tensor k = W_uk->forward(c_kv_flat);
        Tensor v = W_uv->forward(c_kv_flat);
        
        k = k.reshape({batch, total_len, n_heads, head_dim});
        v = v.reshape({batch, total_len, n_heads, head_dim});
        
        // 注意力计算
        Tensor output({batch, seq_len, d_model}, QuantType::FP32);
        float* out = output.as_fp32();
        float* qp = q.as_fp32();
        float* kp = k.as_fp32();
        float* vp = v.as_fp32();
        
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < n_heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    // 计算注意力分数
                    std::vector<float> scores(total_len);
                    for (size_t t = 0; t < total_len; ++t) {
                        float dot = 0;
                        for (size_t d = 0; d < head_dim; ++d) {
                            dot += qp[b * seq_len * n_heads * head_dim + s * n_heads * head_dim + h * head_dim + d]
                                 * kp[b * total_len * n_heads * head_dim + t * n_heads * head_dim + h * head_dim + d];
                        }
                        scores[t] = dot * scale;
                    }
                    
                    // Softmax
                    float max_s = scores[0];
                    for (auto& sc : scores) max_s = std::max(max_s, sc);
                    float sum = 0;
                    for (auto& sc : scores) {
                        sc = std::exp(sc - max_s);
                        sum += sc;
                    }
                    for (auto& sc : scores) sc /= sum;
                    
                    // 加权求和
                    for (size_t d = 0; d < head_dim; ++d) {
                        float val = 0;
                        for (size_t t = 0; t < total_len; ++t) {
                            val += scores[t] * vp[b * total_len * n_heads * head_dim + t * n_heads * head_dim + h * head_dim + d];
                        }
                        out[b * seq_len * d_model + s * d_model + h * head_dim + d] = val;
                    }
                }
            }
        }
        
        output = W_o->forward(output.reshape({batch * seq_len, d_model}));
        output = output.reshape({batch, seq_len, d_model});
        
        // 更新cache
        if (use_cache) {
            float* c = cache.as_fp32();
            float* nk = c_kv.as_fp32();
            // 只保留最新的部分
            size_t keep = std::min(seq_len, max_cache_len - cache_len);
            if (cache_len + seq_len > max_cache_len) {
                // 滑动：丢弃旧的
                size_t shift = cache_len + seq_len - max_cache_len;
                memmove(c, c + shift * d_latent, (cache_len - shift) * d_latent * sizeof(float));
                cache_len -= shift;
            }
            memcpy(c + cache_len * d_latent, nk, seq_len * d_latent * sizeof(float));
            cache_len += seq_len;
        }
        
        return output.reshape({batch, d_model});
    }
    
    // 清空cache
    void clear_cache() {
        cache_len = 0;
        memset(cache.data_.get(), 0, cache.data_size_);
    }
    
    // 获取cache大小 (字节)
    size_t cache_size_bytes() const {
        return cache_len * d_latent * sizeof(float);
    }
    
    // 相比传统KV节省的内存比例
    float memory_saving_ratio() const {
        size_t traditional_size = cache_len * d_model * 2 * sizeof(float);  // K + V
        size_t mla_size = cache_len * d_latent * sizeof(float);
        return 1.0f - static_cast<float>(mla_size) / traditional_size;
    }
};

/**
 * MemoryConsolidationModule
 * 
 * 模拟海马体记忆巩固：
 * 1. Encoding - 记忆编码
 * 2. Retrieval - 注意力检索
 * 3. Consolidation - LTP增强
 */
class MemoryConsolidationModule {
public:
    size_t memory_slots;
    size_t memory_dim;
    float ltp_rate;
    
    // 记忆库
    Tensor memory_bank;  // (slots, dim)
    
    // 投影
    std::shared_ptr<Linear> encode_proj;
    std::shared_ptr<Linear> retrieve_proj;
    std::shared_ptr<Linear> query_proj;
    
    MemoryConsolidationModule(size_t input_dim, size_t slots = 64, size_t dim = 128, float ltp = 0.01f)
        : memory_slots(slots), memory_dim(dim), ltp_rate(ltp) {
        
        memory_bank = Tensor({slots, dim}, QuantType::FP32);
        float* m = memory_bank.as_fp32();
        std::mt19937 init_rng(42);
        std::uniform_real_distribution<float> init_dist(-0.02f, 0.02f);
        for (size_t i = 0; i < memory_bank.numel(); ++i) {
            m[i] = init_dist(init_rng);
        }
        
        encode_proj = std::make_shared<Linear>(input_dim, dim);
        retrieve_proj = std::make_shared<Linear>(dim, input_dim);
        query_proj = std::make_shared<Linear>(input_dim, dim);
    }
    
    // 编码
    Tensor encode(const Tensor& x) {
        return encode_proj->forward(x);
    }
    
    // 检索
    struct RetrievalResult {
        Tensor retrieved;
        Tensor attention;
    };
    
    RetrievalResult retrieve(const Tensor& query) {
        RetrievalResult result;
        
        Tensor q = query_proj->forward(query);  // (batch, dim)
        
        // 注意力: query @ memory_bank.T
        size_t batch = q.shape_[0];
        result.attention = Tensor({batch, memory_slots}, QuantType::FP32);
        
        float* qp = q.as_fp32();
        float* mp = memory_bank.as_fp32();
        float* ap = result.attention.as_fp32();
        
        float scale = 1.0f / std::sqrt(static_cast<float>(memory_dim));
        
        for (size_t b = 0; b < batch; ++b) {
            // 计算分数
            std::vector<float> scores(memory_slots);
            for (size_t s = 0; s < memory_slots; ++s) {
                float dot = 0;
                for (size_t d = 0; d < memory_dim; ++d) {
                    dot += qp[b * memory_dim + d] * mp[s * memory_dim + d];
                }
                scores[s] = dot * scale;
            }
            
            // Softmax
            float max_s = scores[0];
            for (auto& sc : scores) max_s = std::max(max_s, sc);
            float sum = 0;
            for (auto& sc : scores) {
                sc = std::exp(sc - max_s);
                sum += sc;
            }
            for (size_t s = 0; s < memory_slots; ++s) {
                ap[b * memory_slots + s] = scores[s] / sum;
            }
        }
        
        // 检索: attention @ memory_bank
        Tensor retrieved_mem({batch, memory_dim}, QuantType::FP32);
        float* rp = retrieved_mem.as_fp32();
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t d = 0; d < memory_dim; ++d) {
                float val = 0;
                for (size_t s = 0; s < memory_slots; ++s) {
                    val += ap[b * memory_slots + s] * mp[s * memory_dim + d];
                }
                rp[b * memory_dim + d] = val;
            }
        }
        
        result.retrieved = retrieve_proj->forward(retrieved_mem);
        return result;
    }
    
    // 记忆巩固 (LTP模拟)
    void consolidate(const Tensor& x) {
        Tensor encoded = encode(x);
        Tensor q = query_proj->forward(x);
        
        float* qp = q.as_fp32();
        float* mp = memory_bank.as_fp32();
        float* ep = encoded.as_fp32();
        
        size_t batch = x.shape_[0];
        
        // 计算注意力
        std::vector<std::vector<float>> attentions(batch);
        for (size_t b = 0; b < batch; ++b) {
            attentions[b].resize(memory_slots);
            for (size_t s = 0; s < memory_slots; ++s) {
                float dot = 0;
                for (size_t d = 0; d < memory_dim; ++d) {
                    dot += qp[b * memory_dim + d] * mp[s * memory_dim + d];
                }
                attentions[b][s] = dot;
            }
            
            float max_s = attentions[b][0];
            for (auto& sc : attentions[b]) max_s = std::max(max_s, sc);
            float sum = 0;
            for (auto& sc : attentions[b]) {
                sc = std::exp(sc - max_s);
                sum += sc;
            }
            for (auto& sc : attentions[b]) sc /= sum;
        }
        
        // 更新记忆槽 (加权平均)
        for (size_t s = 0; s < memory_slots; ++s) {
            float update = 0;
            float weight_sum = 0;
            for (size_t b = 0; b < batch; ++b) {
                float w = attentions[b][s];
                weight_sum += w;
                for (size_t d = 0; d < memory_dim; ++d) {
                    update += w * ep[b * memory_dim + d];
                }
            }
            if (weight_sum > 0) {
                for (size_t d = 0; d < memory_dim; ++d) {
                    mp[s * memory_dim + d] += ltp_rate * (update / weight_sum - mp[s * memory_dim + d]);
                }
            }
        }
    }
    
    // 前向
    RetrievalResult forward(const Tensor& x) {
        auto result = retrieve(x);
        return result;
    }
};

/**
 * PagedMemoryManager
 * 
 * 支持长记忆的分页系统：
 * 1. 内存中的活跃页
 * 2. 磁盘上的历史页
 * 3. 自动页换入换出
 */
class PagedMemoryManager {
public:
    struct MemoryPage {
        Tensor data;
        size_t page_id;
        size_t access_count;
        bool in_memory;
        std::string disk_path;
    };
    
    size_t page_size;        // 每页槽数量
    size_t max_memory_pages; // 内存最大页数
    size_t memory_dim;
    
    std::unordered_map<size_t, MemoryPage> pages;
    std::queue<size_t> page_order;  // 用于LRU
    
    size_t next_page_id;
    std::string disk_dir;
    
    PagedMemoryManager(size_t page_sz, size_t max_pages, size_t dim, const std::string& dir = "/tmp/neuroflow_mem")
        : page_size(page_sz), max_memory_pages(max_pages), memory_dim(dim), 
          next_page_id(0), disk_dir(dir) {
        // 创建磁盘目录
        // mkdir(disk_dir.c_str(), 0755);  // 实际应用中添加
    }
    
    // 创建新页
    size_t create_page() {
        size_t id = next_page_id++;
        MemoryPage page;
        page.page_id = id;
        page.data = Tensor({page_size, memory_dim}, QuantType::FP32);
        page.access_count = 0;
        page.in_memory = true;
        page.disk_path = disk_dir + "/page_" + std::to_string(id) + ".bin";
        
        pages[id] = page;
        page_order.push(id);
        
        // 如果超过内存限制，换出最旧页
        if (pages.size() > max_memory_pages) {
            evict_oldest();
        }
        
        return id;
    }
    
    // 获取页数据
    Tensor* get_page(size_t id) {
        if (pages.find(id) == pages.end()) return nullptr;
        
        auto& page = pages[id];
        page.access_count++;
        
        // 如果在磁盘，换入
        if (!page.in_memory) {
            load_from_disk(id);
        }
        
        return &page.data;
    }
    
    // 换出最旧页
    void evict_oldest() {
        while (page_order.size() > max_memory_pages) {
            size_t old_id = page_order.front();
            page_order.pop();
            
            auto& page = pages[old_id];
            if (page.in_memory) {
                save_to_disk(old_id);
                page.in_memory = false;
            }
        }
    }
    
    // 保存到磁盘
    void save_to_disk(size_t id) {
        auto& page = pages[id];
        if (page.data.dtype_ != QuantType::FP32)
            throw std::runtime_error("save_to_disk: page " + std::to_string(id) + " is not FP32");
        if (!page.data.data_ || page.data.data_size_ == 0)
            throw std::runtime_error("save_to_disk: page " + std::to_string(id) + " has no data");
        std::ofstream f(page.disk_path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot save page to disk: " + page.disk_path);
        const float* data = page.data.as_fp32();
        f.write(reinterpret_cast<const char*>(data), page.data.data_size_);
        if (!f.good()) throw std::runtime_error("Write error saving page: " + page.disk_path);
        f.close();
    }
    
    void load_from_disk(size_t id) {
        auto& page = pages[id];
        if (page.data.dtype_ != QuantType::FP32)
            throw std::runtime_error("load_from_disk: page " + std::to_string(id) + " is not FP32");
        if (!page.data.data_ || page.data.data_size_ == 0)
            throw std::runtime_error("load_from_disk: page " + std::to_string(id) + " has no data");
        std::ifstream f(page.disk_path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot load page from disk: " + page.disk_path);
        float* data = page.data.as_fp32();
        f.read(reinterpret_cast<char*>(data), page.data.data_size_);
        if (!f.good()) throw std::runtime_error("Read error loading page: " + page.disk_path);
        f.close();
        page.in_memory = true;
        page_order.push(id);
    }
    
    // 获取统计
    struct Stats {
        size_t total_pages;
        size_t in_memory_pages;
        size_t on_disk_pages;
        size_t total_memory_bytes;
    };
    
    Stats get_stats() {
        Stats s;
        s.total_pages = pages.size();
        s.in_memory_pages = 0;
        s.on_disk_pages = 0;
        s.total_memory_bytes = 0;
        
        for (auto& [id, page] : pages) {
            if (page.in_memory) {
                s.in_memory_pages++;
                s.total_memory_bytes += page.data.data_size_;
            } else {
                s.on_disk_pages++;
            }
        }
        return s;
    }
};

} // namespace neuroflow

#endif // NEUROFLOW_MEMORY_HPP