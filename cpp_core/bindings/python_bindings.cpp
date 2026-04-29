/**
 * NeuroFlow Python Bindings
 * 
 * 使用pybind11绑定C++核心到Python
 * 保持与原Python API兼容
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "neuroflow/model.hpp"
#include "neuroflow/tensor.hpp"
#include "neuroflow/networks.hpp"
#include "neuroflow/memory.hpp"

namespace py = pybind11;
using namespace neuroflow;

// numpy数组转换为Tensor
Tensor numpy_to_tensor(py::array_t<float> arr) {
    py::buffer_info buf = arr.request();
    
    std::vector<size_t> shape;
    for (auto dim : buf.shape) shape.push_back(static_cast<size_t>(dim));
    
    Tensor t(shape, QuantType::FP32);
    float* data = t.as_fp32();  // 新创建的Tensor，可以用非const指针
    float* src = static_cast<float*>(buf.ptr);
    
    memcpy(data, src, t.data_size);
    
    return t;
}

// Tensor转换为numpy数组
py::array_t<float> tensor_to_numpy(const Tensor& t) {
    if (t.dtype != QuantType::FP32) {
        throw std::runtime_error("Only FP32 tensors can be converted to numpy");
    }
    
    std::vector<ssize_t> shape;
    for (auto dim : t.shape) shape.push_back(static_cast<ssize_t>(dim));
    
    const float* data = t.as_fp32();
    
    // 创建numpy数组并拷贝数据
    py::array_t<float> arr(shape);
    py::buffer_info buf = arr.request();
    memcpy(buf.ptr, data, t.data_size);
    
    return arr;
}

PYBIND11_MODULE(neuroflow_python, m) {
    m.doc() = "NeuroFlow C++ Core - Lightweight Brain-Inspired Neural Network";
    
    // 版本
    m.attr("__version__") = "1.0.0";
    
    // QuantType枚举
    py::enum_<QuantType>(m, "QuantType")
        .value("FP32", QuantType::FP32)
        .value("FP16", QuantType::FP16)
        .value("INT8", QuantType::INT8)
        .value("INT4", QuantType::INT4)
        .value("FP8_E4M3", QuantType::FP8_E4M3)
        .value("FP8_E5M2", QuantType::FP8_E5M2)
        .export_values();
    
    // MemoryLayout枚举
    py::enum_<MemoryLayout>(m, "MemoryLayout")
        .value("ROW_MAJOR", MemoryLayout::ROW_MAJOR)
        .value("COL_MAJOR", MemoryLayout::COL_MAJOR)
        .export_values();
    
    // Tensor类
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<std::vector<size_t>, QuantType>(), 
             py::arg("shape"), py::arg("dtype") = QuantType::FP32)
        .def_property_readonly("shape", [](const Tensor& t) { return t.shape; })
        .def_property_readonly("dtype", [](const Tensor& t) { return t.dtype; })
        .def_property_readonly("numel", &Tensor::numel)
        .def_property_readonly("data_size", [](const Tensor& t) { return t.data_size; })
        .def("clone", &Tensor::clone)
        .def("reshape", &Tensor::reshape)
        .def("as_numpy", &tensor_to_numpy)
        .def("from_numpy", [](Tensor& t, py::array_t<float> arr) {
            py::buffer_info buf = arr.request();
            float* data = t.as_fp32();
            memcpy(data, buf.ptr, t.data_size);
        })
        .def_static("from_numpy_array", &numpy_to_tensor);
    
    // TensorOps
    py::class_<TensorOps>(m, "TensorOps")
        .def_static("gemm", &TensorOps::gemm,
            py::arg("A"), py::arg("B"), py::arg("C"),
            py::arg("transA") = false, py::arg("transB") = false,
            py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f)
        .def_static("layer_norm", &TensorOps::layer_norm,
            py::arg("x"), py::arg("weight"), py::arg("bias"),
            py::arg("eps") = 1e-5f)
        .def_static("gelu", &TensorOps::gelu)
        .def_static("softmax", &TensorOps::softmax,
            py::arg("x"), py::arg("axis") = -1)
        .def_static("dropout", &TensorOps::dropout,
            py::arg("x"), py::arg("rate"), py::arg("training") = true)
        .def_static("add", &TensorOps::add)
        .def_static("mul", &TensorOps::mul)
        .def_static("concat", &TensorOps::concat,
            py::arg("tensors"), py::arg("axis") = 0)
        .def_static("quantize_int8", &TensorOps::quantize_int8)
        .def_static("dequantize_int8", &TensorOps::dequantize_int8);
    
    // Linear层
    py::class_<Linear>(m, "Linear")
        .def(py::init<size_t, size_t, bool, bool>(),
            py::arg("in_features"), py::arg("out_features"),
            py::arg("use_bias") = true, py::arg("quant") = false)
        .def("forward", &Linear::forward)
        .def("quantize", &Linear::quantize)
        .def_property_readonly("weight", [](Linear& l) { return l.weight; })
        .def_property_readonly("bias", [](Linear& l) { return l.bias; })
        .def_property_readonly("quantized", [](Linear& l) { return l.quantized; });
    
    // LayerNorm
    py::class_<LayerNorm>(m, "LayerNorm")
        .def(py::init<size_t, float>(), py::arg("dim"), py::arg("eps") = 1e-5f)
        .def("forward", &LayerNorm::forward)
        .def_property_readonly("weight", [](LayerNorm& l) { return l.weight; })
        .def_property_readonly("bias", [](LayerNorm& l) { return l.bias; });
    
    // ExecutiveControlNetwork
    py::class_<ExecutiveControlNetwork::Output>(m, "ECNOutput")
        .def_property_readonly("decision", [](ExecutiveControlNetwork::Output& o) { return o.decision; })
        .def_property_readonly("value", [](ExecutiveControlNetwork::Output& o) { return o.value; })
        .def_property_readonly("hidden_states", [](ExecutiveControlNetwork::Output& o) { return o.hidden_states; });
    
    py::class_<ExecutiveControlNetwork>(m, "ExecutiveControlNetwork")
        .def(py::init<size_t, size_t, size_t, size_t>(),
            py::arg("input_dim"), py::arg("hidden_dim"), py::arg("output_dim"),
            py::arg("num_layers") = 2)
        .def("forward", &ExecutiveControlNetwork::forward)
        .def("set_training", &ExecutiveControlNetwork::set_training)
        .def("quantize", &ExecutiveControlNetwork::quantize);
    
    // DefaultModeNetwork
    py::class_<DefaultModeNetwork::Output>(m, "DMNOutput")
        .def_property_readonly("vision", [](DefaultModeNetwork::Output& o) { return o.vision; })
        .def_property_readonly("associations", [](DefaultModeNetwork::Output& o) { return o.associations; })
        .def_property_readonly("latent", [](DefaultModeNetwork::Output& o) { return o.latent; });
    
    py::class_<DefaultModeNetwork>(m, "DefaultModeNetwork")
        .def(py::init<size_t, size_t, size_t>(),
            py::arg("memory_dim"), py::arg("latent_dim"), py::arg("num_associations") = 8)
        .def("forward", &DefaultModeNetwork::forward)
        .def("quantize", &DefaultModeNetwork::quantize);
    
    // SalienceNetwork
    py::class_<SalienceNetwork::Output>(m, "SNOutput")
        .def_property_readonly("saliency", [](SalienceNetwork::Output& o) { return o.saliency; })
        .def_property_readonly("gates", [](SalienceNetwork::Output& o) { return o.gates; })
        .def_property_readonly("anomaly", [](SalienceNetwork::Output& o) { return o.anomaly; });
    
    py::class_<SalienceNetwork>(m, "SalienceNetwork")
        .def(py::init<size_t, size_t>(), py::arg("input_dim"), py::arg("hidden_dim"))
        .def("forward", [](SalienceNetwork& sn, const Tensor& x) { return sn.forward(x); },
            py::arg("x"))
        .def("forward_with_baseline", [](SalienceNetwork& sn, const Tensor& x, const Tensor& baseline) {
            return sn.forward(x, &baseline);
        }, py::arg("x"), py::arg("baseline"))
        .def("quantize", &SalienceNetwork::quantize);
    
    // MemoryConsolidationModule
    py::class_<MemoryConsolidationModule::RetrievalResult>(m, "MemoryResult")
        .def_property_readonly("retrieved", [](MemoryConsolidationModule::RetrievalResult& r) { return r.retrieved; })
        .def_property_readonly("attention", [](MemoryConsolidationModule::RetrievalResult& r) { return r.attention; });
    
    py::class_<MemoryConsolidationModule>(m, "MemoryConsolidationModule")
        .def(py::init<size_t, size_t, size_t, float>(),
            py::arg("input_dim"), py::arg("memory_slots") = 64,
            py::arg("memory_dim") = 128, py::arg("ltp_rate") = 0.01f)
        .def("encode", &MemoryConsolidationModule::encode)
        .def("retrieve", &MemoryConsolidationModule::retrieve)
        .def("consolidate", &MemoryConsolidationModule::consolidate)
        .def("forward", &MemoryConsolidationModule::forward);
    
    // LatentKVCache (MLA)
    py::class_<LatentKVCache>(m, "LatentKVCache")
        .def(py::init<size_t, size_t, size_t, size_t>(),
            py::arg("model_dim"), py::arg("heads"), py::arg("latent_dim"),
            py::arg("max_len") = 4096)
        .def("forward", &LatentKVCache::forward,
            py::arg("x"), py::arg("use_cache") = true)
        .def("clear_cache", &LatentKVCache::clear_cache)
        .def("cache_size_bytes", &LatentKVCache::cache_size_bytes)
        .def("memory_saving_ratio", &LatentKVCache::memory_saving_ratio);
    
    // NeuroFlowModel::Config
    py::class_<NeuroFlowModel::Config>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("input_dim", &NeuroFlowModel::Config::input_dim)
        .def_readwrite("hidden_dim", &NeuroFlowModel::Config::hidden_dim)
        .def_readwrite("output_dim", &NeuroFlowModel::Config::output_dim)
        .def_readwrite("memory_dim", &NeuroFlowModel::Config::memory_dim)
        .def_readwrite("memory_slots", &NeuroFlowModel::Config::memory_slots)
        .def_readwrite("num_layers", &NeuroFlowModel::Config::num_layers)
        .def_readwrite("num_associations", &NeuroFlowModel::Config::num_associations)
        .def_readwrite("use_quantization", &NeuroFlowModel::Config::use_quantization)
        .def_readwrite("use_mla", &NeuroFlowModel::Config::use_mla)
        .def_readwrite("mla_latent_dim", &NeuroFlowModel::Config::mla_latent_dim);
    
    // NeuroFlowModel::Output
    py::class_<NeuroFlowModel::Output>(m, "ModelOutput")
        .def_property_readonly("output", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.output); })
        .def_property_readonly("decision", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.decision); })
        .def_property_readonly("value", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.value); })
        .def_property_readonly("saliency", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.saliency); })
        .def_property_readonly("ecn_gate", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.ecn_gate); })
        .def_property_readonly("dmn_gate", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.dmn_gate); })
        .def_property_readonly("anomaly", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.anomaly); })
        .def_property_readonly("mem_attention", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.mem_attention); })
        .def_property_readonly("retrieved_mem", [](NeuroFlowModel::Output& o) { return tensor_to_numpy(o.retrieved_mem); })
        .def_property_readonly("manifold", [](NeuroFlowModel::Output& o) { 
            if (o.manifold.data) return tensor_to_numpy(o.manifold);
            return py::array_t<float>();
        });
    
    // NeuroFlowModel::Stats
    py::class_<NeuroFlowModel::Stats>(m, "ModelStats")
        .def_readonly("total_params", &NeuroFlowModel::Stats::total_params)
        .def_readonly("memory_bytes", &NeuroFlowModel::Stats::memory_bytes)
        .def_readonly("quantization_ratio", &NeuroFlowModel::Stats::quantization_ratio);
    
    // NeuroFlowModel主类
    py::class_<NeuroFlowModel>(m, "NeuroFlowModel")
        .def(py::init<>())
        .def(py::init<NeuroFlowModel::Config>(), py::arg("config"))
        .def("forward", [](NeuroFlowModel& m, py::array_t<float> x, 
                          py::object memory_input, bool consolidate, bool return_manifold) {
            Tensor input = numpy_to_tensor(x);
            const Tensor* mem_ptr = nullptr;
            Tensor mem;
            
            if (!memory_input.is_none()) {
                mem = numpy_to_tensor(memory_input.cast<py::array_t<float>>());
                mem_ptr = &mem;
            }
            
            return m.forward(input, mem_ptr, consolidate, return_manifold);
        }, py::arg("x"), py::arg("memory_input") = py::none(),
           py::arg("consolidate") = false, py::arg("return_manifold") = false)
        .def("get_manifold_trajectory", [](NeuroFlowModel& m, py::array_t<float> x, size_t steps) {
            Tensor input = numpy_to_tensor(x);
            auto trajectory = m.get_manifold_trajectory(input, steps);
            
            py::list result;
            for (auto& t : trajectory) {
                result.append(tensor_to_numpy(t));
            }
            return result;
        }, py::arg("x"), py::arg("steps") = 10)
        .def("set_training", &NeuroFlowModel::set_training)
        .def("quantize", &NeuroFlowModel::quantize)
        .def("get_stats", &NeuroFlowModel::get_stats)
        .def("save", &NeuroFlowModel::save)
        .def("load", &NeuroFlowModel::load)
        .def_property_readonly("config", [](NeuroFlowModel& m) { return m.config; });
    
    // NeuroFlowLite
    py::class_<NeuroFlowLite, NeuroFlowModel>(m, "NeuroFlowLite")
        .def(py::init<size_t>(), py::arg("input_dim") = 512);
    
    // 便捷函数
    m.def("create_tensor", [](py::array_t<float> arr) {
        return numpy_to_tensor(arr);
    }, "Create Tensor from numpy array");
    
    m.def("benchmark", []() {
        NeuroFlowModel::Config cfg;
        cfg.input_dim = 512;
        cfg.hidden_dim = 256;
        cfg.output_dim = 10;
        
        NeuroFlowModel original(cfg);
        
        NeuroFlowModel::Config lite_cfg;
        lite_cfg.input_dim = 512;
        lite_cfg.hidden_dim = 128;
        lite_cfg.output_dim = 10;
        lite_cfg.memory_dim = 64;
        lite_cfg.memory_slots = 32;
        lite_cfg.num_layers = 1;
        lite_cfg.use_quantization = true;
        
        NeuroFlowModel lite(lite_cfg);
        
        auto orig_stats = original.get_stats();
        auto lite_stats = lite.get_stats();
        
        py::dict result;
        result["original_params"] = orig_stats.total_params;
        result["original_memory_mb"] = orig_stats.memory_bytes / 1024.0 / 1024.0;
        result["lite_params"] = lite_stats.total_params;
        result["lite_memory_mb"] = lite_stats.memory_bytes / 1024.0 / 1024.0;
        result["size_reduction"] = 1.0 - static_cast<float>(lite_stats.total_params) / orig_stats.total_params;
        
        return result;
    }, "Benchmark comparison between original and lite models");
}