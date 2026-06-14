#ifndef NEUROFLOW_WEIGHT_IO_HPP
#define NEUROFLOW_WEIGHT_IO_HPP

#include "neuroflow/tensor.hpp"
#include "neuroflow/model.hpp"
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <unordered_map>
#include <cmath>

namespace neuroflow {

enum class InitStrategy : uint8_t {
    XAVIER_UNIFORM = 0,
    KAIMING_NORMAL = 1,
    ZEROS = 2,
    RANDOM_NORMAL = 3,
};

struct ValidationResult {
    bool all_passed = true;
    std::vector<std::string> failures;
};

class WeightInitializer {
public:
    static void xavier_uniform(Tensor& weight, size_t fan_in, size_t fan_out, std::mt19937& rng);
    static void kaiming_normal(Tensor& weight, size_t fan_in, std::mt19937& rng);
    static void zeros(Tensor& tensor);
    static void random_normal(Tensor& tensor, float mean, float std_dev, std::mt19937& rng);

    static void init_model_weights(NeuroFlowModel& model, InitStrategy strategy, uint32_t seed = 42);
    static ValidationResult validate_dimensions(const NeuroFlowModel& model);
};

void save_binary(const NeuroFlowModel& model, const std::string& path);
void load_binary(NeuroFlowModel& model, const std::string& path);
void save_npz(const NeuroFlowModel& model, const std::string& path);
void load_npz(NeuroFlowModel& model, const std::string& path);
void save_metadata(const NeuroFlowModel& model, const std::string& path);

}

#endif