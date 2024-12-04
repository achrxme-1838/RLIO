#include <torch/script.h>
#include <iostream>
#include <memory>
#include <string>
#include <filesystem>

std::vector<torch::Tensor> flatten_q_value_to_action(
    const torch::Tensor& flatten_q_value,
    const std::vector<torch::Tensor>& action_discrete_ranges,
    int action_number) {
    
    std::vector<torch::Tensor> selected_actions;
    // std::vector<float> selected_actions_float;

    auto device = flatten_q_value.device(); // Get device of input tensor

    for (int i = 0; i < action_number; ++i) {
        torch::Tensor action_range = action_discrete_ranges[i].to(device);

        int start_idx = i * action_range.size(0);
        int end_idx = (i + 1) * action_range.size(0);
        torch::Tensor soft_max_q_value = torch::softmax(flatten_q_value.slice(1, start_idx, end_idx), 1);

        torch::Tensor max_idx = std::get<1>(soft_max_q_value.max(1, true));
        torch::Tensor action_new_onehot = torch::zeros({flatten_q_value.size(0), action_range.size(0)}, device);
        action_new_onehot.scatter_(1, max_idx, 1.0);

        torch::Tensor selected_action_i = action_range.index_select(0, max_idx.squeeze());
        // float selected_action_i_float = selected_action_i.item<float>();

        selected_actions.push_back(selected_action_i);
    }

    return selected_actions;
}

std::vector<torch::Tensor> get_policy_output(   std::vector<torch::jit::IValue> inputs,
                                                const std::vector<torch::Tensor>& action_discrete_ranges,
                                                int action_number,
                                                torch::jit::script::Module model){

    at::Tensor flatten_q_value;

    try {
        auto output = model.forward(inputs).toTuple(); // Get the output as a tuple
        flatten_q_value = output->elements()[0].toTensor();

    } catch (const c10::Error& e) {
        std::cerr << "Error during model forward pass: " << e.what() << "\n";
    }

    std::vector<torch::Tensor> selected_actions = flatten_q_value_to_action(flatten_q_value, action_discrete_ranges, action_number);

    return selected_actions;
}

int main() {
    torch::jit::script::Module model;

    std::string base_dir = "/home/lim/rlio_ws/exported_models/20241129_103031/4800/";
    std::string model_file = "Q_net.pt";
    std::string model_path = base_dir + model_file;

    if (std::filesystem::exists(base_dir)) {
        try {

            model = torch::jit::load(model_path);
            model.eval();
        }    catch (const c10::Error& e) {
                std::cerr << "Error loading the model\n";
        }
    }
    else{
        std::cerr << "Error: Model file does not exist at " << model_path << "\n";
    }

    std::cout << "Model loaded successfully!\n";


    auto device = torch::kCPU; // Use CUDA or torch::kCPU depending on your environment

    // Define action_discrete_ranges
    std::vector<torch::Tensor> action_discrete_ranges = {
        torch::tensor({0.2, 0.4, 0.5, 0.6, 0.8, 1.0}, device),
        torch::tensor({2, 3, 4, 5}, device),
        torch::tensor({0.2, 0.4, 0.6, 0.8, 1.0}, device),
        torch::tensor({0.005, 0.001, 0.05, 0.01, 0.1}, device)
    };

    int action_number = 4;


    // try {
    //     auto output = model.forward(inputs).toTuple(); // Get the output as a tuple
    //     flatten_q_value = output->elements()[0].toTensor();

    // } catch (const c10::Error& e) {
    //     std::cerr << "Error during model forward pass: " << e.what() << "\n";
    // }
    

    // std::vector<torch::Tensor> selected_actions = flatten_q_value_to_action(flatten_q_value, action_discrete_ranges, action_number);

    // for (int i = 0; i < selected_actions.size(); ++i) {
    //     // std::cout << "action idx " << i << ": " << selected_actions[i] << std::endl;
    //     std::cout << "action idx " << i << ": " << selected_actions[i].item<torch::Tensor>() << std::endl;
    // }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 1024})); // Example input shape

    at::Tensor flatten_q_value;


    std::vector<torch::Tensor> selected_actions = get_policy_output(inputs, action_discrete_ranges, action_number, model);
    
    for (int i = 0; i < selected_actions.size(); ++i) {
        // std::cout << "action idx " << i << ": " << selected_actions[i] << std::endl;
        std::cout << "action idx " << i << ": " << selected_actions[i].item<float>() << std::endl;
    }


    return 0;
}
