#include <torch/torch.h>
#include <chrono>

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

using namespace std::chrono;

int main() {

    std::cout << torch::mps::is_available() << std::endl;

    torch::Device device(torch::kMPS);

    // Create a new Net.
    auto net = std::make_shared<Net>();
    net->to(device);

    int batch_size = 64;
    int num_workers = 12;

    auto dataset = torch::data::datasets::MNIST(
        "./data", torch::data::datasets::MNIST::Mode::kTrain).map(
            torch::data::transforms::Stack<>());

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
    );

    // Instantiate an SGD optimization algorithm to update our Net's parameters.
    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    torch::Tensor loss;

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        time_point start = high_resolution_clock::now();
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            batch.data = batch.data.to(device);
            batch.target = batch.target.to(device);
            // Reset gradients.
            optimizer.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            loss = torch::nll_loss(prediction, batch.target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
        }

        time_point end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        std::cout << "Epoch: " << epoch << " | Loss: " << loss.item<float>() << " | Time: " << duration.count() << std::endl;

        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
    }
}