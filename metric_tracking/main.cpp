#include <time.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>

#include "restclient-cpp/restclient.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

class MLFlowLogger {

    public:
        MLFlowLogger(std::string target, std::string experiment_name) {
            m_target = target + "/api/2.0/mlflow/";

            m_experiment_id = find_or_create_experiment(experiment_name);

            m_run_id = create_run(m_experiment_id);
        };

        void log_metric(std::string metric_name, float value, int step) {

            auto now = std::chrono::system_clock::now().time_since_epoch();
            std::chrono::milliseconds casted_now = std::chrono::duration_cast<std::chrono::milliseconds>(now);

            std::ostringstream content;
            content << "{\"run_id\": \"" + m_run_id + "\", ";
            content << "\"key\": \"" + metric_name + "\", ";
            content << "\"timestamp\": ";
            content << casted_now.count();
            content << ", ";
            content << "\"step\": " + std::to_string(step) + ", ";
            content << "\"value\": " + std::to_string(value) + "}";

            auto create_request = mlflow_post_request("runs/log-metric", content.str());
        }

    private:
        std::string m_target;
        std::string m_experiment_id;
        std::string m_run_id;

        std::string find_or_create_experiment(std::string experiment_name) {
            auto find_request = mlflow_get_request("experiments/get-by-name", "?experiment_name=" + experiment_name);

            if(!find_request["error_code"].is_null()) {
                auto create_request = mlflow_post_request("experiments/create", "{\"name\": \"" + experiment_name + "\"}");
                std::cout << "Experiment Created: " << create_request["experiment_id"] << std::endl;
                return create_request["experiment_id"];
            }

            std::cout << "Experiment Found: " << find_request["experiment"]["experiment_id"] << std::endl;

            return find_request["experiment"]["experiment_id"];
        };

        std::string create_run(std::string experiment_name) {

            auto now = std::chrono::system_clock::now().time_since_epoch();
            std::chrono::milliseconds casted_now = std::chrono::duration_cast<std::chrono::milliseconds>(now);

            std::ostringstream content;
            content << "{\"experiment_id\": \"" + m_experiment_id + "\",";
            content << "\"start_time\": ";
            content << casted_now.count();
            content << "}";

            auto create_request = mlflow_post_request("runs/create", content.str());
            std::cout << "Run Created: " << create_request["run"]["info"]["run_id"] << std::endl;

            return create_request["run"]["info"]["run_id"];
        };


        json mlflow_get_request(std::string url, std::string parameters) {
            auto request = m_target + url;
            request += parameters;

            RestClient::Response r = RestClient::get(request);
            std::istringstream request_body(r.body);

            json data = json::parse(request_body);

            return data;
        }

        json mlflow_post_request(std::string url, std::string content) {
            auto request = m_target + url;

            RestClient::Response r = RestClient::post(
                request, 
                "application/json", 
                content
            );

            std::istringstream request_body(r.body);

            json data = json::parse(request_body);

            return data;
        }
        
};

int main() {

    MLFlowLogger logger{"mlflow:5050", "test2"};

    std::vector<float> loss = {300.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5};

    int step = 0;
    for(auto value : loss) {
        logger.log_metric("loss", value, step);
        step++;
    }

}