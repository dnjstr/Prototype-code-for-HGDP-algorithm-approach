#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <cmath>
#include <numeric>
#include <queue>
#include <chrono>
#include <iomanip>
#include <random>
#include <functional>
#include <memory>

// Graph Neural Network components inspired by ScheduleNet
class GraphAttention {
public:
    virtual std::vector<double> computeNodeEmbeddings(const std::vector<std::vector<double>>& node_features,
                                                    const std::vector<std::vector<double>>& edge_features) = 0;
    virtual ~GraphAttention() = default;
};

class TypeAwareGraphAttention : public GraphAttention {
private:
    // Type-aware attention parameters
    std::unordered_map<std::string, std::vector<double>> type_embeddings;
    int embedding_size;
    
public:
    TypeAwareGraphAttention(int embedding_size) : embedding_size(embedding_size) {
        // Initialize type embeddings (in practice, these would be learned)
        type_embeddings["agent"] = std::vector<double>(embedding_size, 0.1);
        type_embeddings["task"] = std::vector<double>(embedding_size, 0.2);
        type_embeddings["active-agent"] = std::vector<double>(embedding_size, 0.3);
        type_embeddings["unassigned-task"] = std::vector<double>(embedding_size, 0.4);
    }

    std::vector<double> computeNodeEmbeddings(const std::vector<std::vector<double>>& node_features,
                                            const std::vector<std::vector<double>>& edge_features) override {
        // Simplified version of type-aware attention
        std::vector<double> embeddings(node_features.size() * embedding_size, 0.0);
        
        // In a real implementation, this would compute attention scores and aggregate features
        for (size_t i = 0; i < node_features.size(); i++) {
            // Get type from features (assuming last element is type indicator)
            std::string node_type = node_features[i].back() > 0.5 ? "agent" : "task";
            
            // Combine with type embedding
            for (int j = 0; j < embedding_size; j++) {
                embeddings[i * embedding_size + j] = node_features[i][j % node_features[i].size()] + 
                                                   type_embeddings[node_type][j];
            }
        }
        
        return embeddings;
    }
};

// Reinforcement Learning components
class RLPolicy {
private:
    std::shared_ptr<GraphAttention> graph_attention;
    std::vector<double> policy_weights;
    double learning_rate;
    
public:
    RLPolicy(std::shared_ptr<GraphAttention> ga, int feature_size, double lr = 0.01) 
        : graph_attention(ga), learning_rate(lr) {
        // Initialize policy weights (in practice, this would be a neural network)
        policy_weights.resize(feature_size, 0.1);
    }
    
    int selectAction(const std::vector<std::vector<double>>& node_features,
                   const std::vector<std::vector<double>>& edge_features) {
        // Compute node embeddings
        auto embeddings = graph_attention->computeNodeEmbeddings(node_features, edge_features);
        
        // Simple linear policy (in practice, would use a neural network)
        double max_score = -std::numeric_limits<double>::max();
        int best_action = 0;
        
        for (size_t i = 0; i < embeddings.size() / node_features.size(); i++) {
            double score = 0.0;
            for (size_t j = 0; j < policy_weights.size(); j++) {
                score += embeddings[i * node_features.size() + j] * policy_weights[j];
            }
            
            if (score > max_score) {
                max_score = score;
                best_action = i;
            }
        }
        
        return best_action;
    }
    
    void updatePolicy(double reward, const std::vector<std::vector<double>>& node_features,
                     const std::vector<std::vector<double>>& edge_features, int action) {
        // Simplified policy update (in practice, would use PPO or other RL algorithm)
        auto embeddings = graph_attention->computeNodeEmbeddings(node_features, edge_features);
        
        // Update weights based on reward
        for (size_t j = 0; j < policy_weights.size(); j++) {
            policy_weights[j] += learning_rate * reward * embeddings[action * node_features.size() + j];
        }
    }
};

// Modified Activity class to work with ScheduleNet approach
class Activity {
public:
    int id;
    double start_time;
    double finish_time;
    double weight;
    std::unordered_map<std::string, double> resources;
    std::string type; // Added type information for graph attention

    Activity(int id, double start, double finish, double weight, std::string type = "task")
        : id(id), start_time(start), finish_time(finish), weight(weight), type(type) {}
    
    // Add the missing method
    void addResource(const std::string& resource_name, double amount) {
        resources[resource_name] = amount;
    }

    std::vector<double> getFeatures() const {
        // Convert activity to feature vector for GNN
        return {static_cast<double>(id), start_time, finish_time, weight, 
                resources.at("cpu"), resources.at("memory"), type == "task" ? 0.0 : 1.0};
    }
};

// Modified Participant class
class Participant {
public:
    int id;
    std::unordered_map<std::string, double> resource_capacities;
    std::string type; // Added type information for graph attention
    
    Participant(int id) : id(id), type("agent") {}
    
    // Add the missing method
    void addResourceCapacity(const std::string& resource_name, double capacity) {
        resource_capacities[resource_name] = capacity;
    }
    
    std::vector<double> getFeatures() const {
        // Convert participant to feature vector for GNN
        return {static_cast<double>(id), resource_capacities.at("cpu"), 
                resource_capacities.at("memory"), 1.0}; // 1.0 indicates agent
    }
};

// Environment class that manages the scheduling process
class SchedulingEnvironment {
private:
    std::vector<Activity> activities;
    std::vector<Participant> participants;
    std::shared_ptr<RLPolicy> policy;
    double current_time;
    double makespan;
    
public:
    SchedulingEnvironment(const std::vector<Activity>& acts, 
                         const std::vector<Participant>& parts,
                         std::shared_ptr<RLPolicy> pol)
        : activities(acts), participants(parts), policy(pol), 
          current_time(0), makespan(0) {}
    
    // Build the agent-task graph
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    buildGraph() const {
        std::vector<std::vector<double>> node_features;
        std::vector<std::vector<double>> edge_features;
        
        // Add participants (agents) to the graph
        for (const auto& participant : participants) {
            node_features.push_back(participant.getFeatures());
        }
        
        // Add activities (tasks) to the graph
        for (const auto& activity : activities) {
            node_features.push_back(activity.getFeatures());
        }
        
        // Create edges (simplified - in practice would have more sophisticated edge features)
        for (size_t i = 0; i < participants.size(); i++) {
            for (size_t j = 0; j < activities.size(); j++) {
                // Edge feature could include distance, resource compatibility, etc.
                double resource_compatibility = 1.0;
                if (participants[i].resource_capacities.at("cpu") < activities[j].resources.at("cpu") ||
                    participants[i].resource_capacities.at("memory") < activities[j].resources.at("memory")) {
                    resource_compatibility = 0.0;
                }
                
                edge_features.push_back({
                    static_cast<double>(i),
                    static_cast<double>(j + participants.size()),
                    resource_compatibility
                });
            }
        }
        
        return {node_features, edge_features};
    }
    
    // Run one episode of scheduling
    double runEpisode() {
        current_time = 0;
        makespan = 0;
        
        while (!allTasksCompleted()) {
            // Get current state as graph
            auto [node_features, edge_features] = buildGraph();
            
            // Select action using policy
            int action = policy->selectAction(node_features, edge_features);
            
            // Execute action (simplified)
            executeAction(action);
            
            // Update current time
            current_time += 1.0; // Simplified time progression
        }
        
        // Calculate makespan (negative for reward)
        double reward = -makespan;
        
        // Update policy (in practice, would use a proper RL update with multiple episodes)
        auto [node_features, edge_features] = buildGraph();
        policy->updatePolicy(reward, node_features, edge_features, 0);
        
        return makespan;
    }
    
private:
    bool allTasksCompleted() const {
        // Simplified completion check
        return current_time > 100; // Just for demonstration
    }
    
    void executeAction(int action) {
        // Simplified action execution
        // In practice, this would assign tasks to agents and update the state
        makespan = std::max(makespan, current_time + 1.0);
    }
};

// Generate activities with type information
std::vector<Activity> generateActivities(int numActivities, int maxStartTime, int maxDuration, int maxWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> startDist(0, maxStartTime);
    std::uniform_int_distribution<> durationDist(1, maxDuration);
    std::lognormal_distribution<> weightDist(std::log(maxWeight/4), 0.8);

    for (int i = 1; i <= numActivities; ++i) {
        int start = startDist(gen);
        int duration = durationDist(gen);
        double raw_weight = weightDist(gen);
        int weight = std::max(1, std::min(maxWeight, static_cast<int>(raw_weight * maxWeight/10)));

        activities.emplace_back(i, start, start + duration, weight, "task");

        std::uniform_int_distribution<> resourceAmountDist(1, 5);
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }

    return activities;
}

// Generate participants with type information
std::vector<Participant> generateParticipants(int numParticipants, int maxCpu, int maxMemory) {
    std::vector<Participant> participants;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> cpuDist(5, maxCpu);
    std::uniform_int_distribution<> memoryDist(5, maxMemory);

    for (int i = 1; i <= numParticipants; ++i) {
        participants.emplace_back(100 + i);
        participants.back().addResourceCapacity("cpu", cpuDist(gen));
        participants.back().addResourceCapacity("memory", memoryDist(gen));
    }

    return participants;
}

int main() {
    // Generate dataset
    int numActivities = 100;
    int numParticipants = 5;
    int maxStartTime = 100;
    int maxDuration = 10;
    int maxWeight = 100;
    int maxCpu = 10;
    int maxMemory = 10;

    auto activities = generateActivities(numActivities, maxStartTime, maxDuration, maxWeight);
    auto participants = generateParticipants(numParticipants, maxCpu, maxMemory);

    // Create GNN and RL policy
    auto graph_attention = std::make_shared<TypeAwareGraphAttention>(32); // 32-dimensional embeddings
    auto policy = std::make_shared<RLPolicy>(graph_attention, 7); // 7 input features

    // Create environment
    SchedulingEnvironment env(activities, participants, policy);

    // Training loop (simplified)
    const int num_episodes = 10;
    std::vector<double> makespans;

    for (int episode = 0; episode < num_episodes; ++episode) {
        double makespan = env.runEpisode();
        makespans.push_back(makespan);
        std::cout << "Episode " << episode << ", Makespan: " << makespan << std::endl;
    }

    // Print results
    double avg_makespan = std::accumulate(makespans.begin(), makespans.end(), 0.0) / makespans.size();
    std::cout << "\nAverage makespan over " << num_episodes << " episodes: " << avg_makespan << std::endl;

    return 0;
}