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








// Class representing an activity
class Activity {
public:
    int id;
    double start_time;
    double finish_time;
    double weight;
    std::unordered_map<std::string, double> resources;








    Activity(int id, double start, double finish, double weight)
        : id(id), start_time(start), finish_time(finish), weight(weight) {}








    // Add resource requirement
    void addResource(const std::string& resource_name, double amount) {
        resources[resource_name] = amount;
    }








    // Value density (weight per unit time)
    double valueDensity() const {
        return weight / (finish_time - start_time);
    }








    // Print activity details
    void print() const {
        std::cout << "Activity " << id << ": [" << start_time << ", " << finish_time
                  << "), weight=" << weight << std::endl;
    }
};








// Class representing a participant
class Participant {
public:
    int id;
    std::unordered_map<std::string, double> resource_capacities;
   
    Participant(int id) : id(id) {}
   
    // Add resource capacity
    void addResourceCapacity(const std::string& resource_name, double capacity) {
        resource_capacities[resource_name] = capacity;
    }
   
    // Print participant details
    void print() const {
        std::cout << "Participant " << id << std::endl;
        for (const auto& [resource, capacity] : resource_capacities) {
            std::cout << "  " << resource << ": " << capacity << std::endl;
        }
    }
};








// Class representing a time slot
class TimeSlot {
public:
    double start_time;
    double end_time;
   
    TimeSlot(double start, double end) : start_time(start), end_time(end) {}
   
    // Check if a time slot contains another time interval
    bool contains(double start, double finish) const {
        return start_time <= start && end_time >= finish;
    }
   
    // Duration of the time slot
    double duration() const {
        return end_time - start_time;
    }
};








// Class implementing the pure dynamic programming algorithm
class HybridAlgorithm {
private:
    std::vector<Activity> activities;
    std::vector<Participant> participants;
    std::unordered_map<int, int> allocation; // Activity ID -> Participant ID
   
public:
    HybridAlgorithm(const std::vector<Activity>& acts, const std::vector<Participant>& parts)
        : activities(acts), participants(parts) {
        // Initialize allocation with all activities unassigned (-1)
        for (const auto& activity : activities) {
            allocation[activity.id] = -1;
        }
    }
   
    // Run the pure dynamic programming algorithm
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();


        std::cout << "Running Pure Dynamic Programming Algorithm" << std::endl;
        auto phase1_start = std::chrono::high_resolution_clock::now();
        pureDynamicProgrammingAllocation();
        auto phase1_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        std::cout << "Runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start).count()
                  << " ms" << std::endl;


        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\nTotal execution time: " << total_duration.count() << " ms" << std::endl;
    }
   
    // Pure Dynamic Programming Allocation
    void pureDynamicProgrammingAllocation() {
        // Sort activities by finish time
        std::vector<int> sorted_activities;
        for (size_t i = 0; i < activities.size(); i++) {
            sorted_activities.push_back(i);
        }
        std::sort(sorted_activities.begin(), sorted_activities.end(),
                  [this](int a, int b) {
                      return activities[a].finish_time < activities[b].finish_time;
                  });


        // Precompute compatibility matrix for activities
        std::vector<std::vector<bool>> activity_compatibility(activities.size(),
                                                              std::vector<bool>(activities.size(), false));
        for (size_t i = 0; i < activities.size(); i++) {
            for (size_t j = 0; j < activities.size(); j++) {
                if (i != j) {
                    activity_compatibility[i][j] = !activitiesOverlap(activities[i], activities[j]);
                }
            }
        }


        // Precompute participant compatibility
        std::vector<std::vector<bool>> participant_compatibility(activities.size(),
                                                                 std::vector<bool>(participants.size(), false));
        for (size_t i = 0; i < activities.size(); i++) {
            for (size_t j = 0; j < participants.size(); j++) {
                participant_compatibility[i][j] = hasResourceCapacity(j, activities[i]);
            }
        }


        // Track assigned activities for each participant
        std::vector<std::vector<int>> participant_assignments(participants.size());
        std::vector<double> participant_values(participants.size(), 0);


        // Sort activities by value density
        std::sort(sorted_activities.begin(), sorted_activities.end(),
                 [this](int a, int b) {
                     return activities[a].valueDensity() > activities[b].valueDensity();
                 });


        // Try to assign activities to participants
        for (int act_idx : sorted_activities) {
            const Activity& activity = activities[act_idx];
            bool assigned = false;


            // Find the participant with the lowest current value
            int best_p = -1;
            double min_value = std::numeric_limits<double>::max();
           
            for (size_t p = 0; p < participants.size(); p++) {
                if (participant_compatibility[act_idx][p] && participant_values[p] < min_value) {
                    bool can_assign = true;
                    // Check conflicts with already assigned activities
                    for (int assigned_idx : participant_assignments[p]) {
                        if (!activity_compatibility[act_idx][assigned_idx]) {
                            can_assign = false;
                            break;
                        }
                    }
                   
                    if (can_assign) {
                        min_value = participant_values[p];
                        best_p = p;
                    }
                }
            }


            // Assign activity to the best participant
            if (best_p != -1) {
                participant_assignments[best_p].push_back(act_idx);
                participant_values[best_p] += activity.weight;
                allocation[activity.id] = participants[best_p].id;
                assigned = true;
            }
        }


        // If we haven't assigned enough activities, try to balance the assignments
        if (std::accumulate(participant_values.begin(), participant_values.end(), 0.0) <
            std::accumulate(activities.begin(), activities.end(), 0.0,
                          [](double sum, const Activity& a) { return sum + a.weight; }) * 0.5) {
           
            // Clear previous assignments
            allocation.clear();
            participant_assignments.clear();
            participant_assignments.resize(participants.size());
            participant_values.assign(participants.size(), 0);


            // Try a different approach: assign activities in rounds
            for (size_t round = 0; round < 3; round++) {  // Try up to 3 rounds
                for (size_t p = 0; p < participants.size(); p++) {
                    for (int act_idx : sorted_activities) {
                        if (allocation.count(activities[act_idx].id) > 0) continue;


                        const Activity& activity = activities[act_idx];
                        if (participant_compatibility[act_idx][p]) {
                            bool can_assign = true;
                            for (int assigned_idx : participant_assignments[p]) {
                                if (!activity_compatibility[act_idx][assigned_idx]) {
                                    can_assign = false;
                                    break;
                                }
                            }


                            if (can_assign) {
                                participant_assignments[p].push_back(act_idx);
                                participant_values[p] += activity.weight;
                                allocation[activity.id] = participants[p].id;
                                break;  // Move to next participant
                            }
                        }
                    }
                }
            }
        }
    }
   
    // Check if participant has sufficient resources for an activity
    bool hasResourceCapacity(int participant_idx, const Activity& activity) const {
        const Participant& participant = participants[participant_idx];


        for (const auto& [resource, required] : activity.resources) {
            auto it = participant.resource_capacities.find(resource);


            // If participant doesn't have this resource or has insufficient capacity
            if (it == participant.resource_capacities.end() || it->second < required) {
                return false;
            }
        }


        return true;
    }


    // Check if two activities overlap
    bool activitiesOverlap(const Activity& a, const Activity& b) const {
        return a.start_time < b.finish_time && a.finish_time > b.start_time;
    }
   
    // Print allocation summary
    void printAllocationSummary() const {
        // Count assigned activities and total weight
        int assigned_count = 0;
        double total_weight = 0;
       
        std::vector<double> participant_values(participants.size(), 0);
       
        for (size_t i = 0; i < activities.size(); i++) {
            // Check if the key exists in the map before accessing
            if (allocation.count(activities[i].id) > 0) {
                int participant_id = allocation.at(activities[i].id);
                if (participant_id != -1) {
                    assigned_count++;
                    total_weight += activities[i].weight;
                   
                    int p_idx = getParticipantIndex(participant_id);
                    if (p_idx != -1) { // Add check to ensure valid p_idx
                        participant_values[p_idx] += activities[i].weight;
                    }
                }
            }
        }
       
        // Calculate fairness metrics
        double fairness = calculateFairness(participant_values);
        double jains_index = calculateJainsIndex(participant_values);
       
        std::cout << "Assigned activities: " << assigned_count << "/" << activities.size() << std::endl;
        std::cout << "Total weight: " << total_weight << std::endl;
        std::cout << "Fairness: " << fairness << std::endl;
        std::cout << "Jain's index: " << jains_index << std::endl;
       
        // Print distribution for each participant
        std::cout << "Value distribution:" << std::endl;
        for (size_t i = 0; i < participants.size(); i++) {
            std::cout << "  Participant " << participants[i].id << ": " << participant_values[i] << std::endl;
        }
    }
   
    // Get participant index from ID
    int getParticipantIndex(int participant_id) const {
        for (size_t i = 0; i < participants.size(); i++) {
            if (participants[i].id == participant_id) {
                return i;
            }
        }
        return -1;
    }
   
    // Calculate normalized fairness metric
    double calculateFairness(const std::vector<double>& values) const {
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        if (sum == 0) return 1.0;  // If all values are 0, we consider it perfectly fair
       
        double avg = sum / values.size();
       
        double sum_squared_diff = 0;
        for (double val : values) {
            sum_squared_diff += (val - avg) * (val - avg);
        }
       
        return 1.0 - std::sqrt(sum_squared_diff / (values.size() * avg * avg));
    }
   
    // Calculate Jain's fairness index
    double calculateJainsIndex(const std::vector<double>& values) const {
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        double sum_squared = 0;
        for (double val : values) {
            sum_squared += val * val;
        }
       
        if (sum_squared == 0) return 1.0;  // If all values are 0, we consider it perfectly fair
        return (sum * sum) / (values.size() * sum_squared);
    }
   
    // Get final allocation
    std::unordered_map<int, int> getAllocation() const {
        return allocation;
    }
};








// Generate SD-1 dataset
std::vector<Activity> generateActivities(int numActivities, int maxStartTime, int maxDuration, int maxWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> startDist(0, maxStartTime);
    std::uniform_int_distribution<> durationDist(1, maxDuration);
    std::uniform_int_distribution<> weightDist(1, maxWeight);




    for (int i = 1; i <= numActivities; ++i) {
        int start = startDist(gen);
        int duration = durationDist(gen);
        int weight = weightDist(gen);
        activities.emplace_back(i, start, start + duration, weight);




        // Add random resource requirements
        std::uniform_int_distribution<> resourceAmountDist(1, 3);
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }




    return activities;
}


std::vector<Activity> generateNormalActivities(int numActivities, int maxStartTime, int maxDuration, int meanWeight, int stdDevWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> startDist(0, maxStartTime);
    std::uniform_int_distribution<> durationDist(1, maxDuration);
    std::normal_distribution<> weightDist(meanWeight, stdDevWeight); // Normal distribution for weights


    for (int i = 1; i <= numActivities; ++i) {
        int start = startDist(gen);
        int duration = durationDist(gen);
        int weight = std::max(1, static_cast<int>(weightDist(gen))); // Ensure weight is at least 1


        activities.emplace_back(i, start, start + duration, weight);


        // Add random resource requirements
        std::uniform_int_distribution<> resourceAmountDist(1, 5); // Moderate resource requirements
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }


    return activities;
}


std::vector<Activity> generateClusteredActivities(int numActivities, int numClusters, int clusterWidth, int maxWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> clusterStartDist(0, 100); // Random cluster start times
    std::uniform_int_distribution<> durationDist(1, clusterWidth); // Activity durations within cluster
    std::uniform_int_distribution<> clusterAssignmentDist(0, numClusters - 1); // Assign activities to clusters


    // Log-normal distribution for weights (creates highly skewed distribution)
    double mu = std::log(maxWeight / 4);  // Location parameter
    double sigma = 0.8;                  // Shape parameter (higher = more skewed)
    std::lognormal_distribution<> weightDist(mu, sigma);


    // Generate cluster start times
    std::vector<int> clusterStartTimes(numClusters);
    for (int i = 0; i < numClusters; ++i) {
        clusterStartTimes[i] = clusterStartDist(gen);
    }


    for (int i = 1; i <= numActivities; ++i) {
        // Assign activity to a random cluster
        int clusterIdx = clusterAssignmentDist(gen);
        int clusterStart = clusterStartTimes[clusterIdx];


        // Generate activity start time and duration within the cluster
        int start = clusterStart + durationDist(gen);
        int duration = durationDist(gen);


        // Generate weight using log-normal distribution
        double raw_weight = weightDist(gen);
        int weight = std::max(1, std::min(maxWeight, static_cast<int>(raw_weight * maxWeight / 10)));


        activities.emplace_back(i, start, start + duration, weight);


        // Add random resource requirements
        std::uniform_int_distribution<> resourceAmountDist(1, 5); // Moderate resource requirements
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }


    return activities;
}


std::vector<Participant> generateParticipants(int numParticipants, int maxCpu, int maxMemory) {
    std::vector<Participant> participants;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> cpuDist(5, maxCpu);    // Moderate CPU capacity
    std::uniform_int_distribution<> memoryDist(5, maxMemory); // Moderate memory capacity


    for (int i = 1; i <= numParticipants; ++i) {
        participants.emplace_back(100 + i);
        participants.back().addResourceCapacity("cpu", cpuDist(gen));
        participants.back().addResourceCapacity("memory", memoryDist(gen));
    }


    return participants;
}




// Calculate fairness penalty for a participant
double fairnessPenalty(const std::vector<double>& participant_values, int participant_idx, double new_weight) {
    double total_weight = std::accumulate(participant_values.begin(), participant_values.end(), 0.0);
    double current_weight = participant_values[participant_idx] + new_weight;


    // Calculate Jain's fairness index
    double sum_squared = 0;
    for (double value : participant_values) {
        sum_squared += value * value;
    }
    sum_squared += current_weight * current_weight;


    double sum = total_weight + current_weight;
    return (sum * sum) / (participant_values.size() * sum_squared);
}




int main() {
    // Generate SD-4 dataset
    int numActivities = 300;  // Edge case dataset with 300 activities
    int numParticipants = 8;  // 8 participants
    int numClusters = 5;      // 5 temporal clusters
    int clusterWidth = 10;    // Activities within a cluster span 10 time units
    int maxWeight = 100;      // Maximum weight for activities
    int maxCpu = 10;          // Maximum CPU capacity
    int maxMemory = 10;       // Maximum memory capacity


    std::vector<Activity> activities = generateClusteredActivities(numActivities, numClusters, clusterWidth, maxWeight);
    std::vector<Participant> participants = generateParticipants(numParticipants, maxCpu, maxMemory);


    // Print problem instance
    std::cout << "Problem Instance (SD-4):" << std::endl;
    std::cout << "Activities: " << activities.size() << std::endl;
    std::cout << "Participants: " << participants.size() << std::endl;


    // Print summary statistics
    double total_weight = 0;
    double min_weight = std::numeric_limits<double>::max();
    double max_weight = 0;
    std::vector<double> weights;
    weights.reserve(activities.size());


    for (const auto& activity : activities) {
        total_weight += activity.weight;
        min_weight = std::min(min_weight, static_cast<double>(activity.weight));
        max_weight = std::max(max_weight, static_cast<double>(activity.weight));
        weights.push_back(activity.weight);
    }


    // Sort weights for percentile calculation
    std::sort(weights.begin(), weights.end());
    double median = weights[weights.size() / 2];
    double p90 = weights[static_cast<size_t>(weights.size() * 0.9)];
    double p95 = weights[static_cast<size_t>(weights.size() * 0.95)];


    double avg_weight = total_weight / activities.size();


    std::cout << "\nActivity Weight Statistics:" << std::endl;
    std::cout << "  Minimum weight: " << min_weight << std::endl;
    std::cout << "  Maximum weight: " << max_weight << std::endl;
    std::cout << "  Average weight: " << avg_weight << std::endl;
    std::cout << "  Median weight: " << median << std::endl;
    std::cout << "  90th percentile: " << p90 << std::endl;
    std::cout << "  95th percentile: " << p95 << std::endl;
    std::cout << "  Total weight: " << total_weight << std::endl;


    // Create and run the hybrid algorithm
    HybridAlgorithm algorithm(activities, participants);
    algorithm.run();


    return 0;
}





