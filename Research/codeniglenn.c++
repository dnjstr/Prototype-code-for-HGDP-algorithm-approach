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
#include <string>
#include <fstream>
#include <sstream>


// Class representing an activity
class Activity {
public:
    int id;
    int job_id;      // Add this
    int job_op_idx;  // Add this (operation index in the job)
    double start_time;
    double finish_time;
    double weight;
    std::unordered_map<std::string, double> resources;


    Activity(int id, int job_id, int job_op_idx, double start, double finish, double weight)
        : id(id), job_id(job_id), job_op_idx(job_op_idx), start_time(start), finish_time(finish), weight(weight) {}


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


// Forward declaration
bool predecessorFinished(const Activity& activity, const std::unordered_map<int, int>& allocation, const std::vector<Activity>& activities);


// Class implementing the hybrid algorithm
class HybridAlgorithm {
private:
    std::vector<Activity> activities;
    std::vector<Participant> participants;
    std::unordered_map<int, int> allocation; // Activity ID -> Participant ID
    double fairness_epsilon; // Fairness tolerance parameter

    mutable double last_total_weight = 0;
    mutable double last_fairness = 0;
    mutable double last_jains_index = 0;
    mutable double last_makespan = 0;
    mutable double last_weighted_completion = 0;
   
public:
    HybridAlgorithm(const std::vector<Activity>& acts, const std::vector<Participant>& parts, double epsilon = 0.1)
        : activities(acts), participants(parts), fairness_epsilon(epsilon) {
        // Initialize allocation with all activities unassigned (-1)
        for (const auto& activity : activities) {
            allocation[activity.id] = -1;
        }
    }
   
    // Run the complete hybrid algorithm
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();


        std::cout << "Phase 1: Initial Greedy Allocation" << std::endl;
        auto phase1_start = std::chrono::high_resolution_clock::now();
        initialGreedyAllocation();
        auto phase1_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        std::cout << "Phase 1 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start).count()
                  << " ms" << std::endl;


        std::cout << "\nPhase 2: Dynamic Programming Refinement" << std::endl;
        auto phase2_start = std::chrono::high_resolution_clock::now();
        dynamicProgrammingRefinement();
        auto phase2_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        std::cout << "Phase 2 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end - phase2_start).count()
                  << " ms" << std::endl;


        std::cout << "\nPhase 3: Fairness Optimization" << std::endl;
        auto phase3_start = std::chrono::high_resolution_clock::now();
        fairnessOptimization();
        auto phase3_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        std::cout << "Phase 3 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase3_end - phase3_start).count()
                  << " ms" << std::endl;


        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\nTotal execution time: " << total_duration.count() << " ms" << std::endl;

        std::cout << "\n>>> Totals after all phases for this instance:" << std::endl;
        printAllocationSummary();
    }
   
    // Phase 1: Initial Greedy Allocation
    void initialGreedyAllocation() {
        // Sort activities by value-to-time ratio (descending)
        std::vector<int> sorted_indices(activities.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [this](int a, int b) {
                double ratio_a = activities[a].weight / (activities[a].finish_time - activities[a].start_time);
                double ratio_b = activities[b].weight / (activities[b].finish_time - activities[b].start_time);
                return ratio_a > ratio_b;
            });


        std::vector<double> available_time(participants.size(), 0);
        std::vector<double> accumulated_value(participants.size(), 0);


        for (int idx : sorted_indices) {
            const Activity& activity = activities[idx];


            int best_participant = -1;
            double min_value = std::numeric_limits<double>::max();


            for (size_t j = 0; j < participants.size(); ++j) {
                if (!hasResourceCapacity(j, activity))
                    continue;


                // Check for overlap with already assigned activities for this participant
                bool overlap = false;
                for (size_t k = 0; k < activities.size(); ++k) {
                    if (allocation.count(activities[k].id) && allocation.at(activities[k].id) == participants[j].id) {
                        if (!(activity.finish_time <= activities[k].start_time || activity.start_time >= activities[k].finish_time)) {
                            overlap = true;
                            break;
                        }
                    }
                }
                if (overlap) continue;


                // Prefer participant with lowest accumulated value so far
                if (accumulated_value[j] < min_value && available_time[j] <= activity.start_time) {
                    best_participant = j;
                    min_value = accumulated_value[j];
                }
            }


            if (best_participant != -1) {
                allocation[activity.id] = participants[best_participant].id;
                available_time[best_participant] = activity.finish_time;
                accumulated_value[best_participant] += activity.weight;
            }
        }
    }
   
    // Phase 2: Dynamic Programming Refinement
    void dynamicProgrammingRefinement() {
        double T = 0;
        for (const auto& a : activities) T = std::max(T, a.finish_time);


        for (size_t p_idx = 0; p_idx < participants.size(); p_idx++) {
            const Participant& participant = participants[p_idx];


            // 1. Collect activities currently assigned to this participant
            std::vector<int> assigned_activities;
            for (size_t i = 0; i < activities.size(); i++) {
                if (allocation[activities[i].id] == participant.id) {
                    assigned_activities.push_back(i);
                }
            }


            // 2. Construct timeline of available slots
            std::vector<TimeSlot> timeline = constructTimeline(assigned_activities);


            // 3. For each available slot, find unassigned activities that fit and optimize
            for (const TimeSlot& slot : timeline) {
                // Find all unassigned activities that could fit in this slot
                std::vector<int> candidates;
                for (size_t i = 0; i < activities.size(); i++) {
                    const Activity& act = activities[i];
                    if (allocation[act.id] == -1 &&
                        hasResourceCapacity(p_idx, act) &&
                        predecessorFinished(act, allocation, activities) &&
                        act.start_time >= slot.start_time &&
                        act.finish_time <= slot.end_time) {
                        candidates.push_back(i);
                    }
                }


                if (!candidates.empty()) {
                    // Run weighted interval scheduling on these candidates
                    std::vector<int> optimal_subset = weightedIntervalScheduling(candidates);
                    // Assign these activities to this participant
                    for (int idx : optimal_subset) {
                        allocation[activities[idx].id] = participant.id;
                    }
                }
            }
        }
    }
   
    // Phase 3: Fairness Optimization
    void fairnessOptimization() {
        bool improved = true;


        while (improved) {
            improved = false;


            // Calculate current value distribution
            std::vector<double> values(participants.size(), 0);
            for (size_t i = 0; i < activities.size(); i++) {
                int participant_id = allocation[activities[i].id];
                if (participant_id != -1) {
                    int p_idx = getParticipantIndex(participant_id);
                    if (p_idx != -1) {
                        values[p_idx] += activities[i].weight;
                    }
                }
            }


            // Calculate target fair value
            double total_value = std::accumulate(values.begin(), values.end(), 0.0);
            double target_value = total_value / participants.size();


            // Identify high-value and low-value participants
            std::vector<int> high_value_participants, low_value_participants;
            for (size_t i = 0; i < participants.size(); i++) {
                if (values[i] > target_value + fairness_epsilon) {
                    high_value_participants.push_back(i);
                } else if (values[i] < target_value - fairness_epsilon) {
                    low_value_participants.push_back(i);
                }
            }


            double best_improvement = 0;
            int best_high = -1, best_low = -1, best_activity = -1;


            // Try all possible transfers from high to low
            for (int high_idx : high_value_participants) {
                for (int low_idx : low_value_participants) {
                    for (size_t i = 0; i < activities.size(); i++) {
                        const Activity& activity = activities[i];
                        if (allocation[activity.id] == participants[high_idx].id &&
                            canTransferActivity(high_idx, low_idx, i)) {


                            double current_deviation = std::abs(values[high_idx] - target_value) +
                                                      std::abs(values[low_idx] - target_value);


                            double new_high_value = values[high_idx] - activity.weight;
                            double new_low_value = values[low_idx] + activity.weight;


                            double new_deviation = std::abs(new_high_value - target_value) +
                                                   std::abs(new_low_value - target_value);


                            double improvement = current_deviation - new_deviation;


                            if (improvement > best_improvement) {
                                best_improvement = improvement;
                                best_high = high_idx;
                                best_low = low_idx;
                                best_activity = i;
                            }
                        }
                    }
                }
            }


            // Perform the best transfer if found
            if (best_activity != -1) {
                const Activity& activity = activities[best_activity];
                allocation[activity.id] = participants[best_low].id;
                values[best_high] -= activity.weight;
                values[best_low] += activity.weight;
                improved = true;
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
   
    // Check if an activity can be transferred between participants
    bool canTransferActivity(int from_idx, int to_idx, int activity_idx) const {
        const Activity& activity = activities[activity_idx];


        // Check if target participant has resource capacity
        if (!hasResourceCapacity(to_idx, activity)) {
            return false;
        }


        // Check for overlap with already assigned activities for this participant
        for (size_t i = 0; i < activities.size(); i++) {
            if (i != activity_idx &&
                allocation.count(activities[i].id) > 0 &&
                allocation.at(activities[i].id) == participants[to_idx].id &&
                activitiesOverlap(activities[i], activity)) {
                return false;
            }
        }


        // Check job precedence for the activity
        // Predecessor must be assigned and finish before this activity's start
        if (activity.job_op_idx > 0) {
            for (const auto& act : activities) {
                if (act.job_id == activity.job_id && act.job_op_idx == activity.job_op_idx - 1) {
                    auto it = allocation.find(act.id);
                    if (it == allocation.end() || it->second == -1) return false;
                    if (act.finish_time > activity.start_time) return false;
                    break;
                }
            }
        }


        return true;
    }
   
    // Construct timeline of non-overlapping time slots
    std::vector<TimeSlot> constructTimeline(const std::vector<int>& assigned_activities) const {
        double horizon = 0;
        for (const auto& a : activities) {
            horizon = std::max(horizon, a.finish_time);
        }


        // Collect intervals to block out
        std::vector<std::pair<double, double>> busy;
        for (int act_idx : assigned_activities) {
            const Activity& activity = activities[act_idx];
            busy.emplace_back(activity.start_time, activity.finish_time);
        }
        std::sort(busy.begin(), busy.end());


        std::vector<TimeSlot> timeline;
        double prev_end = 0;
        for (const auto& [start, end] : busy) {
            if (start > prev_end) {
                timeline.emplace_back(prev_end, start);
            }
            prev_end = std::max(prev_end, end);
        }
        if (prev_end < horizon) {
            timeline.emplace_back(prev_end, horizon);
        }
        return timeline;
    }
   
    // Weighted interval scheduling algorithm using dynamic programming
    std::vector<int> weightedIntervalScheduling(const std::vector<int>& activity_indices) const {
        if (activity_indices.empty()) {
            return {};
        }
   
        // Sort activities by finish time
        std::vector<int> sorted = activity_indices;
        std::sort(sorted.begin(), sorted.end(),
            [this](int a, int b) {
                return activities[a].finish_time < activities[b].finish_time;
            });
   
        // Precompute p[i] - the last compatible activity before i
        std::vector<int> p(sorted.size(), -1);
        for (size_t i = 1; i < sorted.size(); i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (activities[sorted[j]].finish_time <= activities[sorted[i]].start_time) {
                    p[i] = j;
                    break;
                }
            }
        }
   
        // Dynamic programming table
        std::vector<double> dp(sorted.size() + 1, 0);
        for (size_t i = 1; i <= sorted.size(); i++) {
            double include = activities[sorted[i-1]].weight;
            if (p[i-1] != -1) {
                include += dp[p[i-1]+1];
            }
            dp[i] = std::max(include, dp[i-1]);
        }
   
        // Backtrack to find the optimal set
        std::vector<int> result;
        int i = sorted.size();
        while (i > 0) {
            if (p[i-1] == -1) {
                if (activities[sorted[i-1]].weight >= dp[i-1]) {
                    result.push_back(sorted[i-1]);
                    i = 0;
                } else {
                    i--;
                }
            } else {
                if (activities[sorted[i-1]].weight + dp[p[i-1]+1] >= dp[i-1]) {
                    result.push_back(sorted[i-1]);
                    i = p[i-1] + 1;
                } else {
                    i--;
                }
            }
        }
   
        return result;
    }
   
    // Print allocation summary
    void printAllocationSummary() const {
        int assigned_count = 0;
        double total_weight = 0;
        std::vector<double> participant_values(participants.size(), 0);

        for (size_t i = 0; i < activities.size(); i++) {
            if (allocation.count(activities[i].id) > 0) {
                int participant_id = allocation.at(activities[i].id);
                if (participant_id != -1) {
                    assigned_count++;
                    total_weight += activities[i].weight;
                    int p_idx = getParticipantIndex(participant_id);
                    if (p_idx != -1) {
                        participant_values[p_idx] += activities[i].weight;
                    }
                }
            }
        }

        double fairness = calculateFairness(participant_values);
        double jains_index = calculateJainsIndex(participant_values);
        double makespan = computeMakespan();
        double weighted_completion = computeWeightedCompletion();

        // Store for later retrieval
        last_total_weight = total_weight;
        last_fairness = fairness;
        last_jains_index = jains_index;
        last_makespan = makespan;
        last_weighted_completion = weighted_completion;

        std::cout << "Assigned activities: " << assigned_count << "/" << activities.size() << std::endl;
        std::cout << "Total weight: " << total_weight << std::endl;
        std::cout << "Fairness: " << fairness << std::endl;
        std::cout << "Jain's index: " << jains_index << std::endl;
        std::cout << "Makespan: " << makespan << std::endl;
        std::cout << "Weighted completion: " << weighted_completion << std::endl;
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
   
    // Check if two activities overlap
    bool activitiesOverlap(const Activity& a, const Activity& b) const {
        return a.start_time < b.finish_time && a.finish_time > b.start_time;
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
   
    // Set fairness epsilon parameter
    void setFairnessEpsilon(double epsilon) {
        fairness_epsilon = epsilon;
    }
   
    // Calculate adaptive fairness epsilon based on problem characteristics
    double calculateAdaptiveEpsilon() const {
        // Extract weights
        std::vector<double> weights;
        for (const auto& activity : activities) {
            weights.push_back(activity.weight);
        }
       
        // Calculate statistical properties
        double mean = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
       
        double variance = 0;
        for (double w : weights) {
            variance += (w - mean) * (w - mean);
        }
        variance /= weights.size();
       
        double std_dev = std::sqrt(variance);
        double cv = (mean > 0) ? std_dev / mean : 0;  // Coefficient of variation, prevent division by 0
       
        // Calculate range
        double min_weight = *std::min_element(weights.begin(), weights.end());
        double max_weight = *std::max_element(weights.begin(), weights.end());
        double range = max_weight - min_weight;
       
        // Calculate skewness (simplified)
        double skewness = 0;
        if (std_dev > 0) { // Prevent division by 0
            for (double w : weights) {
                skewness += std::pow((w - mean) / std_dev, 3);
            }
            skewness /= weights.size();
        }
       
        // Calculate base epsilon
        double base_epsilon = 0.05;
       
        // Adjust for distribution properties
        double epsilon_distribution = base_epsilon * (1 + cv) * (1 + std::abs(skewness) * 0.1);
        if (mean > 0) { // Prevent division by 0
            epsilon_distribution *= (range / mean * 0.1);
        }
       
        // Adjust for participant count
        double epsilon_participant = std::log10(participants.size() + 1) / std::log10(11);
       
        // Calculate final epsilon
        double epsilon = epsilon_distribution * (1 + epsilon_participant);
       
        // Bound epsilon
        return std::min(std::max(epsilon, 0.01), 0.25);
    }

    // Compute makespan (latest finish time among assigned activities)
    double computeMakespan() const {
        double makespan = 0;
        for (const auto& activity : activities) {
            if (allocation.count(activity.id) && allocation.at(activity.id) != -1) {
                makespan = std::max(makespan, activity.finish_time);
            }
        }
        return makespan;
    }

    // Compute total weighted completion time (sum of weight * finish_time for assigned activities)
    double computeWeightedCompletion() const {
        double total = 0;
        for (const auto& activity : activities) {
            if (allocation.count(activity.id) && allocation.at(activity.id) != -1) {
                total += activity.weight * activity.finish_time;
            }
        }
        return total;
    }

    // Check if resource usage fits within the participant's capacity
    bool resourceUsageFits(int participant_idx, const Activity& activity, const std::unordered_map<std::string, double>& current_usage) const {
        const Participant& participant = participants[participant_idx];








        for (const auto& [resource, required] : activity.resources) {
            auto it = current_usage.find(resource);
            double used = (it != current_usage.end()) ? it->second : 0.0;








            if (used + required > participant.resource_capacities.at(resource)) {
                return false;
            }
        }








        return true;
    }

    // Add getters for the metrics:
    double getLastTotalWeight() const { return last_total_weight; }
    double getLastFairness() const { return last_fairness; }
    double getLastJainsIndex() const { return last_jains_index; }
    double getLastMakespan() const { return last_makespan; }
    double getLastWeightedCompletion() const { return last_weighted_completion; }
};








bool predecessorFinished(const Activity& activity, const std::unordered_map<int, int>& allocation, const std::vector<Activity>& activities) {
    if (activity.job_op_idx == 0) return true; // First operation, no predecessor
    // Find predecessor activity for this job
    for (const auto& act : activities) {
        if (act.job_id == activity.job_id && act.job_op_idx == activity.job_op_idx - 1) {
            auto it = allocation.find(act.id);
            if (it == allocation.end() || it->second == -1) return false;
            if (act.finish_time > activity.start_time) return false;
            return true;
        }
    }
    return false;
}








// Generate SD-3 dataset
std::vector<Activity> generateActivities(int numActivities, int maxStartTime, int maxDuration, int maxWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> startDist(0, maxStartTime);
    std::uniform_int_distribution<> durationDist(1, maxDuration);
   
    // Log-normal distribution for weights (creates highly skewed distribution)
    // Parameters chosen to create a right-skewed distribution
    double mu = std::log(maxWeight/4);  // Location parameter
    double sigma = 0.8;                 // Shape parameter (higher = more skewed)
    std::lognormal_distribution<> weightDist(mu, sigma);




    for (int i = 1; i <= numActivities; ++i) {
        int start = startDist(gen);
        int duration = durationDist(gen);
        // Generate weight using log-normal distribution and scale to desired range
        double raw_weight = weightDist(gen);
        int weight = std::max(1, std::min(maxWeight, static_cast<int>(raw_weight * maxWeight/10)));




        activities.emplace_back(i, 0, 0, start, start + duration, weight);




        // Add random resource requirements with higher variance
        std::uniform_int_distribution<> resourceAmountDist(1, 5); // Increased resource requirements
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




        activities.emplace_back(i, 0, 0, start, start + duration, weight);




        // Add random resource requirements
        std::uniform_int_distribution<> resourceAmountDist(1, 5); // Increased resource requirements
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }




    return activities;
}




std::vector<Participant> generateParticipants(int numParticipants, int maxCpu, int maxMemory) {
    std::vector<Participant> participants;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> cpuDist(5, maxCpu);
    std::uniform_int_distribution<> memoryDist(5, maxMemory);


    for (int m = 0; m < numParticipants; ++m) {
        Participant p(100 + m);
        p.addResourceCapacity("machine" + std::to_string(m), 1.0);
        participants.push_back(p);
    }
    return participants;
}








// Forward declarations for utility functions if needed
std::vector<std::vector<int>> readPTMatrix(const std::string& filename);
void convertMatrixToActivities(
    const std::vector<std::vector<int>>& pt_matrix,
    std::vector<Activity>& activities,
    std::vector<Participant>& participants);
void normalizeActivities(std::vector<Activity>& activities);


// Minimal BenchmarkValidator class
class BenchmarkValidator {
    std::string benchmark_dir;
    std::vector<std::string> instances;
public:
    BenchmarkValidator(const std::string& dir) : benchmark_dir(dir) {}


    void addBenchmarkInstance(const std::string& name) {
        instances.push_back(name);
    }


    struct BenchmarkMetrics {
        double total_weight = 0;
        double fairness = 0;
        double jain_index = 0;
        double runtime_ms = 0;
        int count = 0;
    };


    void runAll() {
        BenchmarkMetrics summary;
        for (const auto& name : instances) {
            std::cout << "\n===== Running benchmark for " << name << " =====" << std::endl;
            std::string file_path = benchmark_dir + "/" + name + ".txt";
            std::vector<Activity> activities;
            std::vector<Participant> participants;


            auto pt_matrix = readPTMatrix(file_path);
            if (!pt_matrix.empty()) {
                convertMatrixToActivities(pt_matrix, activities, participants);
                normalizeActivities(activities);
            } else {
                std::cerr << "Failed to load instance: " << name << std::endl;
                continue;
            }


            HybridAlgorithm algo(activities, participants);
            double adaptive_epsilon = algo.calculateAdaptiveEpsilon();
            algo.setFairnessEpsilon(adaptive_epsilon);


            auto start = std::chrono::high_resolution_clock::now();
            algo.run();
            auto end = std::chrono::high_resolution_clock::now();
            double runtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // Print total metrics for this instance
            std::cout << ">>> Instance totals after all phases:" << std::endl;
            std::cout << "Total weight: " << algo.getLastTotalWeight() << std::endl;
            std::cout << "Fairness: " << algo.getLastFairness() << std::endl;
            std::cout << "Jain's index: " << algo.getLastJainsIndex() << std::endl;
            std::cout << "Makespan: " << algo.getLastMakespan() << std::endl;
            std::cout << "Weighted completion: " << algo.getLastWeightedCompletion() << std::endl;
            std::cout << "Total execution time: " << runtime_ms << " ms" << std::endl;

            // Collect metrics
            // (You may want to expose these from HybridAlgorithm)
            // For now, you can parse from printAllocationSummary or add getters
            // Example:
            // summary.total_weight += algo.getTotalWeight();
            // summary.fairness += algo.getFairness();
            // summary.jain_index += algo.getJainIndex();
            summary.runtime_ms += runtime_ms;
            summary.count++;
        }
        // Print averages
        if (summary.count > 0) {
            std::cout << "\n=== Average Metrics Across All Benchmarks ===" << std::endl;
            std::cout << "Average runtime (ms): " << summary.runtime_ms / summary.count << std::endl;
            // Print other averages if you collect them
        }
    }
};


// Example implementations for utility functions (replace with your actual ones)
std::vector<std::vector<int>> readPTMatrix(const std::string& filename) {
    std::vector<std::vector<int>> matrix;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '/') continue; // skip comments
        std::istringstream iss(line);
        std::vector<int> row;
        int val;
        while (iss >> val) row.push_back(val);
        if (!row.empty()) matrix.push_back(row);
    }
    return matrix;
}


void convertMatrixToActivities(
    const std::vector<std::vector<int>>& pt_matrix,
    std::vector<Activity>& activities,
    std::vector<Participant>& participants)
{
    int num_jobs = pt_matrix.size();
    int num_machines = pt_matrix[0].size();
    // Create participants (machines)
    for (int m = 0; m < num_machines; ++m) {
        Participant p(100 + m);
        p.addResourceCapacity("machine" + std::to_string(m), 1.0);
        participants.push_back(p);
    }
    // Create activities (operations)
    int activity_id = 1;
    for (int j = 0; j < num_jobs; ++j) {
        double current_time = 0;
        for (int m = 0; m < num_machines; ++m) {
            int pt = pt_matrix[j][m];
            double start = current_time;
            double finish = current_time + pt;
            Activity a(activity_id++, j, m, start, finish, pt); // job_id = j, job_op_idx = m
            a.addResource("machine" + std::to_string(m), 1.0);
            activities.push_back(a);
            current_time = finish; // Next operation starts after this one finishes
        }
    }
}


void normalizeActivities(std::vector<Activity>& activities) {
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0;
    double min_weight = std::numeric_limits<double>::max();
    double max_weight = 0;
    for (const auto& a : activities) {
        min_time = std::min(min_time, a.finish_time);
        max_time = std::max(max_time, a.finish_time);
        min_weight = std::min(min_weight, a.weight);
        max_weight = std::max(max_weight, a.weight);
    }
    double T = 100.0;
    for (auto& a : activities) {
        a.start_time = (a.start_time - min_time) / (max_time - min_time) * T;
        a.finish_time = (a.finish_time - min_time) / (max_time - min_time) * T;
        a.weight = 1.0 + (a.weight - min_weight) / (max_weight - min_weight) * 99.0;
    }
}


int main() {
    std::string benchmark_dir = "benchmarks";
    BenchmarkValidator validator(benchmark_dir);


    validator.addBenchmarkInstance("LA16");
    validator.addBenchmarkInstance("LA21");
    validator.addBenchmarkInstance("LA26");
    validator.addBenchmarkInstance("TA31_MA");
    validator.addBenchmarkInstance("TA31_PT");
    validator.addBenchmarkInstance("TA41_MA");
    validator.addBenchmarkInstance("TA41_PT");
    validator.addBenchmarkInstance("TA51_MA");
    validator.addBenchmarkInstance("TA51_PT");
    validator.addBenchmarkInstance("TA61_MA");
    validator.addBenchmarkInstance("TA61_PT");
    validator.addBenchmarkInstance("TA71_MA");
    validator.addBenchmarkInstance("TA71_PT");


    // validator.addBenchmarkInstance("abz5");
    // validator.addBenchmarkInstance("abz6");
    // validator.addBenchmarkInstance("ft06");
    // validator.addBenchmarkInstance("ft10");
    // validator.addBenchmarkInstance("ft20");
    // validator.addBenchmarkInstance("la01");
    // validator.addBenchmarkInstance("la02");
    // validator.addBenchmarkInstance("la03");
    // validator.addBenchmarkInstance("la04");
    // validator.addBenchmarkInstance("la05");




    validator.runAll();


    return 0;
}