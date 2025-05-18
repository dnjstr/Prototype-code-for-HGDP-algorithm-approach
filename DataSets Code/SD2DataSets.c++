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








// Class implementing the hybrid algorithm
class HybridAlgorithm {
private:
    std::vector<Activity> activities;
    std::vector<Participant> participants;
    std::unordered_map<int, int> allocation; // Activity ID -> Participant ID
    double fairness_epsilon; // Fairness tolerance parameter
   
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
    }
   
    // Phase 1: Initial Greedy Allocation
    void initialGreedyAllocation() {
        // Sort activities by value density (weight/duration)
        std::vector<int> sorted_activities;
        for (size_t i = 0; i < activities.size(); i++) {
            sorted_activities.push_back(i);
        }




        std::sort(sorted_activities.begin(), sorted_activities.end(),
                  [this](int a, int b) {
                      return activities[a].valueDensity() > activities[b].valueDensity();
                  });




        // Initialize available time and resource usage for each participant
        std::vector<double> available_time(participants.size(), 0);
        std::vector<std::unordered_map<std::string, double>> resource_usage(participants.size());




        // Allocate activities
        for (int activity_idx : sorted_activities) {
            const Activity& activity = activities[activity_idx];




            // Find the best participant for this activity
            int best_participant = -1;
            double min_value = std::numeric_limits<double>::max();




            for (size_t j = 0; j < participants.size(); j++) {
                if (available_time[j] <= activity.start_time &&
                    hasResourceCapacity(j, activity) &&
                    resourceUsageFits(j, activity, resource_usage[j]) &&
                    min_value > available_time[j]) {
                    best_participant = j;
                    min_value = available_time[j];
                }
            }




            // Assign the activity if a valid participant is found
            if (best_participant != -1) {
                allocation[activity.id] = participants[best_participant].id;
                available_time[best_participant] = activity.finish_time;




                // Update resource usage
                for (const auto& [resource, amount] : activity.resources) {
                    resource_usage[best_participant][resource] += amount;
                }
            }
        }
    }
   
    // Phase 2: Dynamic Programming Refinement
    void dynamicProgrammingRefinement() {
        for (size_t p_idx = 0; p_idx < participants.size(); p_idx++) {
            const Participant& participant = participants[p_idx];




            // Collect activities assigned to this participant
            std::vector<int> assigned_activities;
            for (size_t i = 0; i < activities.size(); i++) {
                if (allocation[activities[i].id] == participant.id) {
                    assigned_activities.push_back(i);
                }
            }




            // Find all unassigned activities that this participant could potentially handle
            std::vector<int> candidate_activities;
            for (size_t i = 0; i < activities.size(); i++) {
                if (allocation[activities[i].id] == -1 && hasResourceCapacity(p_idx, activities[i])) {
                    candidate_activities.push_back(i);
                }
            }




            // Combine assigned and candidate activities
            std::vector<int> all_activities = assigned_activities;
            all_activities.insert(all_activities.end(), candidate_activities.begin(), candidate_activities.end());




            // Run weighted interval scheduling on the combined set
            std::vector<int> optimal_subset = weightedIntervalScheduling(all_activities);




            // Update allocation - only keep activities assigned to this participant
            for (int act_idx : assigned_activities) {
                allocation[activities[act_idx].id] = -1;
            }
            for (int act_idx : optimal_subset) {
                allocation[activities[act_idx].id] = participant.id;
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
            std::vector<int> high_value_participants;
            std::vector<int> low_value_participants;




            for (size_t i = 0; i < participants.size(); i++) {
                if (values[i] > target_value + fairness_epsilon) {
                    high_value_participants.push_back(i);
                } else if (values[i] < target_value - fairness_epsilon) {
                    low_value_participants.push_back(i);
                }
            }




            // Optimize fairness by transferring activities
            for (int high_idx : high_value_participants) {
                for (int low_idx : low_value_participants) {
                    if (values[high_idx] <= target_value + fairness_epsilon ||
                        values[low_idx] >= target_value - fairness_epsilon) {
                        continue;
                    }




                    // Find the best activity to transfer
                    int best_activity = -1;
                    double best_improvement = -1;




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
                                best_activity = i;
                            }
                        }
                    }




                    // Transfer the best activity if found
                    if (best_activity != -1) {
                        const Activity& activity = activities[best_activity];
                        allocation[activity.id] = participants[low_idx].id;
                        values[high_idx] -= activity.weight;
                        values[low_idx] += activity.weight;
                        improved = true;
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
   
    // Check if an activity can be transferred between participants
    bool canTransferActivity(int from_idx, int to_idx, int activity_idx) const {
        const Activity& activity = activities[activity_idx];
       
        // Check if target participant has resource capacity
        if (!hasResourceCapacity(to_idx, activity)) {
            return false;
        }
       
        // Check if activity conflicts with target participant's timeline
        for (size_t i = 0; i < activities.size(); i++) {
            if (i != activity_idx &&
                allocation.count(activities[i].id) > 0 && // Check if key exists first
                allocation.at(activities[i].id) == participants[to_idx].id &&
                activitiesOverlap(activities[i], activity)) {
                return false;
            }
        }
       
        return true;
    }
   
    // Construct timeline of non-overlapping time slots
    std::vector<TimeSlot> constructTimeline(const std::vector<int>& assigned_activities) const {
        double horizon = 0;
        for (size_t i = 0; i < activities.size(); i++) {
            horizon = std::max(horizon, activities[i].finish_time);
        }
       
        // Start with the entire time horizon
        std::vector<TimeSlot> timeline;
        timeline.emplace_back(0, horizon);
       
        // Remove time slots occupied by assigned activities
        for (int act_idx : assigned_activities) {
            const Activity& activity = activities[act_idx];
           
            for (auto it = timeline.begin(); it != timeline.end(); ) {
                // Check if current slot overlaps with activity
                if (it->start_time < activity.finish_time && it->end_time > activity.start_time) {
                    // Activity overlaps with this slot
                    double slot_start = it->start_time;
                    double slot_end = it->end_time;
                   
                    // Remove current slot
                    it = timeline.erase(it);
                   
                    // Add back segments that don't overlap with activity
                    if (slot_start < activity.start_time) {
                        timeline.emplace_back(slot_start, activity.start_time);
                    }
                    if (slot_end > activity.finish_time) {
                        timeline.emplace_back(activity.finish_time, slot_end);
                    }
                } else {
                    ++it;
                }
            }
        }
       
        // Sort timeline by start time
        std::sort(timeline.begin(), timeline.end(),
                 [](const TimeSlot& a, const TimeSlot& b) {
                     return a.start_time < b.start_time;
                 });
                 
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
};








// Generate SD-2 dataset
std::vector<Activity> generateActivities(int numActivities, int maxStartTime, int maxDuration, int maxWeight) {
    std::vector<Activity> activities;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> startDist(0, maxStartTime);
    std::uniform_int_distribution<> durationDist(1, maxDuration);
   
    // Normal distribution for weights
    std::normal_distribution<> weightDist(maxWeight/2, maxWeight/6); // mean = maxWeight/2, stddev = maxWeight/6


    for (int i = 1; i <= numActivities; ++i) {
        int start = startDist(gen);
        int duration = durationDist(gen);
        // Generate weight using normal distribution and clamp to valid range
        double raw_weight = weightDist(gen);
        int weight = std::max(1, std::min(maxWeight, static_cast<int>(raw_weight)));


        activities.emplace_back(i, start, start + duration, weight);


        // Add random resource requirements
        std::uniform_int_distribution<> resourceAmountDist(1, 3);
        activities.back().addResource("cpu", resourceAmountDist(gen));
        activities.back().addResource("memory", resourceAmountDist(gen));
    }


    return activities;
}




std::vector<Participant> generateParticipants(int numParticipants, int maxCpu, int maxMemory) {
    std::vector<Participant> participants;
    std::mt19937 gen(42); // Fixed seed for consistent results
    std::uniform_int_distribution<> cpuDist(1, maxCpu);
    std::uniform_int_distribution<> memoryDist(1, maxMemory);


    for (int i = 1; i <= numParticipants; ++i) {
        participants.emplace_back(100 + i);
        participants.back().addResourceCapacity("cpu", cpuDist(gen));
        participants.back().addResourceCapacity("memory", memoryDist(gen));
    }


    return participants;
}




int main() {
    // Generate SD-2 dataset
    int numActivities = 200;  // Medium-scale dataset
    int numParticipants = 10; // 10 participants
    int maxStartTime = 50;    // Increased time horizon
    int maxDuration = 15;     // Slightly longer activities
    int maxWeight = 100;      // Higher weight range
    int maxCpu = 8;          // Increased resource capacity
    int maxMemory = 8;       // Increased resource capacity


    std::vector<Activity> activities = generateActivities(numActivities, maxStartTime, maxDuration, maxWeight);
    std::vector<Participant> participants = generateParticipants(numParticipants, maxCpu, maxMemory);


    // Print problem instance
    std::cout << "Problem Instance (SD-2):" << std::endl;
    std::cout << "Activities: " << activities.size() << std::endl;
    std::cout << "Participants: " << participants.size() << std::endl;
   
    // Print summary statistics
    double total_weight = 0;
    double min_weight = std::numeric_limits<double>::max();
    double max_weight = 0;
    for (const auto& activity : activities) {
        total_weight += activity.weight;
        min_weight = std::min(min_weight, static_cast<double>(activity.weight));
        max_weight = std::max(max_weight, static_cast<double>(activity.weight));
    }
    double avg_weight = total_weight / activities.size();
   
    std::cout << "\nActivity Weight Statistics:" << std::endl;
    std::cout << "  Minimum weight: " << min_weight << std::endl;
    std::cout << "  Maximum weight: " << max_weight << std::endl;
    std::cout << "  Average weight: " << avg_weight << std::endl;
    std::cout << "  Total weight: " << total_weight << std::endl;


    // Create and run the hybrid algorithm
    std::cout << "\nRunning Hybrid Algorithm..." << std::endl;
    HybridAlgorithm algo(activities, participants);


    // Calculate and set adaptive epsilon
    double adaptive_epsilon = algo.calculateAdaptiveEpsilon();
    std::cout << "Adaptive fairness epsilon: " << adaptive_epsilon << std::endl;
    algo.setFairnessEpsilon(adaptive_epsilon);


    // Run the algorithm
    std::cout << "\nExecuting Algorithm..." << std::endl;
    algo.run();


    // Print final allocation
    std::cout << "\nFinal Allocation:" << std::endl;
    auto allocation = algo.getAllocation();
    int assigned_count = 0;
    double total_assigned_weight = 0;
   
    for (const auto& activity : activities) {
        if (allocation.count(activity.id) > 0) {
            int participant_id = allocation[activity.id];
            if (participant_id != -1) {
                assigned_count++;
                total_assigned_weight += activity.weight;
            }
        }
    }
   
    std::cout << "Assigned activities: " << assigned_count << "/" << activities.size()
              << " (" << (assigned_count * 100.0 / activities.size()) << "%)" << std::endl;
    std::cout << "Total assigned weight: " << total_assigned_weight
              << " (" << (total_assigned_weight * 100.0 / total_weight) << "% of total weight)" << std::endl;


    return 0;
}