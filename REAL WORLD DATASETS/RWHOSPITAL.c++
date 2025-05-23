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




// Enum for medical activity types
enum class ActivityType {
    SURGERY,
    CONSULTATION,
    EMERGENCY,
    DIAGNOSTIC,
    THERAPY,
    ROUTINE_CHECKUP
};




// Enum for healthcare professional types
enum class ProfessionalType {
    SURGEON,
    PHYSICIAN,
    NURSE,
    SPECIALIST,
    TECHNICIAN,
    THERAPIST
};




// Helper function to convert activity type to string
std::string activityTypeToString(ActivityType type) {
    switch (type) {
        case ActivityType::SURGERY: return "Surgery";
        case ActivityType::CONSULTATION: return "Consultation";
        case ActivityType::EMERGENCY: return "Emergency";
        case ActivityType::DIAGNOSTIC: return "Diagnostic";
        case ActivityType::THERAPY: return "Therapy";
        case ActivityType::ROUTINE_CHECKUP: return "Routine Checkup";
        default: return "Unknown";
    }
}




// Helper function to convert professional type to string
std::string professionalTypeToString(ProfessionalType type) {
    switch (type) {
        case ProfessionalType::SURGEON: return "Surgeon";
        case ProfessionalType::PHYSICIAN: return "Physician";
        case ProfessionalType::NURSE: return "Nurse";
        case ProfessionalType::SPECIALIST: return "Specialist";
        case ProfessionalType::TECHNICIAN: return "Technician";
        case ProfessionalType::THERAPIST: return "Therapist";
        default: return "Unknown";
    }
}




// Class representing a medical activity
class Activity {
public:
    int id;
    double start_time;
    double finish_time;
    double weight;
    ActivityType type;
    std::string patient_id;
    int urgency_level; // 1-5, where 5 is most urgent
    std::unordered_map<std::string, double> resources;




    Activity(int id, double start, double finish, double weight, ActivityType type,
             const std::string& patient, int urgency)
        : id(id), start_time(start), finish_time(finish), weight(weight),
          type(type), patient_id(patient), urgency_level(urgency) {}




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
        std::cout << "Activity " << id << " (" << activityTypeToString(type) << "): ["
                  << start_time << ", " << finish_time << "), weight=" << weight
                  << ", patient=" << patient_id << ", urgency=" << urgency_level << std::endl;
    }
};




// Class representing a healthcare professional
class Participant {
public:
    int id;
    ProfessionalType type;
    std::string name;
    int experience_years;
    std::string department;
    std::unordered_map<std::string, double> resource_capacities;
   
    Participant(int id, ProfessionalType type, const std::string& name,
                int experience, const std::string& dept)
        : id(id), type(type), name(name), experience_years(experience), department(dept) {}
   
    // Add resource capacity
    void addResourceCapacity(const std::string& resource_name, double capacity) {
        resource_capacities[resource_name] = capacity;
    }
   
    // Print participant details
    void print() const {
        std::cout << "Professional " << id << " (" << professionalTypeToString(type) << "): "
                  << name << ", Experience: " << experience_years << " years, Dept: " << department << std::endl;
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
        // Sort activities by urgency first, then by value density
        std::vector<int> sorted_activities;
        for (size_t i = 0; i < activities.size(); i++) {
            sorted_activities.push_back(i);
        }




        std::sort(sorted_activities.begin(), sorted_activities.end(),
                  [this](int a, int b) {
                      if (activities[a].urgency_level != activities[b].urgency_level) {
                          return activities[a].urgency_level > activities[b].urgency_level;
                      }
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
                    isCompatibleProfessional(j, activity) &&
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
                if (allocation[activities[i].id] == -1 && hasResourceCapacity(p_idx, activities[i]) &&
                    isCompatibleProfessional(p_idx, activities[i])) {
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




    // Check if professional is compatible with activity type
    bool isCompatibleProfessional(int participant_idx, const Activity& activity) const {
        const Participant& participant = participants[participant_idx];
       
        switch (activity.type) {
            case ActivityType::SURGERY:
                return participant.type == ProfessionalType::SURGEON;
            case ActivityType::CONSULTATION:
                return participant.type == ProfessionalType::PHYSICIAN ||
                       participant.type == ProfessionalType::SPECIALIST;
            case ActivityType::EMERGENCY:
                return participant.type == ProfessionalType::PHYSICIAN ||
                       participant.type == ProfessionalType::NURSE ||
                       participant.type == ProfessionalType::SURGEON;
            case ActivityType::DIAGNOSTIC:
                return participant.type == ProfessionalType::TECHNICIAN ||
                       participant.type == ProfessionalType::PHYSICIAN;
            case ActivityType::THERAPY:
                return participant.type == ProfessionalType::THERAPIST ||
                       participant.type == ProfessionalType::NURSE;
            case ActivityType::ROUTINE_CHECKUP:
                return participant.type == ProfessionalType::NURSE ||
                       participant.type == ProfessionalType::PHYSICIAN;
            default:
                return false;
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
       
        // Check if target participant has resource capacity and is compatible
        if (!hasResourceCapacity(to_idx, activity) || !isCompatibleProfessional(to_idx, activity)) {
            return false;
        }
       
        // Check if activity conflicts with target participant's timeline
        for (size_t i = 0; i < activities.size(); i++) {
            if (i != activity_idx &&
                allocation.count(activities[i].id) > 0 &&
                allocation.at(activities[i].id) == participants[to_idx].id &&
                activitiesOverlap(activities[i], activity)) {
                return false;
            }
        }
       
        return true;
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
       
        // Calculate fairness metrics
        double fairness = calculateFairness(participant_values);
        double jains_index = calculateJainsIndex(participant_values);
       
        // Calculate weighted completion and total importance
        double weighted_completion = calculateWeightedCompletion();
        double total_importance = calculateTotalImportance();
       
        std::cout << "Assigned activities: " << assigned_count << "/" << activities.size() << std::endl;
        std::cout << "Total weight: " << total_weight << std::endl;
        std::cout << "Weighted completion: " << std::fixed << std::setprecision(0) << weighted_completion << std::endl;
        std::cout << "Total importance: " << std::fixed << std::setprecision(2) << total_importance << std::endl;
        std::cout << "Fairness: " << fairness << std::endl;
        std::cout << "Jain's index: " << jains_index << std::endl;
       
        // Print distribution for each participant
        std::cout << "Value distribution:" << std::endl;
        for (size_t i = 0; i < participants.size(); i++) {
            std::cout << "  " << professionalTypeToString(participants[i].type)
                      << " " << participants[i].name << ": " << participant_values[i] << std::endl;
        }
    }


    // Calculate weighted completion rate
    double calculateWeightedCompletion() const {
        double assigned_weight = 0;
       
        for (const auto& activity : activities) {
            if (allocation.count(activity.id) > 0 && allocation.at(activity.id) != -1) {
                assigned_weight += activity.weight;
            }
        }
       
        return assigned_weight;
    }
   
    // Calculate total importance (sum of urgency * weight for assigned activities)
    double calculateTotalImportance() const {
        double total_importance = 0;
       
        for (const auto& activity : activities) {
            if (allocation.count(activity.id) > 0 && allocation.at(activity.id) != -1) {
                total_importance += activity.urgency_level * activity.weight;
            }
        }
       
        return total_importance;
    }




    // Helper functions (same as before)
    int getParticipantIndex(int participant_id) const {
        for (size_t i = 0; i < participants.size(); i++) {
            if (participants[i].id == participant_id) {
                return i;
            }
        }
        return -1;
    }
   
    bool activitiesOverlap(const Activity& a, const Activity& b) const {
        return a.start_time < b.finish_time && a.finish_time > b.start_time;
    }
   
    double calculateFairness(const std::vector<double>& values) const {
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        if (sum == 0) return 1.0;
       
        double avg = sum / values.size();
       
        double sum_squared_diff = 0;
        for (double val : values) {
            sum_squared_diff += (val - avg) * (val - avg);
        }
       
        return 1.0 - std::sqrt(sum_squared_diff / (values.size() * avg * avg));
    }
   
    double calculateJainsIndex(const std::vector<double>& values) const {
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        double sum_squared = 0;
        for (double val : values) {
            sum_squared += val * val;
        }
       
        if (sum_squared == 0) return 1.0;
        return (sum * sum) / (values.size() * sum_squared);
    }
   
    std::unordered_map<int, int> getAllocation() const {
        return allocation;
    }
   
    void setFairnessEpsilon(double epsilon) {
        fairness_epsilon = epsilon;
    }
   
    double calculateAdaptiveEpsilon() const {
        std::vector<double> weights;
        for (const auto& activity : activities) {
            weights.push_back(activity.weight);
        }
       
        double mean = std::accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
       
        double variance = 0;
        for (double w : weights) {
            variance += (w - mean) * (w - mean);
        }
        variance /= weights.size();
       
        double std_dev = std::sqrt(variance);
        double cv = (mean > 0) ? std_dev / mean : 0;
       
        double min_weight = *std::min_element(weights.begin(), weights.end());
        double max_weight = *std::max_element(weights.begin(), weights.end());
        double range = max_weight - min_weight;
       
        double skewness = 0;
        if (std_dev > 0) {
            for (double w : weights) {
                skewness += std::pow((w - mean) / std_dev, 3);
            }
            skewness /= weights.size();
        }
       
        double base_epsilon = 0.05;
        double epsilon_distribution = base_epsilon * (1 + cv) * (1 + std::abs(skewness) * 0.1);
        if (mean > 0) {
            epsilon_distribution *= (range / mean * 0.1);
        }
       
        double epsilon_participant = std::log10(participants.size() + 1) / std::log10(11);
        double epsilon = epsilon_distribution * (1 + epsilon_participant);
       
        return std::min(std::max(epsilon, 0.01), 0.25);
    }




    bool resourceUsageFits(int participant_idx, const Activity& activity,
                          const std::unordered_map<std::string, double>& current_usage) const {
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




// Generate synthetic hospital activities
std::vector<Activity> generateHospitalActivities(int numActivities, int maxTimeHours = 24) {
    std::vector<Activity> activities;
   
    // Use random seed for true randomness each run
    std::random_device rd;
    std::mt19937 gen(rd()); // Random seed instead of fixed seed
   
    // Time distributions (in hours)
    std::uniform_real_distribution<> startDist(0, maxTimeHours - 2); // Leave room for duration
   
    // Activity type probabilities - Emergency department focused
    std::vector<ActivityType> activityTypes = {
        ActivityType::EMERGENCY, ActivityType::EMERGENCY, ActivityType::EMERGENCY, // Higher emergency probability
        ActivityType::DIAGNOSTIC, ActivityType::DIAGNOSTIC,
        ActivityType::CONSULTATION, ActivityType::CONSULTATION,
        ActivityType::SURGERY,
        ActivityType::THERAPY,
        ActivityType::ROUTINE_CHECKUP
    };
    std::uniform_int_distribution<> typeDist(0, activityTypes.size() - 1);
   
    // Higher urgency distribution for emergency department (1-5)
    std::discrete_distribution<> urgencyDist({5, 10, 25, 35, 25}); // More high urgency cases
   
    for (int i = 1; i <= numActivities; ++i) {
        double start = startDist(gen);
        ActivityType type = activityTypes[typeDist(gen)];
        int urgency = urgencyDist(gen) + 1; // Convert to 1-5 scale
       
        // Duration and weight based on activity type - Emergency department focused
        double duration, weight;
        switch (type) {
            case ActivityType::SURGERY:
                duration = std::uniform_real_distribution<>(1.5, 4.0)(gen); // Shorter emergency surgeries
                weight = std::uniform_real_distribution<>(85.0, 100.0)(gen);
                break;
            case ActivityType::CONSULTATION:
                duration = std::uniform_real_distribution<>(0.25, 1.0)(gen); // Quick consultations
                weight = std::uniform_real_distribution<>(25.0, 50.0)(gen);
                break;
            case ActivityType::EMERGENCY:
                duration = std::uniform_real_distribution<>(0.5, 3.0)(gen); // Variable emergency times
                weight = std::uniform_real_distribution<>(60.0, 95.0)(gen);
                break;
            case ActivityType::DIAGNOSTIC:
                duration = std::uniform_real_distribution<>(0.5, 1.5)(gen); // Quick diagnostics
                weight = std::uniform_real_distribution<>(35.0, 65.0)(gen);
                break;
            case ActivityType::THERAPY:
                duration = std::uniform_real_distribution<>(1.0, 2.5)(gen);
                weight = std::uniform_real_distribution<>(45.0, 70.0)(gen);
                break;
            case ActivityType::ROUTINE_CHECKUP:
                duration = std::uniform_real_distribution<>(0.25, 0.75)(gen); // Quick checkups
                weight = std::uniform_real_distribution<>(15.0, 35.0)(gen);
                break;
        }
       
        // Ensure activity doesn't exceed time horizon
        if (start + duration > maxTimeHours) {
            duration = maxTimeHours - start - 0.1; // Leave small buffer
        }
       
        // Adjust weight based on urgency (emergency department priority)
        weight *= (0.7 + 0.15 * urgency);
       
        std::string patient_id = "ED" + std::to_string(2000 + i); // Emergency department prefix
        activities.emplace_back(i, start, start + duration, weight, type, patient_id, urgency);
       
        // Add resource requirements based on activity type - Emergency focused
        switch (type) {
            case ActivityType::SURGERY:
                activities.back().addResource("emergency_or", 1);
                activities.back().addResource("medical_equipment", std::uniform_int_distribution<>(2, 3)(gen));
                activities.back().addResource("nursing_support", std::uniform_int_distribution<>(2, 3)(gen));
                break;
            case ActivityType::CONSULTATION:
                activities.back().addResource("triage_room", 1);
                activities.back().addResource("medical_equipment", 1);
                break;
            case ActivityType::EMERGENCY:
                activities.back().addResource("emergency_bed", 1);
                activities.back().addResource("medical_equipment", std::uniform_int_distribution<>(1, 3)(gen));
                activities.back().addResource("nursing_support", std::uniform_int_distribution<>(1, 2)(gen));
                if (urgency >= 4) {
                    activities.back().addResource("resuscitation_equipment", 1);
                }
                break;
            case ActivityType::DIAGNOSTIC:
                activities.back().addResource("diagnostic_station", 1);
                activities.back().addResource("medical_equipment", std::uniform_int_distribution<>(1, 2)(gen));
                break;
            case ActivityType::THERAPY:
                activities.back().addResource("treatment_room", 1);
                activities.back().addResource("therapy_equipment", std::uniform_int_distribution<>(1, 2)(gen));
                activities.back().addResource("nursing_support", 1);
                break;
            case ActivityType::ROUTINE_CHECKUP:
                activities.back().addResource("triage_room", 1);
                activities.back().addResource("medical_equipment", 1);
                break;
        }
    }
   
    return activities;
}




// Generate synthetic healthcare professionals
std::vector<Participant> generateHealthcareProfessionals(int numProfessionals) {
    std::vector<Participant> professionals;
   
    // Use random seed for true randomness each run
    std::random_device rd;
    std::mt19937 gen(rd()); // Random seed instead of fixed seed
   
    // Emergency department staff distribution
    std::vector<ProfessionalType> profTypes = {
        ProfessionalType::PHYSICIAN, ProfessionalType::PHYSICIAN, ProfessionalType::PHYSICIAN, // More physicians
        ProfessionalType::NURSE, ProfessionalType::NURSE, ProfessionalType::NURSE, ProfessionalType::NURSE, // More nurses
        ProfessionalType::SURGEON, ProfessionalType::SURGEON, // Emergency surgeons
        ProfessionalType::SPECIALIST,
        ProfessionalType::TECHNICIAN, ProfessionalType::TECHNICIAN,
        ProfessionalType::THERAPIST
    };
    std::uniform_int_distribution<> profDist(0, profTypes.size() - 1);
   
    // Emergency department specific names
    std::vector<std::string> physicianNames = {"Dr. Emergency", "Dr. Trauma", "Dr. Critical", "Dr. Urgent", "Dr. Rapid"};
    std::vector<std::string> nurseNames = {"RN Triage", "RN Response", "RN Acute", "RN Care", "RN Swift"};
    std::vector<std::string> surgeonNames = {"Dr. Blade", "Dr. Scalpel", "Dr. Stitch", "Dr. Repair", "Dr. Mend"};
    std::vector<std::string> specialistNames = {"Dr. Expert", "Dr. Focus", "Dr. Precision", "Dr. Insight", "Dr. Analysis"};
    std::vector<std::string> technicianNames = {"Tech Scan", "Tech Image", "Tech Test", "Tech Lab", "Tech Vital"};
    std::vector<std::string> therapistNames = {"PT Mobility", "PT Recovery", "PT Strength", "PT Motion", "PT Heal"};
   
    // Emergency department specific departments
    std::vector<std::string> departments = {"Emergency Medicine", "Trauma Surgery", "Critical Care", "Emergency Radiology",
                                          "Emergency Cardiology", "Pediatric Emergency", "Psychiatric Emergency",
                                          "Emergency Orthopedics", "Toxicology", "Emergency Neurology"};
    std::uniform_int_distribution<> deptDist(0, departments.size() - 1);
    std::uniform_int_distribution<> experienceDist(2, 20); // 2-20 years emergency experience
   
    for (int i = 1; i <= numProfessionals; ++i) {
        ProfessionalType type = profTypes[profDist(gen)];
        std::string name;
       
        // Select name based on professional type
        std::uniform_int_distribution<> nameDist(0, 4);
        switch (type) {
            case ProfessionalType::SURGEON:
                name = surgeonNames[nameDist(gen)];
                break;
            case ProfessionalType::PHYSICIAN:
                name = physicianNames[nameDist(gen)];
                break;
            case ProfessionalType::NURSE:
                name = nurseNames[nameDist(gen)];
                break;
            case ProfessionalType::SPECIALIST:
                name = specialistNames[nameDist(gen)];
                break;
            case ProfessionalType::TECHNICIAN:
                name = technicianNames[nameDist(gen)];
                break;
            case ProfessionalType::THERAPIST:
                name = therapistNames[nameDist(gen)];
                break;
        }
       
        name += "-" + std::to_string(100 + i); // Add unique identifier
        int experience = experienceDist(gen);
        std::string department = departments[deptDist(gen)];
       
        professionals.emplace_back(300 + i, type, name, experience, department);
       
        // Add emergency department specific resource capacities
        switch (type) {
            case ProfessionalType::SURGEON:
                professionals.back().addResourceCapacity("emergency_or", 1);
                professionals.back().addResourceCapacity("medical_equipment", 3 + experience/4);
                professionals.back().addResourceCapacity("nursing_support", 2 + experience/8);
                professionals.back().addResourceCapacity("resuscitation_equipment", 1);
                break;
            case ProfessionalType::PHYSICIAN:
                professionals.back().addResourceCapacity("triage_room", 1);
                professionals.back().addResourceCapacity("emergency_bed", 2);
                professionals.back().addResourceCapacity("treatment_room", 1);
                professionals.back().addResourceCapacity("medical_equipment", 2 + experience/6);
                professionals.back().addResourceCapacity("resuscitation_equipment", 1);
                break;
            case ProfessionalType::NURSE:
                professionals.back().addResourceCapacity("nursing_support", 4); // High nursing support capacity
                professionals.back().addResourceCapacity("emergency_bed", 2);
                professionals.back().addResourceCapacity("triage_room", 1);
                professionals.back().addResourceCapacity("treatment_room", 1);
                professionals.back().addResourceCapacity("medical_equipment", 1 + experience/8);
                break;
            case ProfessionalType::SPECIALIST:
                professionals.back().addResourceCapacity("triage_room", 1);
                professionals.back().addResourceCapacity("emergency_bed", 1);
                professionals.back().addResourceCapacity("medical_equipment", 2 + experience/5);
                professionals.back().addResourceCapacity("diagnostic_station", 1);
                break;
            case ProfessionalType::TECHNICIAN:
                professionals.back().addResourceCapacity("diagnostic_station", 2);
                professionals.back().addResourceCapacity("medical_equipment", 2 + experience/6);
                break;
            case ProfessionalType::THERAPIST:
                professionals.back().addResourceCapacity("treatment_room", 1);
                professionals.back().addResourceCapacity("therapy_equipment", 2 + experience/8);
                professionals.back().addResourceCapacity("nursing_support", 1);
                break;
        }
    }
   
    return professionals;
}




int main() {
// Generate Emergency Department Dataset
    int numActivities = 75;       // Emergency department activities for the day
    int numProfessionals = 12;    // Emergency department staff
    int maxTimeHours = 24;        // 24-hour time horizon
   
    std::vector<Activity> activities = generateHospitalActivities(numActivities, maxTimeHours);
    std::vector<Participant> professionals = generateHealthcareProfessionals(numProfessionals);
   
    // Print problem instance
    std::cout << "=== Hospital Scheduling System ===" << std::endl;
    std::cout << "Medical Activities: " << activities.size() << std::endl;
    std::cout << "Healthcare Professionals: " << professionals.size() << std::endl;
    std::cout << "Time Horizon: " << maxTimeHours << " hours" << std::endl;
   
    // Print activity type distribution
    std::map<ActivityType, int> activityCounts;
    for (const auto& activity : activities) {
        activityCounts[activity.type]++;
    }
   
    std::cout << "\nActivity Type Distribution:" << std::endl;
    for (const auto& [type, count] : activityCounts) {
        std::cout << "  " << activityTypeToString(type) << ": " << count << std::endl;
    }
   
    // Print professional type distribution
    std::map<ProfessionalType, int> professionalCounts;
    for (const auto& prof : professionals) {
        professionalCounts[prof.type]++;
    }
   
    std::cout << "\nProfessional Type Distribution:" << std::endl;
    for (const auto& [type, count] : professionalCounts) {
        std::cout << "  " << professionalTypeToString(type) << ": " << count << std::endl;
    }
   
    // Print summary statistics
    double total_weight = 0;
    double min_weight = std::numeric_limits<double>::max();
    double max_weight = 0;
    int total_urgency = 0;
   
    for (const auto& activity : activities) {
        total_weight += activity.weight;
        min_weight = std::min(min_weight, activity.weight);
        max_weight = std::max(max_weight, activity.weight);
        total_urgency += activity.urgency_level;
    }
   
    double avg_weight = total_weight / activities.size();
    double avg_urgency = static_cast<double>(total_urgency) / activities.size();
   
    std::cout << "\nActivity Statistics:" << std::endl;
    std::cout << "  Weight - Min: " << std::fixed << std::setprecision(2) << min_weight
              << ", Max: " << max_weight << ", Avg: " << avg_weight << std::endl;
    std::cout << "  Average Urgency Level: " << std::setprecision(1) << avg_urgency << std::endl;
    std::cout << "  Total Expected Value: " << std::setprecision(2) << total_weight << std::endl;
   
    // Create and run the hybrid algorithm
    std::cout << "\n=== Running Hospital Scheduling Algorithm ===" << std::endl;
    HybridAlgorithm algo(activities, professionals);
   
    // Calculate and set adaptive epsilon
    double adaptive_epsilon = algo.calculateAdaptiveEpsilon();
    std::cout << "Adaptive fairness epsilon: " << adaptive_epsilon << std::endl;
    algo.setFairnessEpsilon(adaptive_epsilon);
   
    // Run the algorithm
    std::cout << "\nExecuting Algorithm..." << std::endl;
    algo.run();
   
    // Print final results
    std::cout << "\n=== Final Hospital Schedule Results ===" << std::endl;
    auto allocation = algo.getAllocation();
    int assigned_count = 0;
    double total_assigned_weight = 0;
   
    // Count by activity type
    std::map<ActivityType, int> assignedByType;
    std::map<ActivityType, double> weightByType;
   
    for (const auto& activity : activities) {
        if (allocation.count(activity.id) > 0) {
            int professional_id = allocation[activity.id];
            if (professional_id != -1) {
                assigned_count++;
                total_assigned_weight += activity.weight;
                assignedByType[activity.type]++;
                weightByType[activity.type] += activity.weight;
            }
        }
    }
   
    std::cout << "Successfully scheduled: " << assigned_count << "/" << activities.size()
              << " activities (" << std::setprecision(1) << (assigned_count * 100.0 / activities.size()) << "%)" << std::endl;
    std::cout << "Total scheduled value: " << std::setprecision(2) << total_assigned_weight
              << " (" << (total_assigned_weight * 100.0 / total_weight) << "% of total possible)" << std::endl;
   
    std::cout << "\nScheduled Activities by Type:" << std::endl;
    for (const auto& [type, count] : assignedByType) {
        double percentage = (count * 100.0) / activityCounts[type];
        std::cout << "  " << activityTypeToString(type) << ": " << count << "/"
                  << activityCounts[type] << " (" << std::setprecision(1) << percentage << "%)" << std::endl;
    }
   
    // Calculate workload distribution
    std::cout << "\nWorkload Distribution Among Professionals:" << std::endl;
    std::map<ProfessionalType, std::vector<double>> workloadByType;
   
    for (const auto& prof : professionals) {
        double workload = 0;
        int activity_count = 0;
       
        for (const auto& activity : activities) {
            if (allocation.count(activity.id) > 0 && allocation[activity.id] == prof.id) {
                workload += activity.weight;
                activity_count++;
            }
        }
       
        workloadByType[prof.type].push_back(workload);
        std::cout << "  " << professionalTypeToString(prof.type) << " " << prof.name
                  << ": " << activity_count << " activities, value=" << std::setprecision(2) << workload << std::endl;
    }
   
    std::cout << "\n=== Hospital Scheduling Complete ===" << std::endl;
   
    return 0;
}
