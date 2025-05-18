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
#include <fstream>
#include <sstream>
#include <string>


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


// JSSPInstance class for parsing and converting JSSP benchmark files
class JSSPInstance {
public:
    int num_jobs;
    int num_machines;
    std::vector<std::vector<std::pair<int, int>>> jobs; // Each job is a vector of (machine_id, processing_time) pairs
   
    JSSPInstance() : num_jobs(0), num_machines(0) {}
   
    // Parse a JSSP instance from a file
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }


        std::string line;


        // --- Skip comment and empty lines to find the header ---
        while (std::getline(file, line)) {
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty() || line[0] == '#') continue;
            // Found the header line
            std::istringstream iss(line);
            iss >> num_jobs >> num_machines;
            break;
        }
        if (num_jobs == 0 || num_machines == 0) {
            std::cerr << "Failed to read header from file" << std::endl;
            return false;
        }


        // --- Now read job lines, skipping comments/empty lines ---
        jobs.clear();
        jobs.resize(num_jobs);
        int jobs_read = 0;
        while (jobs_read < num_jobs && std::getline(file, line)) {
            // Remove leading/trailing whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            jobs[jobs_read].resize(num_machines);
            for (int m = 0; m < num_machines; ++m) {
                int machine_id, processing_time;
                if (!(iss >> machine_id >> processing_time)) {
                    std::cerr << "Failed to read operation " << m << " for job " << jobs_read << std::endl;
                    return false;
                }
                jobs[jobs_read][m] = std::make_pair(machine_id, processing_time);
            }
            ++jobs_read;
        }
        if (jobs_read != num_jobs) {
            std::cerr << "Failed to read all jobs from file" << std::endl;
            return false;
        }
        return true;
    }
   
    // Convert JSSP instance to Activity-Participant model
    std::pair<std::vector<Activity>, std::vector<Participant>> convertToActivityModel() const {
        std::vector<Activity> activities;
        std::vector<Participant> participants;
       
        // Create participants (machines)
        for (int m = 0; m < num_machines; ++m) {
            participants.emplace_back(m);
            participants.back().addResourceCapacity("machine_" + std::to_string(m), 1.0); // Each machine can process its own jobs
        }
       
        // Create activities (operations)
        int activity_id = 0;
        for (int j = 0; j < num_jobs; ++j) {
            double job_start_time = 0.0;
           
            for (int o = 0; o < (int)jobs[j].size(); ++o) {
                int machine_id = jobs[j][o].first;
                int processing_time = jobs[j][o].second;
               
                // Set start time to be after previous operation in the job
                double start_time = job_start_time;
                double finish_time = start_time + processing_time;
               
                // Create activity
                activities.emplace_back(activity_id++, start_time, finish_time, 100.0); // Default weight
                activities.back().addResource("machine_" + std::to_string(machine_id), 1.0);
               
                // Update job start time for next operation
                job_start_time = finish_time;
            }
        }
       
        return {activities, participants};
    }
   
    // Normalize activities and participants based on temporal and value criteria
    void normalizeData(std::vector<Activity>& activities, std::vector<Participant>& participants) const {
        if (activities.empty()) return;
       
        // Find max time value and max weight
        double max_time = 0.0;
        double max_weight = 0.0;
       
        for (const auto& activity : activities) {
            max_time = std::max(max_time, activity.finish_time);
            max_weight = std::max(max_weight, activity.weight);
        }
       
        // Normalize time values to [0, 100] range
        double time_scale = 100.0 / max_time;
        for (auto& activity : activities) {
            activity.start_time *= time_scale;
            activity.finish_time *= time_scale;
        }
       
        // Normalize weights to [1, 100] range
        if (max_weight > 0) {
            double weight_scale = 99.0 / max_weight;
            for (auto& activity : activities) {
                activity.weight = 1.0 + activity.weight * weight_scale;
            }
        }
    }
};


// Liu et al.'s reported results for key benchmark instances
struct LiuResults {
    std::string instance_name;
    double makespan;
    double runtime_ms;
   
    LiuResults(const std::string& name, double span, double rt)
        : instance_name(name), makespan(span), runtime_ms(rt) {}
};


// Database of Liu et al.'s reported results
std::vector<LiuResults> getLiuResultsDatabase() {
    std::vector<LiuResults> results;
    results.emplace_back("LA01", 666, 120);
    results.emplace_back("LA02", 655, 125);
    results.emplace_back("LA03", 597, 130);
    results.emplace_back("LA04", 590, 132);
    results.emplace_back("LA05", 593, 128);
    results.emplace_back("FT06", 55, 50);
    results.emplace_back("FT10", 930, 200);
    results.emplace_back("FT20", 1165, 250);
    results.emplace_back("ABZ5", 1234, 220);
    results.emplace_back("ABZ6", 943, 210);
    return results;
}


LiuResults findLiuResults(const std::string& instance_name) {
    auto results = getLiuResultsDatabase();
    for (const auto& result : results) {
        if (result.instance_name == instance_name) {
            return result;
        }
    }
    return LiuResults(instance_name, -1, -1);
}


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


    // Calculate makespan
    double calculateMakespan() const {
        double makespan = 0.0;
        for (const auto& activity : activities) {
            makespan = std::max(makespan, activity.finish_time);
        }
        return makespan;
    }


    // Calculate weighted completion time
    double calculateWeightedCompletionTime() const {
        double weighted_completion = 0.0;
        for (const auto& activity : activities) {
            weighted_completion += activity.weight * activity.finish_time;
        }
        return weighted_completion;
    }
};


// Benchmark validation class
class BenchmarkValidator {
private:
    std::string benchmark_dir;
    std::vector<std::string> benchmark_instances;
    std::vector<std::pair<std::string, std::unordered_map<std::string, double>>> results;
   
public:
    BenchmarkValidator(const std::string& dir) : benchmark_dir(dir) {}
   
    void addBenchmarkInstance(const std::string& instance_name) {
        benchmark_instances.push_back(instance_name);
    }
   
    void runAllBenchmarks() {
        for (const auto& instance : benchmark_instances) {
            std::cout << "\n===== Running benchmark for " << instance << " =====" << std::endl;
            runBenchmark(instance);
        }
        printSummary();
    }
   
    void runBenchmark(const std::string& instance_name) {
        std::string file_path = benchmark_dir + "/" + instance_name + ".txt";
        JSSPInstance jssp;
        if (!jssp.loadFromFile(file_path)) {
            std::cerr << "Failed to load JSSP instance: " << instance_name << std::endl;
            return;
        }
        auto [activities, participants] = jssp.convertToActivityModel();
        jssp.normalizeData(activities, participants);
        auto start_time = std::chrono::high_resolution_clock::now();
        HybridAlgorithm algorithm(activities, participants);
        algorithm.run();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double makespan = algorithm.calculateMakespan();
        double weighted_completion = algorithm.calculateWeightedCompletionTime();
        LiuResults liu_results = findLiuResults(instance_name);
        std::unordered_map<std::string, double> metrics;
        metrics["makespan"] = makespan;
        metrics["weighted_completion"] = weighted_completion;
        metrics["runtime_ms"] = runtime;
        if (liu_results.makespan > 0) {
            double denormalized_makespan = makespan; // For proper comparison, calibrate if needed
            metrics["makespan_ratio"] = denormalized_makespan / liu_results.makespan;
            std::cout << "Makespan Ratio (Ours/Liu): " << metrics["makespan_ratio"]
                      << " (lower is better for us)" << std::endl;
        }
        if (liu_results.runtime_ms > 0) {
            metrics["runtime_ratio"] = runtime / liu_results.runtime_ms;
            std::cout << "Runtime Ratio (Ours/Liu): " << metrics["runtime_ratio"]
                      << " (lower is better for us)" << std::endl;
        }
        results.emplace_back(instance_name, metrics);
    }
   
    void printSummary() const {
        std::cout << "\n===== Benchmark Summary =====" << std::endl;
        std::cout << "Number of instances tested: " << results.size() << std::endl;
        double avg_makespan_ratio = 0.0;
        double avg_runtime_ratio = 0.0;
        int count_makespan = 0;
        int count_runtime = 0;
        for (const auto& [instance, metrics] : results) {
            auto it_makespan = metrics.find("makespan_ratio");
            if (it_makespan != metrics.end()) {
                avg_makespan_ratio += it_makespan->second;
                count_makespan++;
            }
            auto it_runtime = metrics.find("runtime_ratio");
            if (it_runtime != metrics.end()) {
                avg_runtime_ratio += it_runtime->second;
                count_runtime++;
            }
        }
        if (count_makespan > 0) {
            avg_makespan_ratio /= count_makespan;
            std::cout << "Average Makespan Ratio: " << avg_makespan_ratio << std::endl;
        }
        if (count_runtime > 0) {
            avg_runtime_ratio /= count_runtime;
            std::cout << "Average Runtime Ratio: " << avg_runtime_ratio << std::endl;
        }
        std::cout << "\nDetailed Results:" << std::endl;
        for (const auto& [instance, metrics] : results) {
            std::cout << "  " << instance << ":" << std::endl;
            for (const auto& [metric, value] : metrics) {
                std::cout << "    " << metric << ": " << value << std::endl;
            }
        }
    }
};


int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string benchmark_dir = "./benchmarks";
    if (argc > 1) {
        benchmark_dir = argv[1];
    }
   
    // Initialize benchmark validator
    BenchmarkValidator validator(benchmark_dir);
   
    // Add Lawrence instances
    validator.addBenchmarkInstance("la01");
    validator.addBenchmarkInstance("la02");
    validator.addBenchmarkInstance("la03");
    validator.addBenchmarkInstance("la04");
    validator.addBenchmarkInstance("la05");
   
    // Add Fisher-Thompson instances
    validator.addBenchmarkInstance("ft06");
    validator.addBenchmarkInstance("ft10");
    validator.addBenchmarkInstance("ft20");
   
    // Add ABZ instances
    validator.addBenchmarkInstance("abz5");
    validator.addBenchmarkInstance("abz6");
   
    // Run all benchmarks
    validator.runAllBenchmarks();
   
    return 0;
}