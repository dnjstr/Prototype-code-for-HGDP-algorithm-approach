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

// Enum for software development task types
enum class TaskType {
    BACKEND_DEVELOPMENT,
    FRONTEND_DEVELOPMENT,
    DATABASE_DESIGN,
    API_DEVELOPMENT,
    TESTING,
    CODE_REVIEW,
    DOCUMENTATION,
    DEPLOYMENT,
    BUG_FIX,
    FEATURE_RESEARCH,
    UI_UX_DESIGN,
    SECURITY_AUDIT
};

// Enum for developer role types
enum class DeveloperRole {
    SENIOR_BACKEND_DEV,
    SENIOR_FRONTEND_DEV,
    FULLSTACK_DEV,
    JUNIOR_DEV,
    QA_ENGINEER,
    DEVOPS_ENGINEER,
    UI_UX_DESIGNER,
    TECH_LEAD,
    SECURITY_SPECIALIST
};

// Helper function to convert task type to string
std::string taskTypeToString(TaskType type) {
    switch (type) {
        case TaskType::BACKEND_DEVELOPMENT: return "Backend Development";
        case TaskType::FRONTEND_DEVELOPMENT: return "Frontend Development";
        case TaskType::DATABASE_DESIGN: return "Database Design";
        case TaskType::API_DEVELOPMENT: return "API Development";
        case TaskType::TESTING: return "Testing";
        case TaskType::CODE_REVIEW: return "Code Review";
        case TaskType::DOCUMENTATION: return "Documentation";
        case TaskType::DEPLOYMENT: return "Deployment";
        case TaskType::BUG_FIX: return "Bug Fix";
        case TaskType::FEATURE_RESEARCH: return "Feature Research";
        case TaskType::UI_UX_DESIGN: return "UI/UX Design";
        case TaskType::SECURITY_AUDIT: return "Security Audit";
        default: return "Unknown";
    }
}

// Helper function to convert developer role to string
std::string developerRoleToString(DeveloperRole role) {
    switch (role) {
        case DeveloperRole::SENIOR_BACKEND_DEV: return "Senior Backend Developer";
        case DeveloperRole::SENIOR_FRONTEND_DEV: return "Senior Frontend Developer";
        case DeveloperRole::FULLSTACK_DEV: return "Fullstack Developer";
        case DeveloperRole::JUNIOR_DEV: return "Junior Developer";
        case DeveloperRole::QA_ENGINEER: return "QA Engineer";
        case DeveloperRole::DEVOPS_ENGINEER: return "DevOps Engineer";
        case DeveloperRole::UI_UX_DESIGNER: return "UI/UX Designer";
        case DeveloperRole::TECH_LEAD: return "Tech Lead";
        case DeveloperRole::SECURITY_SPECIALIST: return "Security Specialist";
        default: return "Unknown";
    }
}

// Class representing a software development task
class Task {
public:
    int id;
    double start_time;      // Start time in hours from project start
    double finish_time;     // Estimated completion time
    double story_points;    // Task complexity/value (similar to weight)
    TaskType type;
    std::string feature_name;
    int priority_level;     // 1-5, where 5 is highest priority
    std::string sprint_name;
    std::vector<int> dependencies; // Task IDs that must be completed first
    std::unordered_map<std::string, double> resources; // Required resources/skills

    Task(int id, double start, double finish, double points, TaskType type,
         const std::string& feature, int priority, const std::string& sprint)
        : id(id), start_time(start), finish_time(finish), story_points(points),
          type(type), feature_name(feature), priority_level(priority), sprint_name(sprint) {}

    // Add resource requirement (skills, tools, etc.)
    void addResource(const std::string& resource_name, double proficiency_required) {
        resources[resource_name] = proficiency_required;
    }

    // Add dependency
    void addDependency(int task_id) {
        dependencies.push_back(task_id);
    }

    // Productivity density (story points per unit time)
    double productivityDensity() const {
        return story_points / (finish_time - start_time);
    }

    // Print task details
    void print() const {
        std::cout << "Task " << id << " (" << taskTypeToString(type) << "): ["
                  << start_time << ", " << finish_time << "), points=" << story_points
                  << ", feature=" << feature_name << ", priority=" << priority_level
                  << ", sprint=" << sprint_name << std::endl;
    }
};

// Class representing a software developer
class Developer {
public:
    int id;
    DeveloperRole role;
    std::string name;
    int experience_years;
    std::string team_name;
    double daily_capacity;  // Hours per day available
    std::unordered_map<std::string, double> skill_levels; // Skill proficiencies (0-1)
    std::vector<std::string> preferred_technologies;
    
    Developer(int id, DeveloperRole role, const std::string& name,
              int experience, const std::string& team, double capacity)
        : id(id), role(role), name(name), experience_years(experience), 
          team_name(team), daily_capacity(capacity) {}
    
    // Add skill proficiency
    void addSkill(const std::string& skill_name, double proficiency) {
        skill_levels[skill_name] = std::min(1.0, std::max(0.0, proficiency));
    }

    // Add preferred technology
    void addPreferredTech(const std::string& tech) {
        preferred_technologies.push_back(tech);
    }
    
    // Print developer details
    void print() const {
        std::cout << "Developer " << id << " (" << developerRoleToString(role) << "): "
                  << name << ", Experience: " << experience_years << " years, Team: " << team_name
                  << ", Capacity: " << daily_capacity << "h/day" << std::endl;
        std::cout << "  Skills: ";
        for (const auto& [skill, level] : skill_levels) {
            std::cout << skill << "(" << std::setprecision(1) << level << ") ";
        }
        std::cout << std::endl;
    }
};

// Class implementing the hybrid algorithm for software development
class SoftwareDevHybridAlgorithm {
private:
    std::vector<Task> tasks;
    std::vector<Developer> developers;
    std::unordered_map<int, int> allocation; // Task ID -> Developer ID
    double fairness_epsilon;
    std::map<int, std::vector<int>> sprint_tasks; // Sprint -> Task IDs
    
public:
    SoftwareDevHybridAlgorithm(const std::vector<Task>& tasks, const std::vector<Developer>& devs, double epsilon = 0.1)
        : tasks(tasks), developers(devs), fairness_epsilon(epsilon) {
        // Initialize allocation with all tasks unassigned (-1)
        for (const auto& task : tasks) {
            allocation[task.id] = -1;
        }
        
        // Group tasks by sprint
        for (const auto& task : tasks) {
            sprint_tasks[std::hash<std::string>{}(task.sprint_name)].push_back(task.id - 1);
        }
    }
    
    // Run the complete hybrid algorithm
    void run() {
        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "Phase 1: Priority-Based Initial Allocation" << std::endl;
        auto phase1_start = std::chrono::high_resolution_clock::now();
        priorityBasedAllocation();
        auto phase1_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        printTotalWeightedCompletion("Phase 1");
        std::cout << "Phase 1 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase1_end - phase1_start).count()
                  << " ms" << std::endl;

        std::cout << "\nPhase 2: Sprint-Aware Dynamic Programming" << std::endl;
        auto phase2_start = std::chrono::high_resolution_clock::now();
        sprintAwareDynamicProgramming();
        auto phase2_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        printTotalWeightedCompletion("Phase 2");
        std::cout << "Phase 2 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase2_end - phase2_start).count()
                  << " ms" << std::endl;

        std::cout << "\nPhase 3: Team Load Balancing" << std::endl;
        auto phase3_start = std::chrono::high_resolution_clock::now();
        teamLoadBalancing();
        auto phase3_end = std::chrono::high_resolution_clock::now();
        printAllocationSummary();
        printTotalWeightedCompletion("Phase 3");
        std::cout << "Phase 3 runtime: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(phase3_end - phase3_start).count()
                  << " ms" << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\nTotal execution time: " << total_duration.count() << " ms" << std::endl;
    }
    
    // Phase 1: Priority-Based Initial Allocation
    void priorityBasedAllocation() {
        // Sort tasks by priority first, then by productivity density
        std::vector<int> sorted_tasks;
        for (size_t i = 0; i < tasks.size(); i++) {
            sorted_tasks.push_back(i);
        }

        std::sort(sorted_tasks.begin(), sorted_tasks.end(),
                  [this](int a, int b) {
                      if (tasks[a].priority_level != tasks[b].priority_level) {
                          return tasks[a].priority_level > tasks[b].priority_level;
                      }
                      return tasks[a].productivityDensity() > tasks[b].productivityDensity();
                  });

        // Initialize developer workload tracking
        std::vector<double> current_workload(developers.size(), 0);
        std::vector<std::unordered_map<std::string, double>> skill_usage(developers.size());

        // Allocate tasks
        for (int task_idx : sorted_tasks) {
            const Task& task = tasks[task_idx];

            // Check if dependencies are satisfied
            if (!dependenciesSatisfied(task)) {
                continue;
            }

            // Find the best developer for this task
            int best_developer = -1;
            double best_score = -1;

            for (size_t j = 0; j < developers.size(); j++) {
                if (canDeveloperHandleTask(j, task) && 
                    hasAvailableCapacity(j, task, current_workload[j])) {
                    
                    double skill_match = calculateSkillMatch(j, task);
                    double workload_factor = 1.0 / (1.0 + current_workload[j] / developers[j].daily_capacity);
                    double score = skill_match * workload_factor;

                    if (score > best_score) {
                        best_score = score;
                        best_developer = j;
                    }
                }
            }

            // Assign the task if a valid developer is found
            if (best_developer != -1) {
                allocation[task.id] = developers[best_developer].id;
                current_workload[best_developer] += (task.finish_time - task.start_time);

                // Update skill usage
                for (const auto& [skill, required] : task.resources) {
                    skill_usage[best_developer][skill] += required;
                }
            }
        }
    }
    
    // Phase 2: Sprint-Aware Dynamic Programming
    void sprintAwareDynamicProgramming() {
        for (size_t d_idx = 0; d_idx < developers.size(); d_idx++) {
            const Developer& developer = developers[d_idx];

            // Collect tasks assigned to this developer
            std::vector<int> assigned_tasks;
            for (size_t i = 0; i < tasks.size(); i++) {
                if (allocation[tasks[i].id] == developer.id) {
                    assigned_tasks.push_back(i);
                }
            }

            // Find all unassigned tasks that this developer could handle
            std::vector<int> candidate_tasks;
            for (size_t i = 0; i < tasks.size(); i++) {
                if (allocation[tasks[i].id] == -1 && 
                    canDeveloperHandleTask(d_idx, tasks[i]) &&
                    dependenciesSatisfied(tasks[i])) {
                    candidate_tasks.push_back(i);
                }
            }

            // Combine assigned and candidate tasks
            std::vector<int> all_tasks = assigned_tasks;
            all_tasks.insert(all_tasks.end(), candidate_tasks.begin(), candidate_tasks.end());

            // Run weighted interval scheduling considering sprint boundaries
            std::vector<int> optimal_subset = sprintAwareWeightedScheduling(all_tasks, d_idx);

            // Update allocation
            for (int task_idx : assigned_tasks) {
                allocation[tasks[task_idx].id] = -1;
            }
            for (int task_idx : optimal_subset) {
                allocation[tasks[task_idx].id] = developer.id;
            }
        }
    }
    
    // Phase 3: Team Load Balancing
    void teamLoadBalancing() {
        bool improved = true;
        int iteration = 0;

        while (improved && iteration < 10) {
            improved = false;
            iteration++;

            // Calculate current workload distribution
            std::vector<double> workloads(developers.size(), 0);
            std::vector<int> task_counts(developers.size(), 0);
            
            for (size_t i = 0; i < tasks.size(); i++) {
                int developer_id = allocation[tasks[i].id];
                if (developer_id != -1) {
                    int d_idx = getDeveloperIndex(developer_id);
                    if (d_idx != -1) {
                        workloads[d_idx] += tasks[i].story_points;
                        task_counts[d_idx]++;
                    }
                }
            }

            // Calculate target workload
            double total_workload = std::accumulate(workloads.begin(), workloads.end(), 0.0);
            double target_workload = total_workload / developers.size();

            // Identify overloaded and underloaded developers
            std::vector<int> overloaded_devs;
            std::vector<int> underloaded_devs;

            for (size_t i = 0; i < developers.size(); i++) {
                if (workloads[i] > target_workload + fairness_epsilon * target_workload) {
                    overloaded_devs.push_back(i);
                } else if (workloads[i] < target_workload - fairness_epsilon * target_workload) {
                    underloaded_devs.push_back(i);
                }
            }

            // Balance workload by transferring tasks
            for (int over_idx : overloaded_devs) {
                for (int under_idx : underloaded_devs) {
                    if (workloads[over_idx] <= target_workload + fairness_epsilon * target_workload ||
                        workloads[under_idx] >= target_workload - fairness_epsilon * target_workload) {
                        continue;
                    }

                    // Find suitable task to transfer
                    int best_task = findBestTaskToTransfer(over_idx, under_idx, workloads, target_workload);

                    if (best_task != -1) {
                        const Task& task = tasks[best_task];
                        allocation[task.id] = developers[under_idx].id;
                        workloads[over_idx] -= task.story_points;
                        workloads[under_idx] += task.story_points;
                        improved = true;
                    }
                }
            }
        }
    }

    // Check if developer can handle a specific task type
    bool canDeveloperHandleTask(int developer_idx, const Task& task) const {
        const Developer& dev = developers[developer_idx];
        
        switch (task.type) {
            case TaskType::BACKEND_DEVELOPMENT:
                return dev.role == DeveloperRole::SENIOR_BACKEND_DEV ||
                       dev.role == DeveloperRole::FULLSTACK_DEV ||
                       dev.role == DeveloperRole::TECH_LEAD ||
                       (dev.role == DeveloperRole::JUNIOR_DEV && dev.experience_years >= 1);
                       
            case TaskType::FRONTEND_DEVELOPMENT:
                return dev.role == DeveloperRole::SENIOR_FRONTEND_DEV ||
                       dev.role == DeveloperRole::FULLSTACK_DEV ||
                       dev.role == DeveloperRole::UI_UX_DESIGNER ||
                       (dev.role == DeveloperRole::JUNIOR_DEV && dev.experience_years >= 1);
                       
            case TaskType::DATABASE_DESIGN:
                return dev.role == DeveloperRole::SENIOR_BACKEND_DEV ||
                       dev.role == DeveloperRole::FULLSTACK_DEV ||
                       dev.role == DeveloperRole::TECH_LEAD;
                       
            case TaskType::API_DEVELOPMENT:
                return dev.role == DeveloperRole::SENIOR_BACKEND_DEV ||
                       dev.role == DeveloperRole::FULLSTACK_DEV ||
                       dev.role == DeveloperRole::TECH_LEAD ||
                       dev.role == DeveloperRole::SENIOR_FRONTEND_DEV;
                       
            case TaskType::TESTING:
                return dev.role == DeveloperRole::QA_ENGINEER ||
                       dev.role == DeveloperRole::TECH_LEAD ||
                       (dev.experience_years >= 2); // Any experienced dev can do testing
                       
            case TaskType::CODE_REVIEW:
                return dev.role == DeveloperRole::TECH_LEAD ||
                       dev.role == DeveloperRole::SENIOR_BACKEND_DEV ||
                       dev.role == DeveloperRole::SENIOR_FRONTEND_DEV ||
                       (dev.experience_years >= 3);
                       
            case TaskType::DOCUMENTATION:
                return true; // Anyone can do documentation
                
            case TaskType::DEPLOYMENT:
                return dev.role == DeveloperRole::DEVOPS_ENGINEER ||
                       dev.role == DeveloperRole::TECH_LEAD ||
                       dev.role == DeveloperRole::SENIOR_BACKEND_DEV;
                       
            case TaskType::BUG_FIX:
                return true; // Anyone can fix bugs in their domain
                
            case TaskType::FEATURE_RESEARCH:
                return dev.role == DeveloperRole::TECH_LEAD ||
                       dev.role == DeveloperRole::SENIOR_BACKEND_DEV ||
                       dev.role == DeveloperRole::SENIOR_FRONTEND_DEV ||
                       (dev.experience_years >= 3);
                       
            case TaskType::UI_UX_DESIGN:
                return dev.role == DeveloperRole::UI_UX_DESIGNER ||
                       dev.role == DeveloperRole::SENIOR_FRONTEND_DEV;
                       
            case TaskType::SECURITY_AUDIT:
                return dev.role == DeveloperRole::SECURITY_SPECIALIST ||
                       dev.role == DeveloperRole::TECH_LEAD ||
                       (dev.role == DeveloperRole::SENIOR_BACKEND_DEV && dev.experience_years >= 5);
                       
            default:
                return false;
        }
    }
    
    // Calculate skill match between developer and task
    double calculateSkillMatch(int developer_idx, const Task& task) const {
        const Developer& dev = developers[developer_idx];
        
        if (task.resources.empty()) {
            return 0.7; // Base compatibility
        }
        
        double total_match = 0;
        int skill_count = 0;
        
        for (const auto& [skill, required_level] : task.resources) {
            auto it = dev.skill_levels.find(skill);
            if (it != dev.skill_levels.end()) {
                double skill_ratio = it->second / required_level;
                total_match += std::min(1.0, skill_ratio);
            } else {
                total_match += 0.1; // Penalty for missing skill
            }
            skill_count++;
        }
        
        return skill_count > 0 ? total_match / skill_count : 0.5;
    }
    
    // Check if developer has available capacity
    bool hasAvailableCapacity(int developer_idx, const Task& task, double current_workload) const {
        const Developer& dev = developers[developer_idx];
        double task_hours = task.finish_time - task.start_time;
        double max_daily_hours = dev.daily_capacity;
        double max_total_hours = max_daily_hours * 30; // Assume 30-day sprint
        
        return (current_workload + task_hours) <= max_total_hours;
    }
    
    // Check if task dependencies are satisfied
    bool dependenciesSatisfied(const Task& task) const {
        for (int dep_id : task.dependencies) {
            if (allocation.count(dep_id) == 0 || allocation.at(dep_id) == -1) {
                return false;
            }
        }
        return true;
    }
    
    // Sprint-aware weighted interval scheduling
    std::vector<int> sprintAwareWeightedScheduling(const std::vector<int>& task_indices, int developer_idx) const {
        if (task_indices.empty()) {
            return {};
        }
    
        // Sort tasks by finish time
        std::vector<int> sorted = task_indices;
        std::sort(sorted.begin(), sorted.end(),
            [this](int a, int b) {
                return tasks[a].finish_time < tasks[b].finish_time;
            });
    
        // Dynamic programming with sprint awareness
        std::vector<double> dp(sorted.size() + 1, 0);
        std::vector<int> p(sorted.size(), -1);
        
        // Precompute compatibility
        for (size_t i = 1; i < sorted.size(); i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (tasks[sorted[j]].finish_time <= tasks[sorted[i]].start_time) {
                    p[i] = j;
                    break;
                }
            }
        }
        
        // Fill DP table with sprint bonus
        for (size_t i = 1; i <= sorted.size(); i++) {
            const Task& task = tasks[sorted[i-1]];
            double task_value = task.story_points;
            
            // Add sprint continuity bonus
            if (i > 1) {
                const Task& prev_task = tasks[sorted[i-2]];
                if (task.sprint_name == prev_task.sprint_name) {
                    task_value *= 1.1; // 10% bonus for sprint continuity
                }
            }
            
            double include = task_value;
            if (p[i-1] != -1) {
                include += dp[p[i-1]+1];
            }
            dp[i] = std::max(include, dp[i-1]);
        }
    
        // Backtrack to find optimal set
        std::vector<int> result;
        int i = sorted.size();
        while (i > 0) {
            const Task& task = tasks[sorted[i-1]];
            double task_value = task.story_points;
            
            if (i > 1) {
                const Task& prev_task = tasks[sorted[i-2]];
                if (task.sprint_name == prev_task.sprint_name) {
                    task_value *= 1.1;
                }
            }
            
            if (p[i-1] == -1) {
                if (task_value >= dp[i-1]) {
                    result.push_back(sorted[i-1]);
                    i = 0;
                } else {
                    i--;
                }
            } else {
                if (task_value + dp[p[i-1]+1] >= dp[i-1]) {
                    result.push_back(sorted[i-1]);
                    i = p[i-1] + 1;
                } else {
                    i--;
                }
            }
        }
    
        return result;
    }
    
    // Find best task to transfer for load balancing
    int findBestTaskToTransfer(int from_idx, int to_idx, 
                              const std::vector<double>& workloads, double target) const {
        int best_task = -1;
        double best_improvement = -1;
        
        for (size_t i = 0; i < tasks.size(); i++) {
            const Task& task = tasks[i];
            
            if (allocation.count(task.id) > 0 && 
                allocation.at(task.id) == developers[from_idx].id &&
                canDeveloperHandleTask(to_idx, task)) {
                
                double current_deviation = std::abs(workloads[from_idx] - target) + 
                                         std::abs(workloads[to_idx] - target);
                
                double new_from_workload = workloads[from_idx] - task.story_points;
                double new_to_workload = workloads[to_idx] + task.story_points;
                
                double new_deviation = std::abs(new_from_workload - target) + 
                                     std::abs(new_to_workload - target);
                
                double improvement = current_deviation - new_deviation;
                
                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    best_task = i;
                }
            }
        }
        
        return best_task;
    }
    
    // Helper functions
    int getDeveloperIndex(int developer_id) const {
        for (size_t i = 0; i < developers.size(); i++) {
            if (developers[i].id == developer_id) {
                return i;
            }
        }
        return -1;
    }
    
    // Print allocation summary
    void printAllocationSummary() const {
        int assigned_count = 0;
        double total_story_points = 0;
        
        std::vector<double> developer_workloads(developers.size(), 0);
        std::vector<int> developer_task_counts(developers.size(), 0);
        
        for (size_t i = 0; i < tasks.size(); i++) {
            if (allocation.count(tasks[i].id) > 0) {
                int developer_id = allocation.at(tasks[i].id);
                if (developer_id != -1) {
                    assigned_count++;
                    total_story_points += tasks[i].story_points;
                    
                    int d_idx = getDeveloperIndex(developer_id);
                    if (d_idx != -1) {
                        developer_workloads[d_idx] += tasks[i].story_points;
                        developer_task_counts[d_idx]++;
                    }
                }
            }
        }
        
        // Calculate fairness metrics
        double fairness = calculateWorkloadFairness(developer_workloads);
        double jains_index = calculateJainsIndex(developer_workloads);
        
        std::cout << "Assigned tasks: " << assigned_count << "/" << tasks.size() << std::endl;
        std::cout << "Total story points: " << total_story_points << std::endl;
        std::cout << "Workload fairness: " << std::setprecision(3) << fairness << std::endl;
        std::cout << "Jain's fairness index: " << jains_index << std::endl;
        
        // Print workload distribution
        std::cout << "Workload distribution:" << std::endl;
        for (size_t i = 0; i < developers.size(); i++) {
            std::cout << "  " << developerRoleToString(developers[i].role)
                      << " " << developers[i].name << ": " 
                      << developer_task_counts[i] << " tasks, "
                      << std::setprecision(1) << developer_workloads[i] << " points" << std::endl;
        }
    }
    
    double calculateWorkloadFairness(const std::vector<double>& workloads) const {
        double sum = std::accumulate(workloads.begin(), workloads.end(), 0.0);
        if (sum == 0) return 1.0;
        
        double avg = sum / workloads.size();
        
        double sum_squared_diff = 0;
        for (double workload : workloads) {
            sum_squared_diff += (workload - avg) * (workload - avg);
        }
        
        return 1.0 - std::sqrt(sum_squared_diff / (workloads.size() * avg * avg));
    }
    
    double calculateJainsIndex(const std::vector<double>& workloads) const {
        double sum = std::accumulate(workloads.begin(), workloads.end(), 0.0);
        double sum_squared = 0;
        for (double workload : workloads) {
            sum_squared += workload * workload;
        }
        
        if (sum_squared == 0) return 1.0;
        return (sum * sum) / (workloads.size() * sum_squared);
    }
    
    std::unordered_map<int, int> getAllocation() const {
        return allocation;
    }
    
    void setFairnessEpsilon(double epsilon) {
        fairness_epsilon = epsilon;
    }

    void printTotalWeightedCompletion(const std::string& phase) const {
        double totalWeightedCompletion = 0;
        for (const auto& task : tasks) {
            if (allocation.count(task.id) && allocation.at(task.id) != -1) {
                totalWeightedCompletion += task.story_points * task.finish_time;
            }
        }
        std::cout << "Total Weighted Completion after " << phase << ": "
                  << std::setprecision(2) << std::fixed << totalWeightedCompletion << std::endl;
    }
};

// Generate software development tasks
std::vector<Task> generateSoftwareTasks(int numTasks, int project_duration_days = 60) {
    std::vector<Task> tasks;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    // Time distributions (in hours)
    double max_hours = project_duration_days * 8; // 8 hours per day
    std::uniform_real_distribution<> startDist(0, max_hours - 8);
    
    // Task type probabilities
    std::vector<TaskType> taskTypes = {
        TaskType::BACKEND_DEVELOPMENT,
        TaskType::FRONTEND_DEVELOPMENT,
        TaskType::DATABASE_DESIGN,
        TaskType::API_DEVELOPMENT,
        TaskType::TESTING,
        TaskType::CODE_REVIEW,
        TaskType::DOCUMENTATION,
        TaskType::DEPLOYMENT,
        TaskType::BUG_FIX,
        TaskType::FEATURE_RESEARCH,
        TaskType::UI_UX_DESIGN,
        TaskType::SECURITY_AUDIT
    };
    
    std::uniform_int_distribution<> taskTypeDist(0, taskTypes.size() - 1);
    std::uniform_real_distribution<> storyPointsDist(1, 13); // Fibonacci-like story points
    std::uniform_int_distribution<> priorityDist(1, 5);
    std::uniform_real_distribution<> durationDist(4, 40); // 4-40 hours per task
    
    // Feature names
    std::vector<std::string> features = {
        "User Authentication", "Payment Processing", "Dashboard Analytics",
        "Notification System", "Search Functionality", "Data Export",
        "Mobile App", "Admin Panel", "Reporting Module", "Integration API",
        "Security Framework", "Performance Optimization", "User Profile",
        "Chat System", "File Upload", "Email Service", "Backup System",
        "Monitoring Dashboard", "Configuration Management", "Testing Suite"
    };
    
    // Sprint names
    std::vector<std::string> sprints = {
        "Sprint 1", "Sprint 2", "Sprint 3", "Sprint 4", "Sprint 5", "Sprint 6"
    };
    
    std::uniform_int_distribution<> featureDist(0, features.size() - 1);
    std::uniform_int_distribution<> sprintDist(0, sprints.size() - 1);
    
    // Common skills for tasks
    std::vector<std::string> skills = {
        "JavaScript", "Python", "Java", "C++", "React", "Angular", "Vue.js",
        "Node.js", "Django", "Spring", "PostgreSQL", "MongoDB", "Redis",
        "Docker", "Kubernetes", "AWS", "Git", "Testing", "Security", "UI/UX"
    };
    
    std::uniform_int_distribution<> skillDist(0, skills.size() - 1);
    std::uniform_real_distribution<> proficiencyDist(0.3, 1.0);
    
    for (int i = 1; i <= numTasks; i++) {
        double start = startDist(gen);
        double duration = durationDist(gen);
        double finish = start + duration;
        
        // Ensure finish time doesn't exceed project duration
        if (finish > max_hours) {
            finish = max_hours;
            start = std::max(0.0, finish - duration);
        }
        
        TaskType type = taskTypes[taskTypeDist(gen)];
        double points = storyPointsDist(gen);
        std::string feature = features[featureDist(gen)];
        int priority = priorityDist(gen);
        std::string sprint = sprints[sprintDist(gen)];
        
        Task task(i, start, finish, points, type, feature, priority, sprint);
        
        // Add random skills required for the task
        int num_skills = std::uniform_int_distribution<>(1, 3)(gen);
        for (int j = 0; j < num_skills; j++) {
            std::string skill = skills[skillDist(gen)];
            double required_proficiency = proficiencyDist(gen);
            task.addResource(skill, required_proficiency);
        }
        
        // Add dependencies (10% chance for each task to have 1-2 dependencies)
        if (i > 1 && std::uniform_real_distribution<>(0, 1)(gen) < 0.1) {
            int num_deps = std::uniform_int_distribution<>(1, 2)(gen);
            for (int j = 0; j < num_deps; j++) {
                int dep_id = std::uniform_int_distribution<>(1, i-1)(gen);
                task.addDependency(dep_id);
            }
        }
        
        tasks.push_back(task);
    }
    
    return tasks;
}

// Generate software developers
std::vector<Developer> generateSoftwareDevelopers(int numDevelopers) {
    std::vector<Developer> developers;
    std::mt19937 gen(123); // Fixed seed for reproducibility
    
    std::vector<DeveloperRole> roles = {
        DeveloperRole::SENIOR_BACKEND_DEV,
        DeveloperRole::SENIOR_FRONTEND_DEV,
        DeveloperRole::FULLSTACK_DEV,
        DeveloperRole::JUNIOR_DEV,
        DeveloperRole::QA_ENGINEER,
        DeveloperRole::DEVOPS_ENGINEER,
        DeveloperRole::UI_UX_DESIGNER,
        DeveloperRole::TECH_LEAD,
        DeveloperRole::SECURITY_SPECIALIST
    };
    
    std::vector<std::string> names = {
        "Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson",
        "Emma Brown", "Frank Miller", "Grace Lee", "Henry Garcia",
        "Iris Martinez", "Jack Anderson", "Kate Thompson", "Liam White",
        "Maya Patel", "Noah Kim", "Olivia Rodriguez", "Peter Chen",
        "Quinn Taylor", "Ruby Jackson", "Sam Williams", "Tina Lopez"
    };
    
    std::vector<std::string> teams = {
        "Core Platform", "Frontend Team", "Backend Team", "DevOps Team",
        "QA Team", "Security Team", "Mobile Team", "Data Team"
    };
    
    std::uniform_int_distribution<> roleDist(0, roles.size() - 1);
    std::uniform_int_distribution<> nameDist(0, names.size() - 1);
    std::uniform_int_distribution<> teamDist(0, teams.size() - 1);
    std::uniform_int_distribution<> experienceDist(1, 15);
    std::uniform_real_distribution<> capacityDist(6, 8); // 6-8 hours per day
    
    // Skills for different roles
    std::map<DeveloperRole, std::vector<std::string>> roleSkills = {
        {DeveloperRole::SENIOR_BACKEND_DEV, {"Java", "Python", "C++", "PostgreSQL", "MongoDB", "Docker", "AWS", "Spring", "Django"}},
        {DeveloperRole::SENIOR_FRONTEND_DEV, {"JavaScript", "React", "Angular", "Vue.js", "CSS", "HTML", "TypeScript"}},
        {DeveloperRole::FULLSTACK_DEV, {"JavaScript", "Python", "React", "Node.js", "PostgreSQL", "MongoDB", "Docker"}},
        {DeveloperRole::JUNIOR_DEV, {"JavaScript", "Python", "Git", "HTML", "CSS"}},
        {DeveloperRole::QA_ENGINEER, {"Testing", "Selenium", "Jest", "Python", "JavaScript"}},
        {DeveloperRole::DEVOPS_ENGINEER, {"Docker", "Kubernetes", "AWS", "Jenkins", "Terraform", "Linux"}},
        {DeveloperRole::UI_UX_DESIGNER, {"UI/UX", "Figma", "Adobe XD", "CSS", "HTML"}},
        {DeveloperRole::TECH_LEAD, {"Java", "Python", "JavaScript", "Architecture", "Management", "Code Review"}},
        {DeveloperRole::SECURITY_SPECIALIST, {"Security", "Penetration Testing", "Cryptography", "Python", "C++"}}
    };
    
    std::uniform_real_distribution<> skillLevelDist(0.4, 1.0);
    std::set<std::string> usedNames;
    
    for (int i = 1; i <= numDevelopers; i++) {
        DeveloperRole role = roles[roleDist(gen)];
        
        // Select unique name
        std::string name;
        do {
            name = names[nameDist(gen)];
        } while (usedNames.count(name) > 0 && usedNames.size() < names.size());
        usedNames.insert(name);
        
        std::string team = teams[teamDist(gen)];
        int experience = experienceDist(gen);
        double capacity = capacityDist(gen);
        
        // Adjust experience based on role
        if (role == DeveloperRole::JUNIOR_DEV) {
            experience = std::min(experience, 3);
        } else if (role == DeveloperRole::TECH_LEAD || 
                   role == DeveloperRole::SENIOR_BACKEND_DEV || 
                   role == DeveloperRole::SENIOR_FRONTEND_DEV) {
            experience = std::max(experience, 5);
        }
        
        Developer dev(i, role, name, experience, team, capacity);
        
        // Add skills based on role
        if (roleSkills.count(role) > 0) {
            for (const std::string& skill : roleSkills[role]) {
                double level = skillLevelDist(gen);
                // Senior roles get higher skill levels
                if (role == DeveloperRole::SENIOR_BACKEND_DEV || 
                    role == DeveloperRole::SENIOR_FRONTEND_DEV ||
                    role == DeveloperRole::TECH_LEAD) {
                    level = std::max(level, 0.7);
                }
                dev.addSkill(skill, level);
            }
        }
        
        developers.push_back(dev);
    }
    
    return developers;
}

// Main function to demonstrate the system
int main() {
    std::cout << "=== Software Development Task Allocation System ===" << std::endl;
    std::cout << "Hybrid Algorithm: Priority + Dynamic Programming + Load Balancing" << std::endl;
    std::cout << std::endl;
    
    // Generate test data
    int numTasks = 50;
    int numDevelopers = 8;
    
    std::cout << "Generating " << numTasks << " tasks and " << numDevelopers << " developers..." << std::endl;
    
    std::vector<Task> tasks = generateSoftwareTasks(numTasks);
    std::vector<Developer> developers = generateSoftwareDevelopers(numDevelopers);
    
    // Print sample tasks and developers
    std::cout << "\n=== Sample Tasks ===" << std::endl;
    for (int i = 0; i < std::min(5, (int)tasks.size()); i++) {
        tasks[i].print();
    }
    
    std::cout << "\n=== Sample Developers ===" << std::endl;
    for (int i = 0; i < std::min(3, (int)developers.size()); i++) {
        developers[i].print();
    }
    
    // Run the hybrid algorithm
    std::cout << "\n=== Running Hybrid Allocation Algorithm ===" << std::endl;
    SoftwareDevHybridAlgorithm algorithm(tasks, developers, 0.15);
    algorithm.run();
    
    // Print final allocation details
    std::cout << "\n=== Final Task Allocation ===" << std::endl;
    auto allocation = algorithm.getAllocation();
    
    // Group tasks by developer
    std::map<int, std::vector<int>> developerTasks;
    for (const auto& [taskId, developerId] : allocation) {
        if (developerId != -1) {
            developerTasks[developerId].push_back(taskId);
        }
    }
    
    // Print allocation by developer
    for (const auto& dev : developers) {
        std::cout << "\n" << developerRoleToString(dev.role) << " " << dev.name << ":" << std::endl;
        if (developerTasks.count(dev.id) > 0) {
            double totalPoints = 0;
            for (int taskId : developerTasks[dev.id]) {
                const Task& task = tasks[taskId - 1]; // Convert ID to index
                std::cout << "  - Task " << taskId << " (" << taskTypeToString(task.type) 
                         << "): " << task.story_points << " points, " << task.feature_name << std::endl;
                totalPoints += task.story_points;
            }
            std::cout << "  Total: " << developerTasks[dev.id].size() << " tasks, " 
                     << totalPoints << " story points" << std::endl;
        } else {
            std::cout << "  No tasks assigned" << std::endl;
        }
    }
    
    // Print unassigned tasks
    std::vector<int> unassignedTasks;
    for (const auto& task : tasks) {
        if (allocation[task.id] == -1) {
            unassignedTasks.push_back(task.id);
        }
    }
    
    if (!unassignedTasks.empty()) {
        std::cout << "\n=== Unassigned Tasks ===" << std::endl;
        for (int taskId : unassignedTasks) {
            const Task& task = tasks[taskId - 1];
            std::cout << "Task " << taskId << " (" << taskTypeToString(task.type) 
                     << "): " << task.story_points << " points, " << task.feature_name << std::endl;
        }
    }
    
    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    double totalStoryPoints = 0;
    double assignedStoryPoints = 0;
    
    for (const auto& task : tasks) {
        totalStoryPoints += task.story_points;
        if (allocation[task.id] != -1) {
            assignedStoryPoints += task.story_points;
        }
    }
    
    double assignmentRate = (assignedStoryPoints / totalStoryPoints) * 100;
    std::cout << "Assignment Rate: " << std::setprecision(1) << std::fixed 
             << assignmentRate << "%" << std::endl;
    std::cout << "Total Story Points: " << totalStoryPoints << std::endl;
    std::cout << "Assigned Story Points: " << assignedStoryPoints << std::endl;
    
    return 0;
}