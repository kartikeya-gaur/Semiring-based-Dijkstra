#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <iomanip>

// ── Data Structures ───────────────────────────────────────────────────

struct Edge {
    int to;
    std::vector<double> costs; // e.g., {time, transfers, cost}
};

struct Label {
    int node;
    std::vector<double> costs;
    
    // Lexicographical comparison for the Priority Queue
    bool operator>(const Label& other) const {
        for (size_t i = 0; i < costs.size(); ++i) {
            if (costs[i] != other.costs[i]) {
                return costs[i] > other.costs[i];
            }
        }
        return false;
    }
};

// ── The Pareto Dominance Check ───────────────────────────────────────
// Returns true if 'a' dominates 'b' (a is <= in all, and < in at least one)
bool dominates(const std::vector<double>& a, const std::vector<double>& b) {
    bool strictly_better = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i]) return false; // 'a' is worse in at least one criterion
        if (a[i] < b[i]) strictly_better = true;
    }
    return strictly_better;
}

// ── The Baseline Algorithm ───────────────────────────────────────────
void paretoDijkstra(int num_nodes, const std::vector<std::vector<Edge>>& graph, int src, int num_criteria) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // The "Pareto Set" for each node. This is the root of the exponential trap.
    // Instead of one scalar per node, we store every non-dominated path.
    std::vector<std::vector<std::vector<double>>> pareto_sets(num_nodes);
    
    std::priority_queue<Label, std::vector<Label>, std::greater<Label>> pq;

    std::vector<double> init_costs(num_criteria, 0.0);
    pq.push({src, init_costs});
    pareto_sets[src].push_back(init_costs);

    // Metrics for Logging
    long long total_queue_insertions = 1;
    long long total_dominance_checks = 0;

    while (!pq.empty()) {
        Label current = pq.top();
        pq.pop();

        // 1. Is this label still non-dominated, or did a better one arrive while it was in the queue?
        bool is_dominated = false;
        for (const auto& existing_cost : pareto_sets[current.node]) {
            total_dominance_checks++;
            if (dominates(existing_cost, current.costs)) {
                is_dominated = true;
                break;
            }
        }
        if (is_dominated) continue;

        // 2. Extend to neighbors
        for (const Edge& edge : graph[current.node]) {
            std::vector<double> new_costs(num_criteria);
            for (int i = 0; i < num_criteria; ++i) {
                new_costs[i] = current.costs[i] + edge.costs[i];
            }

            // 3. The Pareto Trap: Compare against ALL existing paths at the destination
            bool candidate_dominated = false;
            std::vector<std::vector<double>> updated_pareto_set;
            
            for (const auto& existing_cost : pareto_sets[edge.to]) {
                total_dominance_checks++;
                if (dominates(existing_cost, new_costs)) {
                    candidate_dominated = true;
                    break; 
                }
                // If the new path dominates an old one, we don't keep the old one
                if (!dominates(new_costs, existing_cost)) {
                    updated_pareto_set.push_back(existing_cost);
                }
            }

            if (!candidate_dominated) {
                updated_pareto_set.push_back(new_costs);
                pareto_sets[edge.to] = std::move(updated_pareto_set);
                
                pq.push({edge.to, new_costs});
                total_queue_insertions++; // N(q) inflates here!
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Calculate maximum Pareto set size (shows memory bloat)
    size_t max_pareto_size = 0;
    for (const auto& pset : pareto_sets) {
        if (pset.size() > max_pareto_size) max_pareto_size = pset.size();
    }

    // ── Log Output for Python Plotting ───────────────────────────────
    std::cout << "=== PARETO DIJKSTRA METRICS (Criteria = " << num_criteria << ") ===\n";
    std::cout << "Latency (ms):           " << std::fixed << std::setprecision(2) << latency_ms << "\n";
    std::cout << "Total Queue Insertions: " << total_queue_insertions << "\n";
    std::cout << "Dominance Checks:       " << total_dominance_checks << "\n";
    std::cout << "Max Pareto Set Size:    " << max_pareto_size << "\n";
    std::cout << "--------------------------------------------------\n";
}

// ── Dummy Graph Generator for Testing ────────────────────────────────
int main() {
    int num_nodes = 5000;
    int num_criteria = 3; // Try changing this to 1, 2, 4, 5
    
    std::vector<std::vector<Edge>> graph(num_nodes);
    
    // Create a random, dense lattice to force multi-criteria trade-offs
    srand(42); 
    for (int i = 0; i < num_nodes - 1; ++i) {
        for (int j = 1; j <= 3 && i + j < num_nodes; ++j) { // 3 forward edges per node
            std::vector<double> costs(num_criteria);
            for (int c = 0; c < num_criteria; ++c) {
                // Random costs between 1 and 10 to ensure conflicting trade-offs
                costs[c] = (rand() % 10) + 1.0; 
            }
            graph[i].push_back({i + j, costs});
        }
    }

    paretoDijkstra(num_nodes, graph, 0, num_criteria);

    return 0;
}