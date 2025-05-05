#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace chrono;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {} 

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected
    }

    void bfs(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front(); q.pop();
            cout << node << " ";
            for (int neigh : adj[node]) {
                if (!visited[neigh]) {
                    visited[neigh] = true;
                    q.push(neigh);
                }
            }
        }
    }

    void parallelBfs(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int size = q.size();
            vector<int> level;
            for (int i = 0; i < size; ++i) {
                int node = q.front(); q.pop();
                cout << node << " ";
                level.push_back(node);
            }

            #pragma omp parallel for
            for (int i = 0; i < level.size(); ++i) {
                int node = level[i];
                for (int neigh : adj[node]) {
                    #pragma omp critical
                    if (!visited[neigh]) {
                        visited[neigh] = true;
                        q.push(neigh);
                    }
                }
            }
        }
    }

    void dfs(int node, vector<bool>& visited) {
        visited[node] = true;
        cout << node << " ";
        for (int neigh : adj[node])
            if (!visited[neigh])
                dfs(neigh, visited);
    }

    void parallelDfs(int node, vector<bool>& visited) {
        #pragma omp critical
        if (visited[node]) return;
        visited[node] = true;
        cout << node << " ";

        for (int neigh : adj[node]) {
            #pragma omp task
            parallelDfs(neigh, visited);
        }
    }
};

int main() {
    int V, E, u, v, start;
    cout << "Enter number of vertices and edges: ";
    cin >> V >> E;
    Graph g(V);

    cout << "Enter " << E << " edges (u v):\n";
    for (int i = 0; i < E; i++) {
        cin >> u >> v;
        g.addEdge(u, v);
    }

    cout << "Enter starting node: ";
    cin >> start;

    // BFS
    cout << "\nSequential BFS: ";
    auto t1 = high_resolution_clock::now();
    g.bfs(start);
    auto t2 = high_resolution_clock::now();
    cout << "\nTime: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns\n";

    cout << "\nParallel BFS: ";
    t1 = high_resolution_clock::now();
    g.parallelBfs(start);
    t2 = high_resolution_clock::now();
    cout << "\nTime: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns\n";

    // DFS
    cout << "\nSequential DFS: ";
    vector<bool> vis1(V, false);
    t1 = high_resolution_clock::now();
    g.dfs(start, vis1);
    t2 = high_resolution_clock::now();
    cout << "\nTime: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns\n";

    cout << "\nParallel DFS: ";
    vector<bool> vis2(V, false);
    t1 = high_resolution_clock::now();
    #pragma omp parallel
    {
        #pragma omp single
        g.parallelDfs(start, vis2);
    }
    t2 = high_resolution_clock::now();
    cout << "\nTime: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns\n";

    return 0;
}
