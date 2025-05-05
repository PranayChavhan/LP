#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Sequential Bubble Sort
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j)
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}

// Parallel Bubble Sort
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

// Sequential Merge Sort
void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; ++i) L[i] = arr[l + i];
    for (int i = 0; i < n2; ++i) R[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// Parallel Merge Sort
void parallelMergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);

            #pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

// Utility to generate random array
vector<int> generateRandomArray(int size, int maxVal = 10000) {
    vector<int> arr(size);
    for (int& x : arr) x = rand() % maxVal;
    return arr;
}

void printArray(const vector<int>& arr) {
    for (int x : arr) cout << x << " ";
    cout << "\n";
}

int main() {
    srand(time(0));
    int n;
    cout << "Enter array size: ";
    cin >> n;

    vector<int> original = generateRandomArray(n);

    // Bubble Sort
    vector<int> arr1 = original;
    auto t1 = high_resolution_clock::now();
    bubbleSort(arr1);
    auto t2 = high_resolution_clock::now();
    cout << "\nSequential Bubble Sort Time: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns";

    arr1 = original;
    t1 = high_resolution_clock::now();
    parallelBubbleSort(arr1);
    t2 = high_resolution_clock::now();
    cout << "\nParallel Bubble Sort Time: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns";

    // Merge Sort
    vector<int> arr2 = original;
    t1 = high_resolution_clock::now();
    mergeSort(arr2, 0, n - 1);
    t2 = high_resolution_clock::now();
    cout << "\n\nSequential Merge Sort Time: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns";

    arr2 = original;
    t1 = high_resolution_clock::now();
    parallelMergeSort(arr2, 0, n - 1);
    t2 = high_resolution_clock::now();
    cout << "\nParallel Merge Sort Time: " << duration_cast<nanoseconds>(t2 - t1).count() << " ns\n";

    return 0;
}
