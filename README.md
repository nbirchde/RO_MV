# Instructions for the project, specifically meant for Claude 3.7 Sonnet to follow.

# INFO-H-3000 : Challenge Flowshop Bi-objectif

## Introduction

To familiarize yourself with multi-objective optimization concepts, you are tasked with finding the best approximation of the Pareto Optimal Frontier for the bi-objective Flowshop problem. In this problem, a set of parts must pass through a predefined sequence of machines to be processed. For this challenge, these parts are described in the file "instance.csv". Each line of this file represents a part. The first two columns of line $i$ represent the priority $p_{i}$ and the deadline $d_{i}$ of part $i$, respectively. These data will be used to calculate the "Total weighted tardiness" of the parts (see below). The next 10 columns $\left(T_{i}^{j}, \forall j=1, \ldots 10\right)$ represent the time required for part $i$ to be processed by machine $j$.

The flowshop problem is defined as follows:

- The order of the machines is fixed; each part must first be processed on machine $j$ before moving to machine $j+1$.
- A machine can process only one part at a time. For each machine $j$, you must wait for the first part to finish processing before starting the next part. However, the next part can be processed by machine $j-1$ while the first part is being processed on machine $j$.
- Once a part passes through the first machine, it must pass through all subsequent machines as soon as possible (i.e., as soon as they have finished with the previous part).

You are asked to find a set of solutions (sequences of parts, i.e., permutations) that minimize the "Makespan" and the total weighted tardiness. Your set of solutions must form a Pareto Optimal Frontier. The Makespan represents the time required for all parts to be processed (i.e., the time at which the last part exits the last machine). The total weighted tardiness represents the sum of the delays of each part relative to their deadlines (weighted by their priorities). For example, a part $i$ that finishes at time $F_{i}>d_{i}$ would contribute to the total tardiness by the amount $\left(F_{i}-d_{i}\right) * p_{i}$.

## Evaluation of Your Frontier

Your solution will be evaluated using three weights that will be randomly drawn after the challenge:

$$
w_{1}=? ? ; w_{2}=? ? ; w_{3}=? ? \quad w_{1}, w_{2}, w_{3} \in] 0,1[.
$$

These weights will be used to perform a weighted sum for each of your solutions. The group's score will correspond to the score of your best solution:

$$
\text { Score }=\min _{P O} \sum_{i=1}^{3}\left(w_{i} \frac{C_{\max , P O}}{\max \left(C_{\max , P O^{*}}\right)}+\left(1-w_{i}\right) \frac{T_{t o t, P O}}{\max \left(T_{t o t, P O^{*}}\right)}\right)
$$

where:

- $P O$ corresponds to your Pareto Optimal solutions;
- $P O^{*}$ corresponds to our reference Pareto Optimal solutions;
- $C_{\max , X}$ corresponds to the makespan of solution X;
- $T_{t o t, X}$ corresponds to the total tardiness of solution X.

Each solution in your Pareto Optimal Frontier is better than the others for a particular set of weights. Submitting a Pareto Optimal Frontier composed of a large number of solutions increases your probability of having a solution suited to the randomly drawn weights.

# Instructions

Each group must submit a ".csv" file containing the optimal solutions they have found, following these conditions:

- You must implement your algorithm in the programming language of your choice. However, you cannot use external libraries. You must create your own heuristic.
- The file name must be composed as follows:
  <student1>_<student2>.csv
  where <student1> corresponds to the last name followed by the first name of the first student (and similarly for <student2>). <student1> is the student whose last name comes first alphabetically.
- Each line of the file corresponds to a solution, i.e., a permutation of integers between 0 and 199 (not between 1 and 200). The integers are separated by commas.
- The solutions must form a Pareto Optimal Frontier.
- Your file must be submitted to the UV before Wednesday, March 16 (today) at 17:45. To submit your file, you must register in a group on the UV.

Any submission not meeting these conditions will not be considered. The winning group will be required to submit their code and present their approach during the next lab session.
```

The workspace also contains a pre-established structure we can build on. Note the other markdown file: Approach.md contains a very detaild report on how to implement a solution to the given problem. Always consult that file.

the preestablished framework contains the following:

# Pareto Optimization Framework

A Swift framework for multi-objective optimization using Pareto optimality, with GPU acceleration support via Metal.

## Features

- Multi-objective optimization support
- GPU-accelerated dominance computation using Metal
- Generic Solution protocol for custom optimization problems
- Built-in support for numeric and categorical optimization criteria
- Efficient Pareto front calculation

## Requirements

- macOS with Metal support
- Swift 5.0 or later
- Xcode 12.0 or later

## Installation

Add this package to your Xcode project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "your-github-url-here", from: "1.0.0")
]
```

## Usage

```swift
import ParetoOptimization

// Create some solutions
let solutions = [DemoChromosome.random(), DemoChromosome.random()]

// Calculate Pareto front
let paretoFront = ParetoFront(solutions: solutions)

// Use GPU acceleration
let accelerator = MetalAccelerator.shared
let dominanceMatrix = accelerator.checkDominanceRelationships(solutions: solutions)
```

## License

MIT License