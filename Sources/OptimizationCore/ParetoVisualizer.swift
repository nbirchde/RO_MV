import Foundation
import SwiftUI
import Charts

/// A view that displays a Pareto front visualization
public struct ParetoVisualizerView: View {
    private var solutions: [DataPoint]
    private var title: String
    private var xLabel: String
    private var yLabel: String
    
    /// Initialize a visualizer view for the Pareto front
    /// - Parameters:
    ///   - solutions: Data points to visualize
    ///   - title: Chart title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    public init(solutions: [DataPoint], title: String, xLabel: String, yLabel: String) {
        self.solutions = solutions
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
    }
    
    /// Create the visualization for this Pareto front
    public var body: some View {
        VStack {
            Text(title)
                .font(.title)
                .padding()
            
            if solutions.isEmpty {
                Text("No data available to visualize")
                    .font(.headline)
                    .foregroundColor(.gray)
                    .padding()
            } else {
                Chart {
                    ForEach(solutions) { point in
                        PointMark(
                            x: .value(xLabel, point.x),
                            y: .value(yLabel, point.y)
                        )
                        .foregroundStyle(point.color)
                        .annotation(position: .top) {
                            if !point.label.isEmpty {
                                Text(point.label)
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .frame(minHeight: 400)
                .padding()
                .chartXAxisLabel(xLabel)
                .chartYAxisLabel(yLabel)
            }
        }
        .frame(minWidth: 600, minHeight: 500)
    }
}

/// A struct to display a comparison of multiple Pareto fronts
public struct ParetoComparisonView: View {
    private var dataSets: [(name: String, points: [DataPoint])]
    private var title: String
    private var xLabel: String
    private var yLabel: String
    
    /// Initialize a comparison view for multiple Pareto fronts
    /// - Parameters:
    ///   - dataSets: Named collections of data points to visualize
    ///   - title: Chart title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    public init(
        dataSets: [(name: String, points: [DataPoint])],
        title: String,
        xLabel: String,
        yLabel: String
    ) {
        self.dataSets = dataSets
        self.title = title
        self.xLabel = xLabel
        self.yLabel = yLabel
    }
    
    /// Create the visualization comparing multiple Pareto fronts
    public var body: some View {
        VStack {
            Text(title)
                .font(.title)
                .padding()
            
            if dataSets.isEmpty {
                Text("No data available to visualize")
                    .font(.headline)
                    .foregroundColor(.gray)
                    .padding()
            } else {
                Chart {
                    ForEach(dataSets.indices, id: \.self) { i in
                        ForEach(dataSets[i].points) { point in
                            PointMark(
                                x: .value(xLabel, point.x),
                                y: .value(yLabel, point.y)
                            )
                            .foregroundStyle(by: .value("Method", dataSets[i].name))
                        }
                    }
                }
                .frame(minHeight: 400)
                .padding()
                .chartXAxisLabel(xLabel)
                .chartYAxisLabel(yLabel)
            }
            
            // Legend for the comparison
            HStack {
                ForEach(dataSets.indices, id: \.self) { i in
                    if !dataSets[i].points.isEmpty {
                        Label(dataSets[i].name, systemImage: "circle.fill")
                            .foregroundColor(getColorForIndex(i))
                            .padding(.horizontal)
                    }
                }
            }
            .padding()
        }
        .frame(minWidth: 600, minHeight: 500)
    }
    
    /// Get a color for the given index
    private func getColorForIndex(_ index: Int) -> Color {
        let colors: [Color] = [.blue, .red, .green, .orange, .purple, .pink]
        return index < colors.count ? colors[index] : .gray
    }
}

/// A utility class that creates SwiftUI views for Pareto front visualization
public struct ParetoVisualizer {
    /// Create a visualization view for a single Pareto front
    /// - Parameters:
    ///   - solutions: The solutions to visualize
    ///   - criterionX: Index of the criterion to use for X axis
    ///   - criterionY: Index of the criterion to use for Y axis
    ///   - title: Chart title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    /// - Returns: A SwiftUI view that visualizes the Pareto front
    public static func createVisualization<S: Solution>(
        solutions: [S],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) -> some View {
        // Extract data points from the solutions
        let points = extractDataPoints(
            from: solutions,
            criterionX: criterionX,
            criterionY: criterionY
        )
        
        // Create the visualization view
        return ParetoVisualizerView(
            solutions: points,
            title: title,
            xLabel: xLabel ?? "Criterion \(criterionX)",
            yLabel: yLabel ?? "Criterion \(criterionY)"
        )
    }
    
    /// Create a visualization view comparing multiple Pareto fronts
    /// - Parameters:
    ///   - solutions: Dictionary mapping method names to solution arrays
    ///   - criterionX: Index of the criterion to use for X axis
    ///   - criterionY: Index of the criterion to use for Y axis
    ///   - title: Chart title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    /// - Returns: A SwiftUI view that visualizes multiple Pareto fronts
    public static func createComparisonVisualization<S: Solution>(
        solutions: [String: [S]],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier Comparison",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) -> some View {
        // Prepare the dataset for visualization
        var dataSets: [(name: String, points: [DataPoint])] = []
        
        // Process each named set of solutions
        for (name, solutionSet) in solutions {
            let points = extractDataPoints(
                from: solutionSet,
                criterionX: criterionX,
                criterionY: criterionY
            )
            dataSets.append((name: name, points: points))
        }
        
        // Create the comparison view
        return ParetoComparisonView(
            dataSets: dataSets,
            title: title,
            xLabel: xLabel ?? "Criterion \(criterionX)",
            yLabel: yLabel ?? "Criterion \(criterionY)"
        )
    }
    
    /// Extract data points from solution criteria
    private static func extractDataPoints<S: Solution>(
        from solutions: [S],
        criterionX: Int,
        criterionY: Int,
        criterionZ: Int? = nil
    ) -> [DataPoint] {
        var points: [DataPoint] = []
        
        for (_, solution) in solutions.enumerated() {
            guard criterionX < solution.criteria.count, 
                  criterionY < solution.criteria.count else {
                continue
            }
            
            if let xCriterion = solution.criteria[criterionX] as? NumericCriterion,
               let yCriterion = solution.criteria[criterionY] as? NumericCriterion {
                
                // Handle criterion direction (lower/higher is better)
                let xValue = xCriterion.lowerIsBetter ? 
                    xCriterion.value : -xCriterion.value
                let yValue = yCriterion.lowerIsBetter ? 
                    yCriterion.value : -yCriterion.value
                
                // Extract Z value if specified
                var zValue: Double? = nil
                if let criterionZ = criterionZ, 
                   criterionZ < solution.criteria.count,
                   let zCriterion = solution.criteria[criterionZ] as? NumericCriterion {
                    zValue = zCriterion.lowerIsBetter ?
                        zCriterion.value : -zCriterion.value
                }
                
                // Create data point
                points.append(DataPoint(
                    x: xValue, 
                    y: yValue,
                    z: zValue,
                    color: .blue,
                    label: ""
                ))
            }
        }
        
        return points
    }
    
    /// Launch an interactive visualization window
    /// - Parameters:
    ///   - solutions: Solutions to visualize
    ///   - criterionX: X-axis criterion index
    ///   - criterionY: Y-axis criterion index
    ///   - title: Window title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    public static func showVisualization<S: Solution>(
        solutions: [S],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) {
        let view = createVisualization(
            solutions: solutions,
            criterionX: criterionX,
            criterionY: criterionY,
            title: title,
            xLabel: xLabel,
            yLabel: yLabel
        )
        
        // Create a window to display the plot
        let hostingController = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hostingController)
        window.title = title
        window.setContentSize(NSSize(width: 800, height: 600))
        window.center()
        window.makeKeyAndOrderFront(nil)
        
        // Start the app main loop if it's not already running
        if !NSApp.isRunning {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.run()
        }
    }
    
    /// Launch an interactive comparison visualization window
    /// - Parameters:
    ///   - solutions: Dictionary mapping method names to solution arrays
    ///   - criterionX: X-axis criterion index
    ///   - criterionY: Y-axis criterion index
    ///   - title: Window title
    ///   - xLabel: X-axis label
    ///   - yLabel: Y-axis label
    public static func showComparison<S: Solution>(
        solutions: [String: [S]],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier Comparison",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) {
        let view = createComparisonVisualization(
            solutions: solutions,
            criterionX: criterionX,
            criterionY: criterionY,
            title: title,
            xLabel: xLabel,
            yLabel: yLabel
        )
        
        // Create a window to display the plot
        let hostingController = NSHostingController(rootView: view)
        let window = NSWindow(contentViewController: hostingController)
        window.title = title
        window.setContentSize(NSSize(width: 800, height: 600))
        window.center()
        window.makeKeyAndOrderFront(nil)
        
        // Start the app main loop if it's not already running
        if !NSApp.isRunning {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.run()
        }
    }
}