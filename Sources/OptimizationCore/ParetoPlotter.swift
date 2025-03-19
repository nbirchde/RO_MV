import Foundation
import SwiftUI
import Charts

/// A data point for the Pareto frontier
public struct DataPoint: Identifiable {
    public let id = UUID()
    public let x: Double
    public let y: Double
    public let z: Double?
    public let color: Color
    public let label: String
    
    public init(x: Double, y: Double, z: Double? = nil, color: Color = .blue, label: String = "") {
        self.x = x
        self.y = y
        self.z = z
        self.color = color
        self.label = label
    }
}

/// A utility class for plotting Pareto frontiers
public class ParetoPlotter {
    /// Generate a 2D plot of the Pareto frontier for two selected criteria
    /// - Parameters:
    ///   - solutions: Array of solutions to plot
    ///   - criterionX: Index of the criterion to use for X axis
    ///   - criterionY: Index of the criterion to use for Y axis
    ///   - title: Plot title
    ///   - xLabel: X axis label
    ///   - yLabel: Y axis label
    public static func plot2D<S: Solution>(
        solutions: [S],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) -> some View {
        // Create data points for the chart
        let points = extractDataPoints(
            from: solutions,
            criterionX: criterionX,
            criterionY: criterionY
        )
        
        return VStack {
            Text(title)
                .font(.title)
                .padding(.bottom, 8)
            
            if points.isEmpty {
                Text("No data available to plot")
                    .italic()
                    .foregroundColor(.gray)
                    .padding()
            } else {
                Chart {
                    ForEach(points) { point in
                        PointMark(
                            x: .value(xLabel ?? "Criterion \(criterionX)", point.x),
                            y: .value(yLabel ?? "Criterion \(criterionY)", point.y)
                        )
                        .foregroundStyle(point.color)
                        .symbolSize(CGFloat(30))
                    }
                }
                .frame(minHeight: 300)
                .padding()
            }
        }
    }
    
    /// Extract data points from solution criteria
    private static func extractDataPoints<S: Solution>(
        from solutions: [S],
        criterionX: Int,
        criterionY: Int,
        criterionZ: Int? = nil,
        groupKey: KeyPath<S, String>? = nil
    ) -> [DataPoint] {
        var points: [DataPoint] = []
        
        for (i, solution) in solutions.enumerated() {
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
                    label: "Solution \(i)"
                ))
            }
        }
        
        return points
    }
    
    /// Extract data points from multiple solution sets
    private static func extractMultiGroupDataPoints<S: Solution>(
        from solutionGroups: [String: [S]],
        criterionX: Int,
        criterionY: Int
    ) -> [DataPoint] {
        // Define a set of colors for different groups
        let colors: [Color] = [.blue, .red, .green, .orange, .purple, .pink]
        
        var allPoints: [DataPoint] = []
        
        // Process each group of solutions
        for (i, (groupName, solutions)) in solutionGroups.enumerated() {
            let color = colors[i % colors.count]
            
            // Extract points from this group
            for (j, solution) in solutions.enumerated() {
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
                    
                    // Create data point
                    allPoints.append(DataPoint(
                        x: xValue, 
                        y: yValue,
                        color: color,
                        label: "\(groupName) \(j)"
                    ))
                }
            }
        }
        
        return allPoints
    }
    
    /// Generate a plot comparing multiple Pareto frontiers
    public static func comparePareto<S: Solution>(
        solutions: [String: [S]],
        criterionX: Int,
        criterionY: Int,
        title: String = "Pareto Frontier Comparison",
        xLabel: String? = nil,
        yLabel: String? = nil
    ) -> some View {
        // Extract data points from all solution groups
        let points = extractMultiGroupDataPoints(
            from: solutions,
            criterionX: criterionX,
            criterionY: criterionY
        )
        
        // Group data points by color (which represents the solution group)
        let groupedPoints = Dictionary(grouping: points) { $0.color }
        
        // Create a chart with multiple series
        return VStack {
            Text(title)
                .font(.title)
                .padding(.bottom, 8)
            
            if points.isEmpty {
                Text("No data available to plot")
                    .italic()
                    .foregroundColor(.gray)
                    .padding()
            } else {
                Chart {
                    ForEach(Array(solutions.keys.enumerated()), id: \.element) { index, key in
                        // Only include this series if we have points for it
                        if let seriesPoints = groupedPoints[index < [Color.blue, .red, .green, .orange, .purple].count ? 
                                                          [.blue, .red, .green, .orange, .purple][index] : .gray] {
                            ForEach(seriesPoints) { point in
                                PointMark(
                                    x: .value(xLabel ?? "Criterion \(criterionX)", point.x),
                                    y: .value(yLabel ?? "Criterion \(criterionY)", point.y)
                                )
                                .foregroundStyle(point.color)
                                .symbolSize(CGFloat(30))
                            }
                            .foregroundStyle(by: .value("Method", key))
                        }
                    }
                }
                .frame(minHeight: 300)
                .padding()
            }
        }
    }
}