import Foundation
import simd

/// Protocol defining a multi-objective solution with arbitrary criteria
public protocol Solution: Hashable {
    /// The criteria values for this solution
    var criteria: [CriterionValue] { get }
    
    /// Returns true if this solution dominates another solution
    /// A solution dominates another if it is at least as good in all criteria
    /// and strictly better in at least one criterion
    func dominates(_ other: Self) -> Bool
}

/// Default implementation for dominance check
public extension Solution {
    func dominates(_ other: Self) -> Bool {
        // Must be at least as good in all dimensions
        guard criteria.count == other.criteria.count else {
            fatalError("Cannot compare solutions with different number of criteria")
        }
        
        var hasStrictlyBetter = false
        
        for (selfValue, otherValue) in zip(criteria, other.criteria) {
            switch selfValue.compare(to: otherValue) {
            case .worse:
                return false // If we're worse in any dimension, we don't dominate
            case .better:
                hasStrictlyBetter = true // Need at least one dimension where we're strictly better
            case .equal, .incomparable:
                continue // Being equal is okay for dominance
            }
        }
        
        // We dominate if we're at least as good in all dimensions and strictly better in at least one
        return hasStrictlyBetter
    }
}

/// Enum representing comparison between two criterion values
public enum ComparisonResult {
    case better    // This solution is better
    case equal     // Solutions are equal for this criterion
    case worse     // This solution is worse
    case incomparable // Criteria cannot be compared (e.g., different units or categorical values)
}

/// Protocol defining a value for a specific criterion
public protocol CriterionValue {
    /// Compare this criterion value to another, returning the comparison result
    func compare(to other: CriterionValue) -> ComparisonResult
}

/// Numeric criterion value with automatic comparison (lower is better by default)
public struct NumericCriterion: CriterionValue {
    public let value: Double
    public let lowerIsBetter: Bool
    
    public init(_ value: Double, lowerIsBetter: Bool = true) {
        self.value = value
        self.lowerIsBetter = lowerIsBetter
    }
    
    public func compare(to other: CriterionValue) -> ComparisonResult {
        guard let other = other as? NumericCriterion else {
            return .incomparable
        }
        
        // Both criteria must have the same direction for comparison
        guard lowerIsBetter == other.lowerIsBetter else {
            return .incomparable
        }
        
        if value == other.value {
            return .equal
        }
        
        // Compare based on whether lower is better or not
        if lowerIsBetter {
            return value < other.value ? .better : .worse
        } else {
            return value > other.value ? .better : .worse
        }
    }
}

/// Categorical criterion value for non-numeric criteria
public struct CategoricalCriterion: CriterionValue {
    public let value: String
    public let preferenceOrder: [String] // From most preferred to least preferred
    
    public init(_ value: String, preferenceOrder: [String]) {
        self.value = value
        self.preferenceOrder = preferenceOrder
    }
    
    public func compare(to other: CriterionValue) -> ComparisonResult {
        guard let other = other as? CategoricalCriterion else {
            return .incomparable
        }
        
        // Both criteria must have the same preference order for comparison
        guard preferenceOrder == other.preferenceOrder else {
            return .incomparable
        }
        
        guard let selfIndex = preferenceOrder.firstIndex(of: value),
              let otherIndex = preferenceOrder.firstIndex(of: other.value) else {
            return .incomparable
        }
        
        if selfIndex == otherIndex {
            return .equal
        }
        
        // Lower index means more preferred in the preference order
        return selfIndex < otherIndex ? .better : .worse
    }
}