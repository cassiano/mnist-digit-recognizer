export const map = (
  value: number,
  lower: number,
  higher: number,
  projectedLower: number,
  projectedUpper: number,
  withinBounds = false,
) => {
  if (withinBounds) {
    if (lower <= higher) {
      if (value <= lower) return projectedLower
      else if (value >= higher) return projectedUpper
    } else if (value >= lower) {
      return projectedLower
    } else if (value <= higher) {
      return projectedUpper
    }
  }

  return (
    ((value - lower) / (higher - lower)) * (projectedUpper - projectedLower) +
    projectedLower
  )
}
