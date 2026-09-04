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

export const timesMap = <T>(count: number, fn: (index: number) => T): T[] => {
  const results: T[] = []

  for (let i = 0; i < count; i++) results[i] = fn(i)

  return results
}
