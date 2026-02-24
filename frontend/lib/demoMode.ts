/**
 * Demo mode constants and helpers.
 *
 * DEMO_MODE (build-time): whether this is a demo deployment (login page shown).
 * getUserTier / isDemoUser (runtime): resolved from the login response, stored
 * in localStorage. Full-tier users see the full UI; demo-tier users see
 * restricted UI with frame limiting and blocked writes.
 */

export const DEMO_MODE =
  process.env.NEXT_PUBLIC_DEMO_MODE === 'true';

export const DEMO_TOOLTIP =
  'This feature is disabled in demo mode.';

/** NE Atlantic + Mediterranean bounding box for map lock.
 *  Matches backend weather viewport: 30N-60N, 30W-40E. */
export const DEMO_BOUNDS: [[number, number], [number, number]] = [
  [30, -30],
  [60, 40],
];

// ============================================================================
// Runtime tier helpers
// ============================================================================

export type UserTier = 'demo' | 'full' | null;

const TIER_KEY = 'windmar_user_tier';

export function setUserTier(tier: 'demo' | 'full'): void {
  if (typeof window !== 'undefined') {
    localStorage.setItem(TIER_KEY, tier);
  }
}

export function getUserTier(): UserTier {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(TIER_KEY);
  if (stored === 'demo' || stored === 'full') return stored;
  return null;
}

export function clearUserTier(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(TIER_KEY);
  }
}

/**
 * Returns true if the current user is on the demo tier.
 * On non-demo deployments this always returns false.
 */
export function isDemoUser(): boolean {
  if (!DEMO_MODE) return false;
  return getUserTier() === 'demo';
}
