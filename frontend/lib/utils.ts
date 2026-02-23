import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number, decimals: number = 1): string {
  return num.toFixed(decimals);
}

export function formatDistance(nm: number): string {
  return `${formatNumber(nm, 0)} nm`;
}

export function formatFuel(mt: number): string {
  return `${formatNumber(mt, 1)} MT`;
}

export function formatPower(kw: number): string {
  return `${formatNumber(kw, 0)} kW`;
}

export function formatDuration(hours: number): string {
  const days = Math.floor(hours / 24);
  const remainingHours = Math.floor(hours % 24);

  if (days > 0) {
    return `${days}d ${remainingHours}h`;
  }
  return `${remainingHours}h`;
}

export function formatSpeed(kts: number): string {
  return `${formatNumber(kts, 1)} kts`;
}

/**
 * Map SOG to a color gradient: green (service speed) → yellow → red (minimum SOG).
 * Uses HSL interpolation: H=120 (green) at maxSog, H=0 (red) at minSog.
 */
export function sogToColor(sog: number, minSog: number, maxSog: number): string {
  if (maxSog <= minSog) return 'hsl(120, 80%, 45%)';
  const t = Math.max(0, Math.min(1, (sog - minSog) / (maxSog - minSog)));
  const hue = Math.round(t * 120); // 0=red, 60=yellow, 120=green
  return `hsl(${hue}, 80%, 45%)`;
}

export function formatDate(date: string | Date): string {
  const d = new Date(date);
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}
