/**
 * Shared bilinear interpolation utilities for weather grid data.
 * Used by WeatherGridLayer (tile rendering) and WaveInfoPopup (click queries).
 */

/** Bilinear interpolation on a 2D numeric grid */
export function bilinearInterpolate(
  data: number[][],
  latIdx: number,
  lonIdx: number,
  latFrac: number,
  lonFrac: number,
  ny: number,
  nx: number,
): number {
  const i0 = Math.max(0, Math.min(latIdx, ny - 1));
  const i1 = Math.max(0, Math.min(latIdx + 1, ny - 1));
  const j0 = Math.max(0, Math.min(lonIdx, nx - 1));
  const j1 = Math.max(0, Math.min(lonIdx + 1, nx - 1));

  const v00 = data[i0]?.[j0] ?? 0;
  const v01 = data[i0]?.[j1] ?? 0;
  const v10 = data[i1]?.[j0] ?? 0;
  const v11 = data[i1]?.[j1] ?? 0;

  const top = v00 + lonFrac * (v01 - v00);
  const bot = v10 + lonFrac * (v11 - v10);
  return top + latFrac * (bot - top);
}

/** Bilinear interpolation for boolean ocean mask (returns fraction 0-1) */
export function bilinearOcean(
  mask: boolean[][],
  latIdx: number,
  lonIdx: number,
  latFrac: number,
  lonFrac: number,
  ny: number,
  nx: number,
): number {
  const i0 = Math.max(0, Math.min(latIdx, ny - 1));
  const i1 = Math.max(0, Math.min(latIdx + 1, ny - 1));
  const j0 = Math.max(0, Math.min(lonIdx, nx - 1));
  const j1 = Math.max(0, Math.min(lonIdx + 1, nx - 1));

  const v00 = mask[i0]?.[j0] ? 1 : 0;
  const v01 = mask[i0]?.[j1] ? 1 : 0;
  const v10 = mask[i1]?.[j0] ? 1 : 0;
  const v11 = mask[i1]?.[j1] ? 1 : 0;

  const top = v00 + lonFrac * (v01 - v00);
  const bot = v10 + lonFrac * (v11 - v10);
  return top + latFrac * (bot - top);
}

/** Compute fractional grid indices for a given lat/lon within a regular grid */
export function getGridIndices(
  lat: number,
  lon: number,
  lats: number[],
  lons: number[],
): { latIdx: number; lonIdx: number; latFrac: number; lonFrac: number } | null {
  const ny = lats.length;
  const nx = lons.length;
  if (ny < 2 || nx < 2) return null;
  const latMin = lats[0];
  const latMax = lats[ny - 1];
  const lonMin = lons[0];
  const lonMax = lons[nx - 1];

  if (latMax === latMin || lonMax === lonMin) return null;
  if (lat < latMin || lat > latMax || lon < lonMin || lon > lonMax) return null;

  const latFracIdx = ((lat - latMin) / (latMax - latMin)) * (ny - 1);
  const lonFracIdx = ((lon - lonMin) / (lonMax - lonMin)) * (nx - 1);
  const latIdx = Math.floor(latFracIdx);
  const lonIdx = Math.floor(lonFracIdx);
  const latFrac = latFracIdx - latIdx;
  const lonFrac = lonFracIdx - lonIdx;

  return { latIdx, lonIdx, latFrac, lonFrac };
}
