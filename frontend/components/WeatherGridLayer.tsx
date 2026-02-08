'use client';

import { useEffect, useRef, useState } from 'react';
import { WindFieldData, WaveFieldData } from '@/lib/api';

interface WeatherGridLayerProps {
  mode: 'wind' | 'waves';
  windData?: WindFieldData | null;
  waveData?: WaveFieldData | null;
  opacity?: number;
  showArrows?: boolean;
}

// Wind color scale: m/s → RGBA
function windColor(speed: number): [number, number, number, number] {
  // 0→blue, 5→cyan, 10→green, 15→yellow, 20→orange, 25+→red
  const stops: [number, number, number, number][] = [
    [0,  30,  80, 220],  // 0 m/s  - blue
    [5,  0,  200, 220],  // 5 m/s  - cyan
    [10, 0,  200,  50],  // 10 m/s - green
    [15, 240, 220,  0],  // 15 m/s - yellow
    [20, 240, 130,  0],  // 20 m/s - orange
    [25, 220,  30, 30],  // 25 m/s - red
  ];

  if (speed <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], 180];
  if (speed >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], 200];

  for (let i = 0; i < stops.length - 1; i++) {
    if (speed >= stops[i][0] && speed < stops[i + 1][0]) {
      const t = (speed - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        180,
      ];
    }
  }
  return [220, 30, 30, 200];
}

// Wave color scale: meters → RGBA
function waveColor(height: number): [number, number, number, number] {
  // 0→green, 1→yellow, 2→orange, 3→red, 5+→dark red
  const stops: [number, number, number, number][] = [
    [0,   0, 200,  50],  // 0m  - green
    [1, 240, 220,   0],  // 1m  - yellow
    [2, 240, 130,   0],  // 2m  - orange
    [3, 220,  30,  30],  // 3m  - red
    [5, 128,   0,   0],  // 5m+ - dark red
  ];

  if (height <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], 160];
  if (height >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], 200];

  for (let i = 0; i < stops.length - 1; i++) {
    if (height >= stops[i][0] && height < stops[i + 1][0]) {
      const t = (height - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        170,
      ];
    }
  }
  return [128, 0, 0, 200];
}

// Bilinear interpolation helper
function bilinearInterpolate(
  data: number[][],
  latIdx: number,
  lonIdx: number,
  latFrac: number,
  lonFrac: number,
  ny: number,
  nx: number,
): number {
  const i0 = Math.min(latIdx, ny - 1);
  const i1 = Math.min(latIdx + 1, ny - 1);
  const j0 = Math.min(lonIdx, nx - 1);
  const j1 = Math.min(lonIdx + 1, nx - 1);

  const v00 = data[i0]?.[j0] ?? 0;
  const v01 = data[i0]?.[j1] ?? 0;
  const v10 = data[i1]?.[j0] ?? 0;
  const v11 = data[i1]?.[j1] ?? 0;

  const top = v00 + lonFrac * (v01 - v00);
  const bot = v10 + lonFrac * (v11 - v10);
  return top + latFrac * (bot - top);
}

// Bilinear interpolation for boolean ocean mask (returns fraction 0-1)
function bilinearOcean(
  mask: boolean[][],
  latIdx: number,
  lonIdx: number,
  latFrac: number,
  lonFrac: number,
  ny: number,
  nx: number,
): number {
  const i0 = Math.min(latIdx, ny - 1);
  const i1 = Math.min(latIdx + 1, ny - 1);
  const j0 = Math.min(lonIdx, nx - 1);
  const j1 = Math.min(lonIdx + 1, nx - 1);

  const v00 = mask[i0]?.[j0] ? 1 : 0;
  const v01 = mask[i0]?.[j1] ? 1 : 0;
  const v10 = mask[i1]?.[j0] ? 1 : 0;
  const v11 = mask[i1]?.[j1] ? 1 : 0;

  const top = v00 + lonFrac * (v01 - v00);
  const bot = v10 + lonFrac * (v11 - v10);
  return top + latFrac * (bot - top);
}

export default function WeatherGridLayer(props: WeatherGridLayerProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return null;

  return <WeatherGridLayerInner {...props} />;
}

function WeatherGridLayerInner({
  mode,
  windData,
  waveData,
  opacity = 0.7,
  showArrows = true,
}: WeatherGridLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);

  useEffect(() => {
    const data = mode === 'wind' ? windData : waveData;
    if (!data) return;

    const lats = data.lats;
    const lons = data.lons;
    const ny = lats.length;
    const nx = lons.length;

    // High-res ocean mask (separate grid from weather data)
    const oceanMask = data.ocean_mask;
    const maskLats = data.ocean_mask_lats || lats;
    const maskLons = data.ocean_mask_lons || lons;
    const maskNy = maskLats.length;
    const maskNx = maskLons.length;
    const maskLatMin = maskLats[0];
    const maskLatMax = maskLats[maskNy - 1];
    const maskLonMin = maskLons[0];
    const maskLonMax = maskLons[maskNx - 1];

    // Determine data arrays
    const isWind = mode === 'wind' && windData;
    const uData = isWind ? (windData as WindFieldData).u : null;
    const vData = isWind ? (windData as WindFieldData).v : null;
    const waveValues = mode === 'waves' && waveData ? (waveData as WaveFieldData).data : null;
    const waveDir = mode === 'waves' && waveData ? (waveData as WaveFieldData).direction : null;

    // Compute lat/lon ranges
    const latMin = lats[0];
    const latMax = lats[ny - 1];
    const lonMin = lons[0];
    const lonMax = lons[nx - 1];

    // Downsampled tile size for performance
    const DS = 64; // render at 64x64, upscale to 256x256

    const WeatherTileLayer = L.GridLayer.extend({
      createTile(coords: any) {
        const tile = document.createElement('canvas') as HTMLCanvasElement;
        tile.width = 256;
        tile.height = 256;

        const ctx = tile.getContext('2d');
        if (!ctx) return tile;

        // Create offscreen canvas at downsampled resolution
        const offscreen = document.createElement('canvas');
        offscreen.width = DS;
        offscreen.height = DS;
        const offCtx = offscreen.getContext('2d');
        if (!offCtx) return tile;

        const imgData = offCtx.createImageData(DS, DS);
        const pixels = imgData.data;

        // Compute tile lat/lon bounds
        const tileSize = 256;
        const zoom = coords.z;

        for (let py = 0; py < DS; py++) {
          for (let px = 0; px < DS; px++) {
            // Map pixel to real pixel coords within the tile
            const realX = (px / DS) * tileSize;
            const realY = (py / DS) * tileSize;

            // Convert tile pixel to global point
            const globalX = coords.x * tileSize + realX;
            const globalY = coords.y * tileSize + realY;

            // Convert pixel to lat/lng
            const lng = (globalX / Math.pow(2, zoom) / tileSize) * 360 - 180;
            const latRad = Math.atan(
              Math.sinh(Math.PI * (1 - (2 * globalY) / (Math.pow(2, zoom) * tileSize)))
            );
            const lat = (latRad * 180) / Math.PI;

            // Check if within data bounds (with fade margin)
            if (lat < latMin || lat > latMax || lng < lonMin || lng > lonMax) {
              const idx = (py * DS + px) * 4;
              pixels[idx + 3] = 0; // transparent
              continue;
            }

            // Edge fade: fade out over ~2 degrees near data boundary
            const fadeDeg = 2;
            const edgeFade = Math.min(
              Math.min((lat - latMin) / fadeDeg, (latMax - lat) / fadeDeg),
              Math.min((lng - lonMin) / fadeDeg, (lonMax - lng) / fadeDeg),
              1,
            );

            // Find grid indices (fractional)
            const latFracIdx = ((lat - latMin) / (latMax - latMin)) * (ny - 1);
            const lonFracIdx = ((lng - lonMin) / (lonMax - lonMin)) * (nx - 1);
            const latIdx = Math.floor(latFracIdx);
            const lonIdx = Math.floor(lonFracIdx);
            const latFrac = latFracIdx - latIdx;
            const lonFrac = lonFracIdx - lonIdx;

            // Check ocean mask (nearest-neighbor against high-res grid)
            if (oceanMask) {
              const mLatIdx = Math.round(((lat - maskLatMin) / (maskLatMax - maskLatMin)) * (maskNy - 1));
              const mLonIdx = Math.round(((lng - maskLonMin) / (maskLonMax - maskLonMin)) * (maskNx - 1));
              if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) {
                const idx = (py * DS + px) * 4;
                pixels[idx + 3] = 0; // land → transparent
                continue;
              }
            }

            let color: [number, number, number, number];
            if (uData && vData) {
              const u = bilinearInterpolate(uData, latIdx, lonIdx, latFrac, lonFrac, ny, nx);
              const v = bilinearInterpolate(vData, latIdx, lonIdx, latFrac, lonFrac, ny, nx);
              const speed = Math.sqrt(u * u + v * v);
              color = windColor(speed);
            } else if (waveValues) {
              const h = bilinearInterpolate(waveValues, latIdx, lonIdx, latFrac, lonFrac, ny, nx);
              color = waveColor(h);
            } else {
              const idx = (py * DS + px) * 4;
              pixels[idx + 3] = 0;
              continue;
            }

            // Data boundary fade
            let alpha = color[3];
            alpha = Math.round(alpha * Math.max(0, Math.min(1, edgeFade)));

            const idx = (py * DS + px) * 4;
            pixels[idx] = color[0];
            pixels[idx + 1] = color[1];
            pixels[idx + 2] = color[2];
            pixels[idx + 3] = alpha;
          }
        }

        offCtx.putImageData(imgData, 0, 0);

        // Upscale to 256x256 with smoothing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(offscreen, 0, 0, DS, DS, 0, 0, 256, 256);

        // Draw wind arrows on top (sparse, every ~40px)
        if (showArrows && mode === 'wind' && uData && vData) {
          const arrowSpacing = 40;
          ctx.save();
          for (let ay = arrowSpacing / 2; ay < 256; ay += arrowSpacing) {
            for (let ax = arrowSpacing / 2; ax < 256; ax += arrowSpacing) {
              const globalX = coords.x * tileSize + ax;
              const globalY = coords.y * tileSize + ay;

              const lng = (globalX / Math.pow(2, zoom) / tileSize) * 360 - 180;
              const latRad = Math.atan(
                Math.sinh(Math.PI * (1 - (2 * globalY) / (Math.pow(2, zoom) * tileSize)))
              );
              const aLat = (latRad * 180) / Math.PI;

              if (aLat < latMin || aLat > latMax || lng < lonMin || lng > lonMax) continue;

              const latFracIdx = ((aLat - latMin) / (latMax - latMin)) * (ny - 1);
              const lonFracIdx = ((lng - lonMin) / (lonMax - lonMin)) * (nx - 1);
              const aLatIdx = Math.floor(latFracIdx);
              const aLonIdx = Math.floor(lonFracIdx);
              const aLatFrac = latFracIdx - aLatIdx;
              const aLonFrac = lonFracIdx - aLonIdx;

              // Skip land (nearest-neighbor on high-res mask)
              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLatMin) / (maskLatMax - maskLatMin)) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLonMin) / (maskLonMax - maskLonMin)) * (maskNx - 1));
                if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              const u = bilinearInterpolate(uData, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              const v = bilinearInterpolate(vData, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              const speed = Math.sqrt(u * u + v * v);
              if (speed < 1) continue;

              const angle = Math.atan2(-v, u); // screen Y is inverted
              const len = Math.min(16, 4 + speed * 0.8);

              ctx.translate(ax, ay);
              ctx.rotate(angle);

              ctx.strokeStyle = 'rgba(255,255,255,0.8)';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.moveTo(-len / 2, 0);
              ctx.lineTo(len / 2, 0);
              ctx.stroke();

              // Arrowhead
              ctx.fillStyle = 'rgba(255,255,255,0.8)';
              ctx.beginPath();
              ctx.moveTo(len / 2, 0);
              ctx.lineTo(len / 2 - 4, -2.5);
              ctx.lineTo(len / 2 - 4, 2.5);
              ctx.closePath();
              ctx.fill();

              ctx.setTransform(1, 0, 0, 1, 0, 0); // reset transform
            }
          }
          ctx.restore();
        }

        // Draw swell & wind-wave direction triangles (Windy-style filled triangles)
        if (showArrows && mode === 'waves') {
          const waveW = waveData as any; // access decomposition fields
          const swellDir = waveW?.swell?.direction as number[][] | undefined;
          const swellHt = waveW?.swell?.height as number[][] | undefined;
          const wwDir = waveW?.windwave?.direction as number[][] | undefined;
          const wwHt = waveW?.windwave?.height as number[][] | undefined;
          const hasSwell = swellDir && swellHt;
          const hasWW = wwDir && wwHt;

          const spacing = 30; // denser grid
          ctx.save();

          // Helper: draw a filled triangle at (ax,ay) pointing in propagation direction
          const drawTriangle = (
            ax: number, ay: number, dirDeg: number, size: number,
            fillColor: string, strokeColor: string,
          ) => {
            // Met "from" → propagation: +180°. Then to canvas angle.
            const propRad = ((dirDeg + 180) * Math.PI) / 180;
            const angle = propRad - Math.PI / 2;
            ctx.translate(ax, ay);
            ctx.rotate(angle);
            ctx.fillStyle = fillColor;
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(size, 0);                    // tip
            ctx.lineTo(-size * 0.55, -size * 0.45); // left
            ctx.lineTo(-size * 0.55, size * 0.45);  // right
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.setTransform(1, 0, 0, 1, 0, 0);
          };

          for (let ay = spacing / 2; ay < 256; ay += spacing) {
            for (let ax = spacing / 2; ax < 256; ax += spacing) {
              const globalX = coords.x * tileSize + ax;
              const globalY = coords.y * tileSize + ay;
              const lng = (globalX / Math.pow(2, zoom) / tileSize) * 360 - 180;
              const latRad = Math.atan(
                Math.sinh(Math.PI * (1 - (2 * globalY) / (Math.pow(2, zoom) * tileSize)))
              );
              const aLat = (latRad * 180) / Math.PI;
              if (aLat < latMin || aLat > latMax || lng < lonMin || lng > lonMax) continue;

              const latFracIdx = ((aLat - latMin) / (latMax - latMin)) * (ny - 1);
              const lonFracIdx = ((lng - lonMin) / (lonMax - lonMin)) * (nx - 1);
              const aLatIdx = Math.floor(latFracIdx);
              const aLonIdx = Math.floor(lonFracIdx);
              const aLatFrac = latFracIdx - aLatIdx;
              const aLonFrac = lonFracIdx - aLonIdx;

              // Skip land
              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLatMin) / (maskLatMax - maskLatMin)) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLonMin) / (maskLonMax - maskLonMin)) * (maskNx - 1));
                if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              // Primary swell: white filled triangle, size ∝ height
              if (hasSwell) {
                const sh = bilinearInterpolate(swellHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (sh > 0.2) {
                  const sd = bilinearInterpolate(swellDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const size = Math.min(9, 4 + sh * 2);
                  drawTriangle(ax, ay, sd, size, 'rgba(255,255,255,0.85)', 'rgba(180,180,180,0.6)');
                }
              }

              // Wind-wave: cyan filled triangle (smaller, offset slightly)
              if (hasWW) {
                const wh = bilinearInterpolate(wwHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (wh > 0.2) {
                  const wd = bilinearInterpolate(wwDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const size = Math.min(7, 3 + wh * 1.5);
                  drawTriangle(ax + 6, ay + 6, wd, size, 'rgba(0,210,230,0.8)', 'rgba(0,160,180,0.5)');
                }
              }

              // Fallback: mean direction if no decomposition
              if (!hasSwell && !hasWW && waveDir && waveValues) {
                const h = bilinearInterpolate(waveValues, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (h > 0.3) {
                  const d = bilinearInterpolate(waveDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const size = Math.min(8, 4 + h * 2);
                  drawTriangle(ax, ay, d, size, 'rgba(255,255,255,0.8)', 'rgba(180,180,180,0.5)');
                }
              }
            }
          }
          ctx.restore();
        }

        return tile;
      },
    });

    const layer = new WeatherTileLayer({
      opacity,
      tileSize: 256,
      updateWhenZooming: false,
      updateWhenIdle: true,
    });

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [mode, windData, waveData, opacity, map, L]);

  return null;
}
