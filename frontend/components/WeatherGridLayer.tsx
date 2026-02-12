'use client';

import { useEffect, useRef, useState } from 'react';
import { WindFieldData, WaveFieldData, GridFieldData } from '@/lib/api';
import { debugLog } from '@/lib/debugLog';
import { bilinearInterpolate, bilinearOcean } from '@/lib/gridInterpolation';

interface WeatherGridLayerProps {
  mode: 'wind' | 'waves' | 'ice' | 'visibility' | 'sst' | 'swell';
  windData?: WindFieldData | null;
  waveData?: WaveFieldData | null;
  extendedData?: GridFieldData | null;
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

// Ice concentration color scale: fraction (0-1) → RGBA
// 0-5% blue-tint, 5-15% yellow-warning, >15% red-exclusion
function iceColor(concentration: number): [number, number, number, number] {
  if (concentration <= 0.01) return [0, 0, 0, 0]; // below 1% — transparent
  if (concentration <= 0.05) {
    const t = concentration / 0.05;
    return [Math.round(100 + t * 80), Math.round(180 + t * 40), Math.round(220 - t * 20), 150];
  }
  if (concentration <= 0.15) {
    const t = (concentration - 0.05) / 0.10;
    return [Math.round(180 + t * 60), Math.round(220 - t * 100), Math.round(50 - t * 30), 170];
  }
  // >15% — red zone (exclusion territory)
  const t = Math.min(1, (concentration - 0.15) / 0.35);
  return [Math.round(220 + t * 20), Math.round(30 + t * 10), Math.round(20), 190];
}

// Visibility color scale: meters → RGBA
// <1000m dark grey (fog), 1000-5000m grey, >5000m fading out
function visibilityColor(vis_m: number): [number, number, number, number] {
  if (vis_m > 10000) return [0, 0, 0, 0]; // clear — transparent
  if (vis_m > 5000) {
    const t = (10000 - vis_m) / 5000;
    return [180, 180, 180, Math.round(t * 80)];
  }
  if (vis_m > 2000) {
    const t = (5000 - vis_m) / 3000;
    return [Math.round(180 - t * 40), Math.round(180 - t * 40), Math.round(180 - t * 20), Math.round(80 + t * 60)];
  }
  if (vis_m > 1000) {
    const t = (2000 - vis_m) / 1000;
    return [Math.round(140 - t * 30), Math.round(140 - t * 30), Math.round(160 - t * 20), Math.round(140 + t * 30)];
  }
  // <1000m — dense fog, dark
  const t = Math.min(1, (1000 - vis_m) / 1000);
  return [Math.round(110 - t * 30), Math.round(110 - t * 30), Math.round(140 - t * 30), Math.round(170 + t * 30)];
}

// SST color scale: Celsius → RGBA (blue=cold → cyan → green → yellow → red=warm)
function sstColor(temp: number): [number, number, number, number] {
  const stops: [number, number, number, number][] = [
    [-2,   30,  40, 180],  // below freezing — deep blue
    [ 5,   50, 120, 220],  // cold — blue
    [10,    0, 200, 220],  // cool — cyan
    [15,    0, 200,  80],  // mild — green
    [20,  200, 220,   0],  // warm — yellow
    [25,  240, 140,   0],  // hot — orange
    [30,  220,  40,  30],  // tropical — red
  ];

  if (temp <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], 160];
  if (temp >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], 180];

  for (let i = 0; i < stops.length - 1; i++) {
    if (temp >= stops[i][0] && temp < stops[i + 1][0]) {
      const t = (temp - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        170,
      ];
    }
  }
  return [220, 40, 30, 180];
}

// Swell height color scale: meters → RGBA (reuse wave palette, softer)
function swellColor(height: number): [number, number, number, number] {
  const stops: [number, number, number, number][] = [
    [0,   60, 120, 200],  // 0m  - calm blue
    [1,    0, 200, 180],  // 1m  - teal
    [2,  100, 200,  50],  // 2m  - green
    [3,  240, 200,   0],  // 3m  - yellow
    [5,  240, 100,   0],  // 5m  - orange
    [8,  200,  30,  30],  // 8m+ - red
  ];

  if (height <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], 140];
  if (height >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], 190];

  for (let i = 0; i < stops.length - 1; i++) {
    if (height >= stops[i][0] && height < stops[i + 1][0]) {
      const t = (height - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        160,
      ];
    }
  }
  return [200, 30, 30, 190];
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
  extendedData,
  opacity = 0.7,
  showArrows = true,
}: WeatherGridLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);

  // Store data in refs so createTile always reads the latest values
  // without triggering a full layer destroy/recreate cycle.
  const windDataRef = useRef(windData);
  const waveDataRef = useRef(waveData);
  const extendedDataRef = useRef(extendedData);
  useEffect(() => { windDataRef.current = windData; }, [windData]);
  useEffect(() => { waveDataRef.current = waveData; }, [waveData]);
  useEffect(() => { extendedDataRef.current = extendedData; }, [extendedData]);

  // Create the GridLayer ONCE per mode/map/opacity — NOT per data change.
  // The createTile closure reads data from refs, so it always gets current values.
  useEffect(() => {
    const currentMode = mode;
    const currentShowArrows = showArrows;
    const DS = 64; // render at 64x64, upscale to 256x256

    const WeatherTileLayer = L.GridLayer.extend({
      createTile(coords: any) {
        const tile = document.createElement('canvas') as HTMLCanvasElement;
        tile.width = 256;
        tile.height = 256;

        // Read latest data from refs
        const currentWindData = windDataRef.current;
        const currentWaveData = waveDataRef.current;
        const currentExtendedData = extendedDataRef.current;
        const isExtended = currentMode === 'ice' || currentMode === 'visibility' || currentMode === 'sst' || currentMode === 'swell';
        const data = isExtended ? currentExtendedData : (currentMode === 'wind' ? currentWindData : currentWaveData);
        if (!data) return tile;

        const ctx = tile.getContext('2d');
        if (!ctx) return tile;

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
        const isWind = currentMode === 'wind' && currentWindData;
        const uData = isWind ? (currentWindData as WindFieldData).u : null;
        const vData = isWind ? (currentWindData as WindFieldData).v : null;
        const waveValues = currentMode === 'waves' && currentWaveData ? (currentWaveData as WaveFieldData).data : null;
        const waveDir = currentMode === 'waves' && currentWaveData ? (currentWaveData as WaveFieldData).direction : null;
        const extValues = isExtended && currentExtendedData ? currentExtendedData.data : null;

        // Compute lat/lon ranges (handle both ascending and descending order)
        const latStart = lats[0];
        const latEnd = lats[ny - 1];
        const latMin = Math.min(latStart, latEnd);
        const latMax = Math.max(latStart, latEnd);
        const lonStart = lons[0];
        const lonEnd = lons[nx - 1];
        const lonMin = Math.min(lonStart, lonEnd);
        const lonMax = Math.max(lonStart, lonEnd);

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

            // Find grid indices (fractional) — use original array ordering
            const latFracIdx = ((lat - latStart) / (latEnd - latStart)) * (ny - 1);
            const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
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
            } else if (extValues) {
              const val = bilinearInterpolate(extValues, latIdx, lonIdx, latFrac, lonFrac, ny, nx);
              if (currentMode === 'ice') color = iceColor(val);
              else if (currentMode === 'visibility') color = visibilityColor(val);
              else if (currentMode === 'sst') color = sstColor(val);
              else color = swellColor(val); // swell
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
        if (currentShowArrows && currentMode === 'wind' && uData && vData) {
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

              const latFracIdx = ((aLat - latStart) / (latEnd - latStart)) * (ny - 1);
              const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
              const aLatIdx = Math.floor(latFracIdx);
              const aLonIdx = Math.floor(lonFracIdx);
              const aLatFrac = latFracIdx - aLatIdx;
              const aLonFrac = lonFracIdx - aLonIdx;

              // Skip land (nearest-neighbor on high-res mask)
              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLats[0]) / (maskLats[maskNy - 1] - maskLats[0])) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLons[0]) / (maskLons[maskNx - 1] - maskLons[0])) * (maskNx - 1));
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

        // Draw swell & wind-wave direction crest marks (Windy-style arcs)
        if (currentShowArrows && currentMode === 'waves') {
          const waveW = currentWaveData as any; // access decomposition fields
          const swellDir = waveW?.swell?.direction as number[][] | undefined;
          const swellHt = waveW?.swell?.height as number[][] | undefined;
          const wwDir = waveW?.windwave?.direction as number[][] | undefined;
          const wwHt = waveW?.windwave?.height as number[][] | undefined;
          const hasSwell = swellDir && swellHt;
          const hasWW = wwDir && wwHt;

          const spacing = 30;
          ctx.save();

          // Helper: draw wave crest arcs perpendicular to propagation direction
          const drawWaveCrest = (
            cx: number, cy: number, dirDeg: number, height: number,
            color: string, alpha: number,
          ) => {
            // Met "from" → propagation direction (+180°)
            const propRad = ((dirDeg + 180) * Math.PI) / 180;
            // Perpendicular to propagation (crest line)
            const perpRad = propRad + Math.PI / 2;
            // Arc length proportional to wave height
            const arcLen = Math.min(12, 4 + height * 3);
            // Curvature amount (how much the arc bows toward propagation)
            const curve = Math.min(4, 1.5 + height * 0.8);
            // Offset between parallel crests along propagation direction
            const crestGap = 3.5;

            ctx.lineCap = 'round';

            // Draw 3 parallel crest arcs
            for (let k = -1; k <= 1; k++) {
              const ox = cx + Math.cos(propRad) * k * crestGap;
              const oy = cy + Math.sin(propRad) * k * crestGap;
              // Scale: center arc is largest, outer arcs are smaller
              const scale = k === 0 ? 1.0 : 0.7;
              const halfLen = arcLen * scale * 0.5;

              // Start and end of arc along perpendicular direction
              const x0 = ox - Math.cos(perpRad) * halfLen;
              const y0 = oy - Math.sin(perpRad) * halfLen;
              const x1 = ox + Math.cos(perpRad) * halfLen;
              const y1 = oy + Math.sin(perpRad) * halfLen;

              // Control point bowed in propagation direction
              const cpx = ox + Math.cos(propRad) * curve * scale;
              const cpy = oy + Math.sin(propRad) * curve * scale;

              ctx.strokeStyle = color;
              ctx.globalAlpha = alpha * (k === 0 ? 1.0 : 0.6);
              ctx.lineWidth = k === 0 ? 1.5 : 1.0;
              ctx.beginPath();
              ctx.moveTo(x0, y0);
              ctx.quadraticCurveTo(cpx, cpy, x1, y1);
              ctx.stroke();
            }
            ctx.globalAlpha = 1.0;
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

              const latFracIdx = ((aLat - latStart) / (latEnd - latStart)) * (ny - 1);
              const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
              const aLatIdx = Math.floor(latFracIdx);
              const aLonIdx = Math.floor(lonFracIdx);
              const aLatFrac = latFracIdx - aLatIdx;
              const aLonFrac = lonFracIdx - aLonIdx;

              // Skip land
              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLats[0]) / (maskLats[maskNy - 1] - maskLats[0])) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLons[0]) / (maskLons[maskNx - 1] - maskLons[0])) * (maskNx - 1));
                if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              // Primary swell: white crest arcs, size ∝ height
              if (hasSwell) {
                const sh = bilinearInterpolate(swellHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (sh > 0.2) {
                  const sd = bilinearInterpolate(swellDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const alpha = Math.min(0.95, 0.5 + sh * 0.15);
                  drawWaveCrest(ax, ay, sd, sh, 'rgba(255,255,255,1)', alpha);
                }
              }

              // Wind-wave: lighter/smaller crest arcs, offset slightly
              if (hasWW) {
                const wh = bilinearInterpolate(wwHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (wh > 0.2) {
                  const wd = bilinearInterpolate(wwDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const alpha = Math.min(0.8, 0.4 + wh * 0.12);
                  drawWaveCrest(ax + 6, ay + 6, wd, wh * 0.8, 'rgba(200,230,255,1)', alpha);
                }
              }

              // Fallback: mean direction if no decomposition
              if (!hasSwell && !hasWW && waveDir && waveValues) {
                const h = bilinearInterpolate(waveValues, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (h > 0.3) {
                  const d = bilinearInterpolate(waveDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const alpha = Math.min(0.9, 0.5 + h * 0.15);
                  drawWaveCrest(ax, ay, d, h, 'rgba(255,255,255,1)', alpha);
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
  }, [mode, opacity, map, L, showArrows]); // NOT windData/waveData — data is read from refs

  // When data changes, just redraw tiles (no layer destroy/recreate).
  // This is the key fix: avoids creating hundreds of canvas elements per frame change.
  const redrawCountRef = useRef(0);
  useEffect(() => {
    if (layerRef.current) {
      redrawCountRef.current++;
      const isExt = mode === 'ice' || mode === 'visibility' || mode === 'sst' || mode === 'swell';
      const sample = isExt && extendedData?.data
        ? extendedData.data[Math.floor(extendedData.data.length / 2)]?.[0]?.toFixed(2) ?? '?'
        : mode === 'waves' && waveData?.data
          ? waveData.data[Math.floor(waveData.data.length / 2)]?.[0]?.toFixed(2) ?? '?'
          : mode === 'wind' && windData?.u
            ? windData.u[Math.floor(windData.u.length / 2)]?.[0]?.toFixed(2) ?? '?'
            : '?';
      debugLog('debug', 'RENDER', `GridLayer redraw #${redrawCountRef.current}: mode=${mode}, sample=${sample}, hasLayer=${!!layerRef.current}`);
      layerRef.current.redraw();
    }
  }, [windData, waveData, extendedData, mode]);

  return null;
}
