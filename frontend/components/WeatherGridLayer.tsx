'use client';

import { useEffect, useRef, useState } from 'react';
import { WindFieldData, WaveFieldData, GridFieldData, SwellFieldData } from '@/lib/api';
import { debugLog } from '@/lib/debugLog';
import { bilinearInterpolate, bilinearOcean } from '@/lib/gridInterpolation';

interface WeatherGridLayerProps {
  mode: 'wind' | 'waves' | 'ice' | 'visibility' | 'sst' | 'swell';
  windData?: WindFieldData | null;
  waveData?: WaveFieldData | null;
  extendedData?: GridFieldData | null;
  opacity?: number;
  showArrows?: boolean;
  arrowsOnly?: boolean;
}

// Reusable color ramp interpolator for meteorological fields
type ColorStop = [number, number, number, number]; // [threshold, R, G, B]

function interpolateColorRamp(
  value: number, stops: ColorStop[],
  alphaLow: number, alphaHigh: number, alphaDefault: number
): [number, number, number, number] {
  if (value <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], alphaLow];
  if (value >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], alphaHigh];

  for (let i = 0; i < stops.length - 1; i++) {
    if (value >= stops[i][0] && value < stops[i + 1][0]) {
      const t = (value - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        alphaDefault,
      ];
    }
  }
  return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], alphaHigh];
}

// WMO/TD-No. 1215 ice concentration ramp stops (fraction 0-1)
const ICE_RAMP: ColorStop[] = [
  [0.00,   0, 100, 255],  // ice free — blue
  [0.10, 150, 200, 255],  // < 1/10 — light blue
  [0.30, 140, 255, 160],  // 1-3/10 — green
  [0.60, 255, 255,   0],  // 4-6/10 — yellow
  [0.80, 255, 125,   7],  // 7-8/10 — orange
  [1.00, 255,   0,   0],  // 9-10/10 — red
];

// SST ramp with near-freezing stop for Baltic/Arctic (°C)
const SST_RAMP: ColorStop[] = [
  [-2,  20,  30, 140],  // below freezing — deep blue
  [ 2,  40,  80, 200],  // near freezing — blue
  [ 8,   0, 180, 220],  // cold — cyan
  [14,   0, 200,  80],  // cool — green
  [20, 220, 220,   0],  // warm — yellow
  [26, 240, 130,   0],  // hot — orange
  [32, 220,  30,  30],  // tropical — red
];

// Swell height ramp (meters)
const SWELL_RAMP: ColorStop[] = [
  [0,  60, 120, 200],  // calm — blue
  [1,   0, 200, 180],  // 1m — teal
  [2, 100, 200,  50],  // 2m — green
  [3, 240, 200,   0],  // 3m — yellow
  [5, 240, 100,   0],  // 5m — orange
  [8, 200,  30,  30],  // 8m+ — red
];

// Swell period ramp for arrow coloring (seconds → color)
// Short wind-sea spillover = white, moderate swell = gold, long ocean swell = cyan, very long = deep blue
const SWELL_PERIOD_RAMP: ColorStop[] = [
  [ 5, 255, 255, 255],  // 5s — white (short-period)
  [ 7, 255, 220, 120],  // 7s — light gold
  [10, 240, 180,  40],  // 10s — amber
  [12,  40, 220, 240],  // 12s — cyan
  [15,  20, 160, 240],  // 15s — bright blue
  [18,  30,  60, 180],  // 18s+ — deep blue
];

function swellPeriodColor(period: number): string {
  const [r, g, b] = interpolateColorRamp(period, SWELL_PERIOD_RAMP, 255, 255, 255);
  return `rgb(${r},${g},${b})`;
}

// Visibility ramp (km) — green nuances, only rendered where vis < 20 km
const VIS_RAMP: ColorStop[] = [
  [  0,  20,  80,  10],  // 0 km — dark green (dense fog)
  [  1,  40, 120,  20],  // 1 km — deep green (fog)
  [  4,  80, 170,  40],  // 4 km — green (poor)
  [ 10, 130, 210,  70],  // 10 km — bright green (moderate)
  [ 20, 180, 240, 120],  // 20 km — light green (fading out)
];

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

// Wave color scale: meters → RGBA (Windy-inspired 8-stop vibrant ramp)
function waveColor(height: number): [number, number, number, number] {
  // Brighter ramp: calm seas must be visible against the dark basemap.
  // 0→blue, 0.5→sky blue, 1→cyan-green, 1.5→green-yellow,
  // 2→yellow, 3→orange, 4→red-magenta, 6+→purple
  const stops: [number, number, number, number][] = [
    [0,    60, 110, 220],  // 0m   - medium blue (calm — visible on dark bg)
    [0.5,  30, 160, 240],  // 0.5m - sky blue
    [1,     0, 200, 170],  // 1m   - cyan-green
    [1.5, 120, 220,  40],  // 1.5m - green-yellow
    [2,   240, 220,   0],  // 2m   - yellow
    [3,   240, 130,   0],  // 3m   - orange
    [4,   220,  30,  80],  // 4m   - red-magenta
    [6,   160,   0, 180],  // 6m+  - purple
  ];

  if (height <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3], 170];
  if (height >= stops[stops.length - 1][0])
    return [stops[stops.length - 1][1], stops[stops.length - 1][2], stops[stops.length - 1][3], 190];

  for (let i = 0; i < stops.length - 1; i++) {
    if (height >= stops[i][0] && height < stops[i + 1][0]) {
      const t = (height - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + t * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + t * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + t * (stops[i + 1][3] - stops[i][3])),
        175,
      ];
    }
  }
  return [160, 0, 180, 190];
}

// Ice concentration color scale: fraction (0-1) → RGBA (WMO/TD-No. 1215)
function iceColor(concentration: number): [number, number, number, number] {
  if (concentration <= 0.01) return [0, 0, 0, 0]; // below 1% — transparent
  return interpolateColorRamp(concentration, ICE_RAMP, 120, 200, 180);
}

// Visibility color scale: km → RGBA (transparent background, green overlay for reduced vis)
function visibilityColor(vis_km: number): [number, number, number, number] {
  if (vis_km < 0 || vis_km > 20) return [0, 0, 0, 0]; // clear or invalid — transparent
  const [r, g, b] = interpolateColorRamp(vis_km, VIS_RAMP, 0, 0, 0);
  // Inverse alpha: fog (0 km) = 220, poor (4 km) ≈ 175, moderate (10 km) ≈ 110, good (20 km) = 0
  const alpha = Math.round(220 * (1 - vis_km / 20));
  return [r, g, b, alpha];
}

// SST color scale: Celsius → RGBA (7-stop ramp with near-freezing for Baltic/Arctic)
function sstColor(temp: number): [number, number, number, number] {
  return interpolateColorRamp(temp, SST_RAMP, 160, 180, 170);
}

// Swell height color scale: meters → RGBA
function swellColor(height: number): [number, number, number, number] {
  return interpolateColorRamp(height, SWELL_RAMP, 140, 190, 160);
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
  arrowsOnly = false,
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
    const currentArrowsOnly = arrowsOnly;
    const DS = 128; // render at 128x128, upscale to 256x256
    const tileSize = 256;

    // ---- Extracted tile painter (shared by createTile + refreshTiles) ----
    // Renders heatmap to a small offscreen canvas then copies to the visible
    // tile. clearRect is deferred until just before drawImage so old content
    // stays visible while the new frame renders (no flicker).
    function paintTile(tile: HTMLCanvasElement, coords: any) {
        const currentWindData = windDataRef.current;
        const currentWaveData = waveDataRef.current;
        const currentExtendedData = extendedDataRef.current;
        const isExtended = currentMode === 'ice' || currentMode === 'visibility' || currentMode === 'sst' || currentMode === 'swell';
        const data = isExtended ? currentExtendedData : (currentMode === 'wind' ? currentWindData : currentWaveData);

        const ctx = tile.getContext('2d');
        if (!ctx) return;
        if (!data) { ctx.clearRect(0, 0, 256, 256); return; }

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

        // SST auto-scaling: map data_min..data_max → ramp range for visible animation
        const sstScale = (currentMode === 'sst' && currentExtendedData?.colorscale?.data_min != null)
          ? { dMin: currentExtendedData.colorscale.data_min as number, dMax: currentExtendedData.colorscale.data_max as number }
          : null;

        // Compute lat/lon ranges (handle both ascending and descending order)
        const latStart = lats[0];
        const latEnd = lats[ny - 1];
        const latMin = Math.min(latStart, latEnd);
        const latMax = Math.max(latStart, latEnd);
        const lonStart = lons[0];
        const lonEnd = lons[nx - 1];
        const lonMin = Math.min(lonStart, lonEnd);
        const lonMax = Math.max(lonStart, lonEnd);

        const zoom = coords.z;

        // In arrowsOnly mode, skip the heatmap pixel loop (tiles handle it)
        if (currentArrowsOnly) {
          ctx.clearRect(0, 0, 256, 256);
        } else {

        // Create offscreen canvas at downsampled resolution
        const offscreen = document.createElement('canvas');
        offscreen.width = DS;
        offscreen.height = DS;
        const offCtx = offscreen.getContext('2d');
        if (!offCtx) return;

        const imgData = offCtx.createImageData(DS, DS);
        const pixels = imgData.data;

        for (let py = 0; py < DS; py++) {
          for (let px = 0; px < DS; px++) {
            const realX = (px / DS) * tileSize;
            const realY = (py / DS) * tileSize;

            const globalX = coords.x * tileSize + realX;
            const globalY = coords.y * tileSize + realY;

            const lng = (globalX / Math.pow(2, zoom) / tileSize) * 360 - 180;
            const latRad = Math.atan(
              Math.sinh(Math.PI * (1 - (2 * globalY) / (Math.pow(2, zoom) * tileSize)))
            );
            const lat = (latRad * 180) / Math.PI;

            if (lat < latMin || lat > latMax || lng < lonMin || lng > lonMax) {
              const idx = (py * DS + px) * 4;
              pixels[idx + 3] = 0;
              continue;
            }

            const fadeDeg = 2;
            const edgeFade = Math.min(
              Math.min((lat - latMin) / fadeDeg, (latMax - lat) / fadeDeg),
              Math.min((lng - lonMin) / fadeDeg, (lonMax - lng) / fadeDeg),
              1,
            );

            const latFracIdx = ((lat - latStart) / (latEnd - latStart)) * (ny - 1);
            const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
            const latIdx = Math.floor(latFracIdx);
            const lonIdx = Math.floor(lonFracIdx);
            const latFrac = latFracIdx - latIdx;
            const lonFrac = lonFracIdx - lonIdx;

            if (oceanMask) {
              const mLatIdx = Math.round(((lat - maskLatMin) / (maskLatMax - maskLatMin)) * (maskNy - 1));
              const mLonIdx = Math.round(((lng - maskLonMin) / (maskLonMax - maskLonMin)) * (maskNx - 1));
              if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) {
                const idx = (py * DS + px) * 4;
                pixels[idx + 3] = 0;
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
              if (Number.isNaN(h)) {
                const idx = (py * DS + px) * 4;
                pixels[idx + 3] = 0;
                continue;
              }
              color = waveColor(h);
            } else if (extValues) {
              const val = bilinearInterpolate(extValues, latIdx, lonIdx, latFrac, lonFrac, ny, nx);
              if (Number.isNaN(val) || val < -100) {
                const idx = (py * DS + px) * 4;
                pixels[idx + 3] = 0;
                continue;
              }
              if (currentMode === 'ice') color = iceColor(val);
              else if (currentMode === 'visibility') color = visibilityColor(val);
              else if (currentMode === 'sst') {
                // Auto-scale: map [data_min, data_max] → [-2, 32] so the full
                // color ramp is utilized and frame-to-frame changes are visible.
                if (sstScale && sstScale.dMax > sstScale.dMin) {
                  const t = (val - sstScale.dMin) / (sstScale.dMax - sstScale.dMin);
                  color = sstColor(-2 + t * 34);
                } else {
                  color = sstColor(val);
                }
              }
              else color = swellColor(val);
            } else {
              const idx = (py * DS + px) * 4;
              pixels[idx + 3] = 0;
              continue;
            }

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

        // Copy heatmap to visible tile (clear just before to minimise blank time)
        ctx.clearRect(0, 0, 256, 256);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(offscreen, 0, 0, DS, DS, 0, 0, 256, 256);

        } // end if (!currentArrowsOnly)

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

              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLats[0]) / (maskLats[maskNy - 1] - maskLats[0])) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLons[0]) / (maskLons[maskNx - 1] - maskLons[0])) * (maskNx - 1));
                if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              const u = bilinearInterpolate(uData, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              const v = bilinearInterpolate(vData, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              const speed = Math.sqrt(u * u + v * v);
              if (speed < 1) continue;

              const angle = Math.atan2(-v, u);
              const len = Math.min(16, 4 + speed * 0.8);

              ctx.translate(ax, ay);
              ctx.rotate(angle);

              ctx.strokeStyle = 'rgba(255,255,255,0.8)';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.moveTo(-len / 2, 0);
              ctx.lineTo(len / 2, 0);
              ctx.stroke();

              ctx.fillStyle = 'rgba(255,255,255,0.8)';
              ctx.beginPath();
              ctx.moveTo(len / 2, 0);
              ctx.lineTo(len / 2 - 4, -2.5);
              ctx.lineTo(len / 2 - 4, 2.5);
              ctx.closePath();
              ctx.fill();

              ctx.setTransform(1, 0, 0, 1, 0, 0);
            }
          }
          ctx.restore();
        }

        // Draw swell & wind-wave direction crest marks on composite (Windy-style arcs)
        if (currentShowArrows && currentMode === 'waves') {
          const waveW = currentWaveData as any;
          const swellDir = waveW?.swell?.direction as number[][] | undefined;
          const swellHt = waveW?.swell?.height as number[][] | undefined;
          const wwDir = waveW?.windwave?.direction as number[][] | undefined;
          const wwHt = waveW?.windwave?.height as number[][] | undefined;
          const hasSwell = swellDir && swellHt;
          const hasWW = wwDir && wwHt;

          const spacing = 30;
          ctx.save();

          const drawWaveCrest = (
            cx: number, cy: number, dirDeg: number, height: number,
            color: string, alpha: number,
          ) => {
            // Compass bearing to canvas angle: +180 (from→propagation), -90 (compass→canvas)
            const propRad = ((dirDeg + 90) * Math.PI) / 180;
            const perpRad = propRad + Math.PI / 2;
            const arcLen = Math.min(16, 6 + height * 4);
            const curve = Math.min(5, 2 + height * 1.0);
            const crestGap = 4.5;

            ctx.lineCap = 'round';

            for (let k = -1; k <= 1; k++) {
              const ox = cx + Math.cos(propRad) * k * crestGap;
              const oy = cy + Math.sin(propRad) * k * crestGap;
              const scale = k === 0 ? 1.0 : 0.7;
              const halfLen = arcLen * scale * 0.5;

              const x0 = ox - Math.cos(perpRad) * halfLen;
              const y0 = oy - Math.sin(perpRad) * halfLen;
              const x1 = ox + Math.cos(perpRad) * halfLen;
              const y1 = oy + Math.sin(perpRad) * halfLen;

              const cpx = ox + Math.cos(propRad) * curve * scale;
              const cpy = oy + Math.sin(propRad) * curve * scale;

              // Dark outline for contrast against heatmap
              ctx.strokeStyle = 'rgba(0,0,0,0.3)';
              ctx.globalAlpha = 1.0;
              ctx.lineWidth = (k === 0 ? 2.0 : 1.2) + 1.0;
              ctx.beginPath();
              ctx.moveTo(x0, y0);
              ctx.quadraticCurveTo(cpx, cpy, x1, y1);
              ctx.stroke();

              // Main crest stroke
              ctx.strokeStyle = color;
              ctx.globalAlpha = alpha * (k === 0 ? 1.0 : 0.6);
              ctx.lineWidth = k === 0 ? 2.0 : 1.2;
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

              if (oceanMask) {
                const mLatIdx = Math.round(((aLat - maskLats[0]) / (maskLats[maskNy - 1] - maskLats[0])) * (maskNy - 1));
                const mLonIdx = Math.round(((lng - maskLons[0]) / (maskLons[maskNx - 1] - maskLons[0])) * (maskNx - 1));
                if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              if (hasSwell) {
                const sh = bilinearInterpolate(swellHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (sh > 0.2) {
                  const sd = bilinearInterpolate(swellDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const alpha = Math.min(0.95, 0.5 + sh * 0.15);
                  drawWaveCrest(ax, ay, sd, sh, 'rgba(255,255,255,1)', alpha);
                }
              }

              if (hasWW) {
                const wh = bilinearInterpolate(wwHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (wh > 0.2) {
                  const wd = bilinearInterpolate(wwDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                  const alpha = Math.min(0.8, 0.4 + wh * 0.12);
                  drawWaveCrest(ax + 6, ay + 6, wd, wh * 0.8, 'rgba(200,230,255,1)', alpha);
                }
              }

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

        // Draw swell directional arrows (period-colored, height-scaled)
        if (currentShowArrows && currentMode === 'swell' && currentExtendedData) {
          const swellExt = currentExtendedData as SwellFieldData;
          const swDir = swellExt.swell_dir;
          const swHs = swellExt.swell_hs;
          const swTp = swellExt.swell_tp;

          if (swDir && swHs) {
            const spacing = 40;
            ctx.save();

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

                if (oceanMask) {
                  const mLatIdx = Math.round(((aLat - maskLats[0]) / (maskLats[maskNy - 1] - maskLats[0])) * (maskNy - 1));
                  const mLonIdx = Math.round(((lng - maskLons[0]) / (maskLons[maskNx - 1] - maskLons[0])) * (maskNx - 1));
                  if (mLatIdx < 0 || mLatIdx >= maskNy || mLonIdx < 0 || mLonIdx >= maskNx || !oceanMask[mLatIdx]?.[mLonIdx]) continue;
                }

                const hs = bilinearInterpolate(swHs, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                if (hs < 0.3) continue;

                const dir = bilinearInterpolate(swDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
                const tp = swTp ? bilinearInterpolate(swTp, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx) : 10;

                // Arrow points in propagation direction (where swell is going)
                const propRad = ((dir + 180) * Math.PI) / 180;
                // Length scaled by height: 10px at 0.3m, up to 20px at 5m+
                const len = Math.min(20, 10 + hs * 2.5);
                const color = swellPeriodColor(tp);

                ctx.translate(ax, ay);
                ctx.rotate(propRad); // arrow drawn pointing up (-y), rotate by compass bearing

                // Dark outline for contrast against heatmap
                ctx.globalAlpha = 0.6;
                ctx.strokeStyle = 'rgba(0,0,0,0.8)';
                ctx.lineWidth = 3;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(0, len / 2);
                ctx.lineTo(0, -len / 2);
                ctx.stroke();

                // Colored arrow shaft
                ctx.globalAlpha = 0.9;
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.8;
                ctx.beginPath();
                ctx.moveTo(0, len / 2);
                ctx.lineTo(0, -len / 2);
                ctx.stroke();

                // Colored arrowhead
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(0, -len / 2 - 1);
                ctx.lineTo(-4, -len / 2 + 5);
                ctx.lineTo(4, -len / 2 + 5);
                ctx.closePath();
                ctx.fill();

                ctx.globalAlpha = 1.0;
                ctx.setTransform(1, 0, 0, 1, 0, 0);
              }
            }
            ctx.restore();
          }
        }

    }

    const WeatherTileLayer = L.GridLayer.extend({
      createTile(coords: any) {
        const tile = document.createElement('canvas') as HTMLCanvasElement;
        tile.width = 256;
        tile.height = 256;
        paintTile(tile, coords);
        return tile;
      },
      // Repaint existing tile canvases in-place (no flash)
      refreshTiles() {
        const tiles = this._tiles;
        if (!tiles) return;
        for (const key in tiles) {
          const t = tiles[key];
          if (t.el && t.coords) {
            paintTile(t.el as HTMLCanvasElement, t.coords);
          }
        }
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
  }, [mode, opacity, map, L, showArrows, arrowsOnly]); // NOT windData/waveData — data is read from refs

  // When data changes, repaint existing tile canvases in-place (no flash).
  // refreshTiles() reuses DOM elements — avoids the destroy/recreate cycle.
  // rAF-throttled: multiple rapid data changes → one repaint per browser frame.
  const redrawCountRef = useRef(0);
  const refreshRafRef = useRef<number | null>(null);
  useEffect(() => {
    if (layerRef.current) {
      if (refreshRafRef.current !== null) cancelAnimationFrame(refreshRafRef.current);
      refreshRafRef.current = requestAnimationFrame(() => {
        refreshRafRef.current = null;
        if (!layerRef.current) return;
        redrawCountRef.current++;
        const isExt = mode === 'ice' || mode === 'visibility' || mode === 'sst' || mode === 'swell';
        const sample = isExt && extendedData?.data
          ? extendedData.data[Math.floor(extendedData.data.length / 2)]?.[0]?.toFixed(2) ?? '?'
          : mode === 'waves' && waveData?.data
            ? waveData.data[Math.floor(waveData.data.length / 2)]?.[0]?.toFixed(2) ?? '?'
            : mode === 'wind' && windData?.u
              ? windData.u[Math.floor(windData.u.length / 2)]?.[0]?.toFixed(2) ?? '?'
              : '?';
        debugLog('debug', 'RENDER', `GridLayer refresh #${redrawCountRef.current}: mode=${mode}, sample=${sample}`);
        if (layerRef.current.refreshTiles) {
          layerRef.current.refreshTiles();
        } else {
          layerRef.current.redraw();
        }
      });
    }
    return () => {
      if (refreshRafRef.current !== null) { cancelAnimationFrame(refreshRafRef.current); refreshRafRef.current = null; }
    };
  }, [windData, waveData, extendedData, mode]);

  return null;
}
