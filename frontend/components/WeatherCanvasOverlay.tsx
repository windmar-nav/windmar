'use client';

import { useEffect, useRef, useCallback, useState } from 'react';
import { WindFieldData, WaveFieldData, GridFieldData, SwellFieldData } from '@/lib/api';

interface WeatherCanvasOverlayProps {
  mode: 'wind' | 'waves' | 'ice' | 'visibility' | 'sst' | 'swell';
  windData?: WindFieldData | null;
  waveData?: WaveFieldData | null;
  extendedData?: GridFieldData | SwellFieldData | null;
  opacity?: number;
}

// ── Color ramp definitions (same stops as WeatherGridLayer) ────────────

type ColorStop = [number, number, number, number]; // [threshold, R, G, B]

const WIND_RAMP: ColorStop[] = [
  [0,  30,  80, 220], [5,   0, 200, 220], [10,  0, 200,  50],
  [15, 240, 220,   0], [20, 240, 130,   0], [25, 220,  30,  30],
];
const WAVE_RAMP: ColorStop[] = [
  [0,    60, 110, 220], [0.5,  30, 160, 240], [1,     0, 200, 170],
  [1.5, 120, 220,  40], [2,   240, 220,   0], [3,   240, 130,   0],
  [4,   220,  30,  80], [6,   160,   0, 180],
];
const ICE_RAMP: ColorStop[] = [
  [0.00,   0, 100, 255], [0.10, 150, 200, 255], [0.30, 140, 255, 160],
  [0.60, 255, 255,   0], [0.80, 255, 125,   7], [1.00, 255,   0,   0],
];
const SST_RAMP: ColorStop[] = [
  [-2,  20,  30, 140], [ 2,  40,  80, 200], [ 8,   0, 180, 220],
  [14,   0, 200,  80], [20, 220, 220,   0], [26, 240, 130,   0],
  [32, 220,  30,  30],
];
const SWELL_RAMP: ColorStop[] = [
  [0,  60, 120, 200], [1,   0, 200, 180], [2, 100, 200,  50],
  [3, 240, 200,   0], [5, 240, 100,   0], [8, 200,  30,  30],
];
const VIS_RAMP: ColorStop[] = [
  [  0,  20,  80,  10], [  1,  40, 120,  20], [  4,  80, 170,  40],
  [ 10, 130, 210,  70], [ 20, 180, 240, 120],
];

// ── LUT builder ────────────────────────────────────────────────────────

const LUT_SIZE = 1024;

function interpolateRamp(t: number, stops: ColorStop[]): [number, number, number] {
  if (t <= stops[0][0]) return [stops[0][1], stops[0][2], stops[0][3]];
  const last = stops[stops.length - 1];
  if (t >= last[0]) return [last[1], last[2], last[3]];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t < stops[i + 1][0]) {
      const f = (t - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      return [
        Math.round(stops[i][1] + f * (stops[i + 1][1] - stops[i][1])),
        Math.round(stops[i][2] + f * (stops[i + 1][2] - stops[i][2])),
        Math.round(stops[i][3] + f * (stops[i + 1][3] - stops[i][3])),
      ];
    }
  }
  return [last[1], last[2], last[3]];
}

function buildColorLUT(mode: string, vMin: number, vMax: number): Uint8Array {
  const lut = new Uint8Array(LUT_SIZE * 4);
  const range = vMax - vMin || 1;

  let ramp: ColorStop[];
  let baseAlpha: number;
  switch (mode) {
    case 'wind':       ramp = WIND_RAMP;  baseAlpha = 180; break;
    case 'waves':      ramp = WAVE_RAMP;  baseAlpha = 175; break;
    case 'ice':        ramp = ICE_RAMP;   baseAlpha = 180; break;
    case 'sst':        ramp = SST_RAMP;   baseAlpha = 170; break;
    case 'swell':      ramp = SWELL_RAMP; baseAlpha = 160; break;
    case 'visibility': ramp = VIS_RAMP;   baseAlpha = 180; break;
    default:           ramp = WIND_RAMP;  baseAlpha = 180;
  }

  for (let i = 0; i < LUT_SIZE; i++) {
    const value = vMin + (i / (LUT_SIZE - 1)) * range;
    const [r, g, b] = interpolateRamp(value, ramp);
    const off = i * 4;
    lut[off]     = r;
    lut[off + 1] = g;
    lut[off + 2] = b;
    if (mode === 'ice' && value <= 0.01) {
      lut[off + 3] = 0;
    } else if (mode === 'visibility') {
      lut[off + 3] = (value < 0 || value > 20) ? 0 : Math.round(220 * (1 - value / 20));
    } else {
      lut[off + 3] = baseAlpha;
    }
  }
  return lut;
}

// ── Value ranges per mode ──────────────────────────────────────────────

function getValueRange(
  mode: string,
  extData?: GridFieldData | SwellFieldData | null,
): [number, number] {
  switch (mode) {
    case 'wind':       return [0, 25];
    case 'waves':      return [0, 6];
    case 'ice':        return [0, 1];
    case 'swell':      return [0, 8];
    case 'visibility': return [0, 20];
    case 'sst': {
      const cs = extData?.colorscale;
      if (cs?.data_min != null && cs?.data_max != null && cs.data_max > cs.data_min)
        return [cs.data_min, cs.data_max];
      return [-2, 32];
    }
    default: return [0, 25];
  }
}

// ── Bilinear interpolation (inlined for tight loop perf) ───────────────

function bilinear(
  data: number[][], latFI: number, lonFI: number, ny: number, nx: number,
): number {
  const i0 = Math.min(Math.floor(latFI), ny - 1);
  const i1 = Math.min(i0 + 1, ny - 1);
  const j0 = Math.min(Math.floor(lonFI), nx - 1);
  const j1 = Math.min(j0 + 1, nx - 1);
  const lf = latFI - i0;
  const cf = lonFI - j0;
  const v00 = data[i0]?.[j0] ?? 0;
  const v01 = data[i0]?.[j1] ?? 0;
  const v10 = data[i1]?.[j0] ?? 0;
  const v11 = data[i1]?.[j1] ?? 0;
  return (v00 + cf * (v01 - v00)) + lf * ((v10 + cf * (v11 - v10)) - (v00 + cf * (v01 - v00)));
}

// ── Component ──────────────────────────────────────────────────────────

export default function WeatherCanvasOverlay(props: WeatherCanvasOverlayProps) {
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => { setIsMounted(true); }, []);
  if (!isMounted) return null;
  return <WeatherCanvasOverlayInner {...props} />;
}

function WeatherCanvasOverlayInner({
  mode,
  windData,
  waveData,
  extendedData,
  opacity = 0.6,
}: WeatherCanvasOverlayProps) {
  const { useMap } = require('react-leaflet');
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const lutRef = useRef<Uint8Array | null>(null);
  const lutKeyRef = useRef('');

  const isExtended = mode === 'ice' || mode === 'visibility' || mode === 'sst' || mode === 'swell';
  const activeData = isExtended ? extendedData : (mode === 'wind' ? windData : waveData);

  // ── Canvas lifecycle: create once, attach to Leaflet pane ────────

  useEffect(() => {
    const PANE = 'weatherHeatPane';
    if (!map.getPane(PANE)) {
      const pane = map.createPane(PANE);
      pane.style.zIndex = '300';
      pane.style.pointerEvents = 'none';
    }
    const pane = map.getPane(PANE)!;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.imageRendering = 'pixelated';
    canvas.style.pointerEvents = 'none';
    pane.appendChild(canvas);
    canvasRef.current = canvas;

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      pane.removeChild(canvas);
      canvasRef.current = null;
    };
  }, [map]);

  // ── Core render ──────────────────────────────────────────────────

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeData) return;

    const size = map.getSize();
    const cw: number = size.x;
    const ch: number = size.y;
    if (cw === 0 || ch === 0) return;

    // Half-resolution rendering — weather data can't benefit from more
    const scale = 0.5;
    const w = Math.max(200, Math.round(cw * scale));
    const h = Math.max(150, Math.round(ch * scale));

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    // Position canvas to cover viewport despite pane CSS transforms.
    // containerPointToLayerPoint converts container coords → layer (pane) coords.
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    canvas.style.transform = `translate(${topLeft.x}px, ${topLeft.y}px)`;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    // ── Data grid metadata ──
    const lats = activeData.lats;
    const lons = activeData.lons;
    const ny = lats.length;
    const nx = lons.length;
    if (ny < 2 || nx < 2) return;

    const latStart = lats[0];
    const latEnd = lats[ny - 1];
    const latMin = Math.min(latStart, latEnd);
    const latMax = Math.max(latStart, latEnd);
    const lonStart = lons[0];
    const lonEnd = lons[nx - 1];
    const lonMin = Math.min(lonStart, lonEnd);
    const lonMax = Math.max(lonStart, lonEnd);
    const latRange = latEnd - latStart;
    const lonRange = lonEnd - lonStart;

    // ── Data arrays ──
    const isWind = mode === 'wind' && windData;
    const uData = isWind ? (windData as WindFieldData).u : null;
    const vData = isWind ? (windData as WindFieldData).v : null;
    const scalarData = !isWind
      ? (isExtended && extendedData ? extendedData.data : (waveData ? waveData.data : null))
      : null;
    if (!uData && !scalarData) return;

    // ── Color LUT (rebuild only when mode/range changes) ──
    const [vMin, vMax] = getValueRange(mode, extendedData);
    const lutKey = `${mode}:${vMin}:${vMax}`;
    if (lutKeyRef.current !== lutKey) {
      lutRef.current = buildColorLUT(mode, vMin, vMax);
      lutKeyRef.current = lutKey;
    }
    const lut = lutRef.current!;

    // ── Pre-compute lat/lon lookup arrays (O(w+h) Mercator math) ──
    const pixelBounds = map.getPixelBounds();
    const pbMinX: number = pixelBounds.min.x;
    const pbMinY: number = pixelBounds.min.y;
    const zoom = map.getZoom();
    const mapSize = 256 * Math.pow(2, zoom);
    const invMapSize = 1 / mapSize;
    const PI = Math.PI;

    // Lat lookup: Mercator inverse projection per row
    const latForRow = new Float32Array(h);
    for (let py = 0; py < h; py++) {
      const globalY = pbMinY + (py / h) * ch;
      const latRad = Math.atan(Math.sinh(PI * (1 - 2 * globalY * invMapSize)));
      latForRow[py] = latRad * (180 / PI);
    }

    // Lon lookup: linear per column
    const lonForCol = new Float32Array(w);
    for (let px = 0; px < w; px++) {
      const globalX = pbMinX + (px / w) * cw;
      lonForCol[px] = globalX * invMapSize * 360 - 180;
    }

    // ── Pixel loop ──
    const imgData = ctx.createImageData(w, h);
    const pixels = imgData.data;
    const vRange = vMax - vMin || 1;
    const lutScale = (LUT_SIZE - 1) / vRange;
    const fadeDeg = 2;
    const invFade = 1 / fadeDeg;

    for (let py = 0; py < h; py++) {
      const lat = latForRow[py];
      if (lat < latMin || lat > latMax) continue;

      const latFracIdx = ((lat - latStart) / latRange) * (ny - 1);
      const edgeLat = Math.min((lat - latMin) * invFade, (latMax - lat) * invFade, 1);
      if (edgeLat <= 0) continue;

      const rowOff = py * w;

      for (let px = 0; px < w; px++) {
        const lon = lonForCol[px];
        if (lon < lonMin || lon > lonMax) continue;

        const lonFracIdx = ((lon - lonStart) / lonRange) * (nx - 1);

        const edgeLon = Math.min((lon - lonMin) * invFade, (lonMax - lon) * invFade, 1);
        if (edgeLon <= 0) continue;
        const edgeFade = Math.min(edgeLat, edgeLon);

        let value: number;
        if (uData && vData) {
          const u = bilinear(uData, latFracIdx, lonFracIdx, ny, nx);
          const v = bilinear(vData, latFracIdx, lonFracIdx, ny, nx);
          value = Math.sqrt(u * u + v * v);
          if (value < 0.1) continue; // land mask (backend zeros wind over land)
        } else if (scalarData) {
          value = bilinear(scalarData, latFracIdx, lonFracIdx, ny, nx);
          if (Number.isNaN(value) || value < -100) continue; // NaN / sentinel → land
        } else {
          continue;
        }

        const lutIdx = Math.min(LUT_SIZE - 1, Math.max(0, Math.round((value - vMin) * lutScale))) * 4;
        const offset = (rowOff + px) * 4;
        pixels[offset]     = lut[lutIdx];
        pixels[offset + 1] = lut[lutIdx + 1];
        pixels[offset + 2] = lut[lutIdx + 2];
        pixels[offset + 3] = Math.round(lut[lutIdx + 3] * opacity * edgeFade);
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [map, mode, activeData, windData, waveData, extendedData, isExtended, opacity]);

  // ── Trigger render on data/mode/viewport changes ─────────────────

  useEffect(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => { rafRef.current = null; render(); });
  }, [render]);

  useEffect(() => {
    const onViewChange = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(() => { rafRef.current = null; render(); });
    };
    map.on('move', onViewChange);
    map.on('zoomend', onViewChange);
    map.on('resize', onViewChange);
    return () => {
      map.off('move', onViewChange);
      map.off('zoomend', onViewChange);
      map.off('resize', onViewChange);
    };
  }, [map, render]);

  return null;
}
