'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { WindFieldData, WaveFieldData, GridFieldData, SwellFieldData } from '@/lib/api';
import { bilinearInterpolate } from '@/lib/gridInterpolation';

interface WeatherGridLayerProps {
  mode: 'wind' | 'waves' | 'swell';
  windData?: WindFieldData | null;
  waveData?: WaveFieldData | null;
  extendedData?: GridFieldData | SwellFieldData | null;
  opacity?: number;
}

// Swell period ramp for arrow coloring (seconds → color)
type ColorStop = [number, number, number, number];

const SWELL_PERIOD_RAMP: ColorStop[] = [
  [ 5, 255, 255, 255],
  [ 7, 255, 220, 120],
  [10, 240, 180,  40],
  [12,  40, 220, 240],
  [15,  20, 160, 240],
  [18,  30,  60, 180],
];

// Wind speed ramp (knots → color), matching server tile_renderer _WIND_RAMP
// Server ramp is in m/s [0,5,10,15,20,25] → converted to knots [0,~10,~19,~29,~39,~49]
const WIND_SPEED_RAMP: ColorStop[] = [
  [ 0,  80, 220, 240],
  [10,   0, 200, 220],
  [19,   0, 200,  50],
  [29, 240, 220,   0],
  [39, 240, 130,   0],
  [49, 220,  30,  30],
];

function interpolateColorRamp(
  value: number, stops: ColorStop[],
  alphaLow: number, alphaHigh: number, alphaDefault: number,
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

function swellPeriodColor(period: number): string {
  const [r, g, b] = interpolateColorRamp(period, SWELL_PERIOD_RAMP, 255, 255, 255);
  return `rgb(${r},${g},${b})`;
}

function windSpeedColor(knots: number): string {
  const [r, g, b] = interpolateColorRamp(knots, WIND_SPEED_RAMP, 255, 255, 255);
  return `rgb(${r},${g},${b})`;
}

/** Draw a WMO-standard wind barb on a canvas context.
 *  cx,cy   = center position on the canvas
 *  uMs,vMs = u/v wind components in m/s (positive east / positive north)
 *  The barb staff points INTO the wind (direction wind comes FROM).
 *
 *  WMO convention:
 *    calm  = circle             half barb = 5 kt
 *    full barb = 10 kt          pennant (flag) = 50 kt
 */
function drawWindBarb(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  uMs: number, vMs: number,
) {
  const MS_TO_KT = 1.94384;
  const speed = Math.sqrt(uMs * uMs + vMs * vMs) * MS_TO_KT;

  const color = windSpeedColor(Math.max(speed, 1));

  // ── Calm: draw circle ──
  if (speed < 2.5) {
    ctx.strokeStyle = 'rgba(0,0,0,0.5)';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }

  // Meteorological direction: angle wind is coming FROM, measured CW from north
  const fromRad = Math.atan2(-uMs, -vMs);

  const staffLen = 24;
  const barbLen = 12;
  const halfBarbLen = 6;
  const barbSpacing = 4;
  // Barbs angled ~60° from staff axis (matching WMO reference)
  const barbAngle = 60 * Math.PI / 180;
  // Pennant height along staff
  const pennantH = 6;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(fromRad);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // ── Staff (shadow + colored line) ──
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -staffLen);
  ctx.stroke();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -staffLen);
  ctx.stroke();

  // ── Decompose speed ──
  let remaining = Math.round(speed / 5) * 5;
  const pennants = Math.floor(remaining / 50);
  remaining -= pennants * 50;
  const fullBarbs = Math.floor(remaining / 10);
  remaining -= fullBarbs * 10;
  const halfBarbs = Math.floor(remaining / 5);

  let y = -staffLen; // start drawing from tip

  // Helper: draw a line segment with shadow + color
  const drawLine = (x0: number, y0: number, x1: number, y1: number) => {
    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  };

  // ── Pennants (filled triangles, 50 kt each) ──
  for (let p = 0; p < pennants; p++) {
    const bx = Math.sin(barbAngle) * barbLen;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(bx, y + pennantH * 0.5);
    ctx.lineTo(0, y + pennantH);
    ctx.closePath();
    // Shadow fill
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.fill();
    // Color fill
    ctx.fillStyle = color;
    ctx.fill();
    // Outline
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
    y += pennantH;
    // Small gap after pennants before barbs
    if (p === pennants - 1 && (fullBarbs > 0 || halfBarbs > 0)) {
      y += 2;
    }
  }

  // ── Full barbs (10 kt each) ──
  for (let f = 0; f < fullBarbs; f++) {
    const bx = Math.sin(barbAngle) * barbLen;
    const by = Math.cos(barbAngle) * barbLen;
    drawLine(0, y, bx, y + by);
    y += barbSpacing;
  }

  // ── Half barbs (5 kt each) ──
  for (let h = 0; h < halfBarbs; h++) {
    const bx = Math.sin(barbAngle) * halfBarbLen;
    const by = Math.cos(barbAngle) * halfBarbLen;
    // If only a half barb with no others, offset from tip
    if (pennants === 0 && fullBarbs === 0 && h === 0) y += barbSpacing;
    drawLine(0, y, bx, y + by);
    y += barbSpacing;
  }

  ctx.restore();
}

export default function WeatherGridLayer(props: WeatherGridLayerProps) {
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => { setIsMounted(true); }, []);
  if (!isMounted) return null;
  return <WeatherGridLayerInner {...props} />;
}

// ---------------------------------------------------------------------------
// Single-canvas pane overlay (replaces tile-based L.GridLayer).
// Draws wind barbs / wave crests / swell arrows onto one double-buffered
// canvas that covers the full viewport.  No tiles = no tile-by-tile flicker.
// ---------------------------------------------------------------------------
function WeatherGridLayerInner({
  mode,
  windData,
  waveData,
  extendedData,
  opacity = 0.7,
}: WeatherGridLayerProps) {
  const { useMap } = require('react-leaflet');
  const map = useMap();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bufferRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);

  // ── Canvas lifecycle: create once, attach to Leaflet pane ────────

  useEffect(() => {
    const PANE = 'weatherArrowPane';
    if (!map.getPane(PANE)) {
      const pane = map.createPane(PANE);
      pane.style.zIndex = '310';
      pane.style.pointerEvents = 'none';
    }
    const pane = map.getPane(PANE)!;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.pointerEvents = 'none';
    pane.appendChild(canvas);
    canvasRef.current = canvas;

    const buffer = document.createElement('canvas');
    bufferRef.current = buffer;

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      pane.removeChild(canvas);
      canvasRef.current = null;
      bufferRef.current = null;
    };
  }, [map]);

  // ── Core render (double-buffered, single canvas) ─────────────────

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const buffer = bufferRef.current;
    if (!canvas || !buffer) return;

    const data = mode === 'swell' ? extendedData
      : (mode === 'wind' ? windData : waveData);
    if (!data) return;

    const size = map.getSize();
    const cw: number = size.x;
    const ch: number = size.y;
    if (cw === 0 || ch === 0) return;

    // Resize offscreen buffer (visible canvas stays untouched until blit)
    if (buffer.width !== cw || buffer.height !== ch) {
      buffer.width = cw;
      buffer.height = ch;
    }
    const ctx = buffer.getContext('2d')!;
    ctx.clearRect(0, 0, cw, ch);

    // ── Grid metadata ──
    const lats = data.lats;
    const lons = data.lons;
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

    const oceanMask = data.ocean_mask;
    const maskLats = data.ocean_mask_lats || lats;
    const maskLons = data.ocean_mask_lons || lons;
    const maskNy = maskLats.length;
    const maskNx = maskLons.length;

    // ── Viewport → geographic coordinate helpers ──
    const pixelBounds = map.getPixelBounds();
    const pbMinX: number = pixelBounds.min.x;
    const pbMinY: number = pixelBounds.min.y;
    const zoom = map.getZoom();
    const mapSize = 256 * Math.pow(2, zoom);
    const PI = Math.PI;

    function screenToGeo(ax: number, ay: number): [number, number] {
      const globalX = pbMinX + ax;
      const globalY = pbMinY + ay;
      const lng = (globalX / mapSize) * 360 - 180;
      const latRad = Math.atan(Math.sinh(PI * (1 - 2 * globalY / mapSize)));
      return [(latRad * 180) / PI, lng];
    }

    function checkOcean(lat: number, lng: number): boolean {
      if (!oceanMask) return true;
      const mLatRange = maskLats[maskNy - 1] - maskLats[0];
      const mLonRange = maskLons[maskNx - 1] - maskLons[0];
      if (mLatRange === 0 || mLonRange === 0) return true;
      const mfi = ((lat - maskLats[0]) / mLatRange) * (maskNy - 1);
      const mfj = ((lng - maskLons[0]) / mLonRange) * (maskNx - 1);
      if (mfi < 0 || mfi > maskNy - 1 || mfj < 0 || mfj > maskNx - 1) return true;
      const mi0 = Math.min(Math.floor(mfi), maskNy - 1);
      const mi1 = Math.min(mi0 + 1, maskNy - 1);
      const mj0 = Math.min(Math.floor(mfj), maskNx - 1);
      const mj1 = Math.min(mj0 + 1, maskNx - 1);
      const mf = mfi - mi0;
      const mc = mfj - mj0;
      const oceanFrac =
        (oceanMask[mi0]?.[mj0] ? 1 : 0) * (1 - mf) * (1 - mc) +
        (oceanMask[mi0]?.[mj1] ? 1 : 0) * (1 - mf) * mc +
        (oceanMask[mi1]?.[mj0] ? 1 : 0) * mf * (1 - mc) +
        (oceanMask[mi1]?.[mj1] ? 1 : 0) * mf * mc;
      return oceanFrac >= 0.5;
    }

    // ── Wave direction crest marks (Windy-style arcs) ──
    if (mode === 'waves' && waveData) {
      const waveW = waveData as any;
      const swellDir = waveW?.swell?.direction as number[][] | undefined;
      const swellHt = waveW?.swell?.height as number[][] | undefined;
      const wwDir = waveW?.windwave?.direction as number[][] | undefined;
      const wwHt = waveW?.windwave?.height as number[][] | undefined;
      const waveValues = waveW?.data as number[][] | undefined;
      const waveDir = waveW?.direction as number[][] | undefined;
      const hasSwell = swellDir && swellHt;
      const hasWW = wwDir && wwHt;

      const spacing = 30;
      ctx.save();

      const drawWaveCrest = (
        cx: number, cy: number, dirDeg: number, height: number,
        color: string, alpha: number,
      ) => {
        const propRad = ((dirDeg + 90) * PI) / 180;
        const perpRad = propRad + PI / 2;
        const arcLen = Math.min(16, 6 + height * 4);
        const curve = Math.min(5, 2 + height * 1.0);
        const crestGap = 4.5;
        ctx.lineCap = 'round';

        for (let k = -1; k <= 1; k++) {
          const ox = cx + Math.cos(propRad) * k * crestGap;
          const oy = cy + Math.sin(propRad) * k * crestGap;
          const sc = k === 0 ? 1.0 : 0.7;
          const halfLen = arcLen * sc * 0.5;
          const x0 = ox - Math.cos(perpRad) * halfLen;
          const y0 = oy - Math.sin(perpRad) * halfLen;
          const x1 = ox + Math.cos(perpRad) * halfLen;
          const y1 = oy + Math.sin(perpRad) * halfLen;
          const cpx = ox + Math.cos(propRad) * curve * sc;
          const cpy = oy + Math.sin(propRad) * curve * sc;

          ctx.strokeStyle = 'rgba(0,0,0,0.3)';
          ctx.globalAlpha = 1.0;
          ctx.lineWidth = (k === 0 ? 2.0 : 1.2) + 1.0;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.quadraticCurveTo(cpx, cpy, x1, y1); ctx.stroke();

          ctx.strokeStyle = color;
          ctx.globalAlpha = alpha * (k === 0 ? 1.0 : 0.6);
          ctx.lineWidth = k === 0 ? 2.0 : 1.2;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.quadraticCurveTo(cpx, cpy, x1, y1); ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
      };

      for (let ay = spacing / 2; ay < ch; ay += spacing) {
        for (let ax = spacing / 2; ax < cw; ax += spacing) {
          const [aLat, lng] = screenToGeo(ax, ay);
          if (aLat < latMin || aLat > latMax || lng < lonMin || lng > lonMax) continue;
          if (!checkOcean(aLat, lng)) continue;

          const latFracIdx = ((aLat - latStart) / (latEnd - latStart)) * (ny - 1);
          const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
          const aLatIdx = Math.floor(latFracIdx);
          const aLonIdx = Math.floor(lonFracIdx);
          const aLatFrac = latFracIdx - aLatIdx;
          const aLonFrac = lonFracIdx - aLonIdx;

          if (hasSwell) {
            const sh = bilinearInterpolate(swellHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            if (sh > 0.2) {
              const sd = bilinearInterpolate(swellDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              drawWaveCrest(ax, ay, sd, sh, 'rgba(255,255,255,1)', Math.min(0.95, 0.5 + sh * 0.15));
            }
          }

          if (hasWW) {
            const wh = bilinearInterpolate(wwHt, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            if (wh > 0.2) {
              const wd = bilinearInterpolate(wwDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              drawWaveCrest(ax + 6, ay + 6, wd, wh * 0.8, 'rgba(200,230,255,1)', Math.min(0.8, 0.4 + wh * 0.12));
            }
          }

          if (!hasSwell && !hasWW && waveDir && waveValues) {
            const h = bilinearInterpolate(waveValues, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            if (h > 0.3) {
              const d = bilinearInterpolate(waveDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              drawWaveCrest(ax, ay, d, h, 'rgba(255,255,255,1)', Math.min(0.9, 0.5 + h * 0.15));
            }
          }
        }
      }
      ctx.restore();
    }

    // ── Wind barbs (WMO standard: half=5kt, full=10kt, pennant=50kt) ──
    if (mode === 'wind' && windData) {
      const uGrid = windData.u;
      const vGrid = windData.v;
      if (uGrid && vGrid) {
        const spacing = 36;
        ctx.save();

        for (let ay = spacing / 2; ay < ch; ay += spacing) {
          for (let ax = spacing / 2; ax < cw; ax += spacing) {
            const [aLat, lng] = screenToGeo(ax, ay);
            if (aLat < latMin || aLat > latMax || lng < lonMin || lng > lonMax) continue;
            if (!checkOcean(aLat, lng)) continue;

            const latFracIdx = ((aLat - latStart) / (latEnd - latStart)) * (ny - 1);
            const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
            const aLatIdx = Math.floor(latFracIdx);
            const aLonIdx = Math.floor(lonFracIdx);
            const aLatFrac = latFracIdx - aLatIdx;
            const aLonFrac = lonFracIdx - aLonIdx;

            const u = bilinearInterpolate(uGrid, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            const v = bilinearInterpolate(vGrid, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);

            drawWindBarb(ctx, ax, ay, u, v);
          }
        }
        ctx.restore();
      }
    }

    // ── Swell directional arrows (period-colored, height-scaled) ──
    if (mode === 'swell' && extendedData) {
      const swellExt = extendedData as SwellFieldData;
      const swDir = swellExt.swell_dir;
      const swHs = swellExt.swell_hs;
      const swTp = swellExt.swell_tp;

      if (swDir && swHs) {
        const spacing = 40;
        ctx.save();

        for (let ay = spacing / 2; ay < ch; ay += spacing) {
          for (let ax = spacing / 2; ax < cw; ax += spacing) {
            const [aLat, lng] = screenToGeo(ax, ay);
            if (aLat < latMin || aLat > latMax || lng < lonMin || lng > lonMax) continue;
            if (!checkOcean(aLat, lng)) continue;

            const latFracIdx = ((aLat - latStart) / (latEnd - latStart)) * (ny - 1);
            const lonFracIdx = ((lng - lonStart) / (lonEnd - lonStart)) * (nx - 1);
            const aLatIdx = Math.floor(latFracIdx);
            const aLonIdx = Math.floor(lonFracIdx);
            const aLatFrac = latFracIdx - aLatIdx;
            const aLonFrac = lonFracIdx - aLonIdx;

            const hs = bilinearInterpolate(swHs, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            if (hs < 0.3) continue;

            const dir = bilinearInterpolate(swDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
            const tp = swTp ? bilinearInterpolate(swTp, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx) : 10;

            const propRad = ((dir + 180) * PI) / 180;
            const len = Math.min(20, 10 + hs * 2.5);
            const color = swellPeriodColor(tp);

            ctx.translate(ax, ay);
            ctx.rotate(propRad);

            ctx.globalAlpha = 0.6;
            ctx.strokeStyle = 'rgba(0,0,0,0.8)';
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            ctx.beginPath(); ctx.moveTo(0, len / 2); ctx.lineTo(0, -len / 2); ctx.stroke();

            ctx.globalAlpha = 0.9;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.8;
            ctx.beginPath(); ctx.moveTo(0, len / 2); ctx.lineTo(0, -len / 2); ctx.stroke();

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

    // ── Atomic blit: offscreen buffer → visible canvas ──
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    const topLeft = map.containerPointToLayerPoint([0, 0]);
    canvas.style.transform = `translate(${topLeft.x}px, ${topLeft.y}px)`;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    canvas.style.opacity = String(opacity);

    const visCtx = canvas.getContext('2d');
    if (!visCtx) return;
    visCtx.clearRect(0, 0, cw, ch);
    visCtx.drawImage(buffer, 0, 0);
  }, [map, mode, windData, waveData, extendedData, opacity]);

  // ── Data change: render synchronously (no rAF delay) ─────────────

  useEffect(() => {
    render();
  }, [render]);

  // ── Viewport change: debounced rAF (pan/zoom) ───────────────────

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
