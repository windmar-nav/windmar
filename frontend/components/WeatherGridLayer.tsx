'use client';

import { useEffect, useRef, useState } from 'react';
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

export default function WeatherGridLayer(props: WeatherGridLayerProps) {
  const [isMounted, setIsMounted] = useState(false);
  useEffect(() => { setIsMounted(true); }, []);
  if (!isMounted) return null;
  return <WeatherGridLayerInner {...props} />;
}

function WeatherGridLayerInner({
  mode,
  windData,
  waveData,
  extendedData,
  opacity = 0.7,
}: WeatherGridLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);

  const windDataRef = useRef(windData);
  const waveDataRef = useRef(waveData);
  const extendedDataRef = useRef(extendedData);
  useEffect(() => { windDataRef.current = windData; }, [windData]);
  useEffect(() => { waveDataRef.current = waveData; }, [waveData]);
  useEffect(() => { extendedDataRef.current = extendedData; }, [extendedData]);

  useEffect(() => {
    const currentMode = mode;
    const tileSize = 256;

    function paintTile(tile: HTMLCanvasElement, coords: any) {
      const currentWindData = windDataRef.current;
      const currentWaveData = waveDataRef.current;
      const currentExtendedData = extendedDataRef.current;
      const data = currentMode === 'swell' ? currentExtendedData
        : (currentMode === 'wind' ? currentWindData : currentWaveData);

      const ctx = tile.getContext('2d');
      if (!ctx || !data) { if (ctx) ctx.clearRect(0, 0, 256, 256); return; }

      const lats = data.lats;
      const lons = data.lons;
      const ny = lats.length;
      const nx = lons.length;
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

      const zoom = coords.z;
      ctx.clearRect(0, 0, 256, 256);

      // ── Wave direction crest marks (Windy-style arcs) ──
      if (currentMode === 'waves') {
        const waveW = currentWaveData as any;
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
          const propRad = ((dirDeg + 90) * Math.PI) / 180;
          const perpRad = propRad + Math.PI / 2;
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
              if (mLatIdx >= 0 && mLatIdx < maskNy && mLonIdx >= 0 && mLonIdx < maskNx && !oceanMask[mLatIdx]?.[mLonIdx]) continue;
            }

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

      // ── Swell directional arrows (period-colored, height-scaled) ──
      if (currentMode === 'swell' && currentExtendedData) {
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
                if (mLatIdx >= 0 && mLatIdx < maskNy && mLonIdx >= 0 && mLonIdx < maskNx && !oceanMask[mLatIdx]?.[mLonIdx]) continue;
              }

              const hs = bilinearInterpolate(swHs, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              if (hs < 0.3) continue;

              const dir = bilinearInterpolate(swDir, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx);
              const tp = swTp ? bilinearInterpolate(swTp, aLatIdx, aLonIdx, aLatFrac, aLonFrac, ny, nx) : 10;

              const propRad = ((dir + 180) * Math.PI) / 180;
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
    }

    const ArrowTileLayer = L.GridLayer.extend({
      createTile(coords: any) {
        const tile = document.createElement('canvas') as HTMLCanvasElement;
        tile.width = 256;
        tile.height = 256;
        paintTile(tile, coords);
        return tile;
      },
      refreshTiles() {
        const tiles = this._tiles;
        if (!tiles) return;
        for (const key in tiles) {
          const t = tiles[key];
          if (t.el && t.coords) paintTile(t.el as HTMLCanvasElement, t.coords);
        }
      },
    });

    const ARROW_PANE = 'weatherArrowPane';
    if (!map.getPane(ARROW_PANE)) {
      const pane = map.createPane(ARROW_PANE);
      pane.style.zIndex = '310';
      pane.style.pointerEvents = 'none';
    }

    const layer = new ArrowTileLayer({
      opacity,
      tileSize: 256,
      updateWhenZooming: false,
      updateWhenIdle: true,
      pane: ARROW_PANE,
    });

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [mode, opacity, map, L]);

  // Repaint arrows when data changes
  useEffect(() => {
    if (layerRef.current?.refreshTiles) {
      requestAnimationFrame(() => { layerRef.current?.refreshTiles(); });
    }
  }, [windData, waveData, extendedData]);

  return null;
}
