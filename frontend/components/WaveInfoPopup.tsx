'use client';

import { useEffect, useRef } from 'react';
import { WindFieldData, WaveFieldData, SwellFieldData } from '@/lib/api';
import { bilinearInterpolate, getGridIndices } from '@/lib/gridInterpolation';

interface WaveInfoPopupProps {
  active: boolean;
  waveData: WaveFieldData | null;
  windData: WindFieldData | null;
  /** Swell extended data — used when weatherLayer === 'swell' */
  swellData: SwellFieldData | null;
}

/** 16-point compass label */
function dirLabel(deg: number): string {
  const d = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return d[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
}

/**
 * Hover tooltip for the waves and swell layers.
 *
 * Uses direct DOM manipulation on Leaflet's mousemove for performance —
 * no React state updates per frame, throttled to ~16 fps.
 */
export default function WaveInfoPopup({ active, waveData, windData, swellData }: WaveInfoPopupProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();

  const waveRef  = useRef(waveData);
  const windRef  = useRef(windData);
  const swellRef = useRef(swellData);
  const activeRef = useRef(active);
  waveRef.current  = waveData;
  windRef.current  = windData;
  swellRef.current = swellData;
  activeRef.current = active;

  useEffect(() => {
    const container = map.getContainer();

    const tip = L.DomUtil.create('div', '', container) as HTMLDivElement;
    tip.style.cssText = [
      'position:absolute',
      'pointer-events:none',
      'z-index:1000',
      'display:none',
      'background:rgba(15,23,42,0.92)',
      'border:1px solid rgba(255,255,255,0.12)',
      'border-radius:6px',
      'padding:6px 10px',
      'font-family:system-ui,-apple-system,sans-serif',
      'font-size:11px',
      'line-height:1.45',
      'color:#e2e8f0',
      'white-space:nowrap',
      'backdrop-filter:blur(4px)',
      'box-shadow:0 2px 8px rgba(0,0,0,0.45)',
      'transition:left .04s,top .04s',
    ].join(';');

    let last = 0;

    const onMove = (e: any) => {
      const now = performance.now();
      if (now - last < 60) return;
      last = now;

      if (!activeRef.current) { tip.style.display = 'none'; return; }

      const wd  = waveRef.current;
      const swd = swellRef.current;
      // Need at least one data source
      const grid = wd || swd;
      if (!grid) { tip.style.display = 'none'; return; }

      const { lat, lng: lon } = e.latlng;
      const pt: { x: number; y: number } = e.containerPoint;

      // --- ocean mask check (use whichever grid is active) ---
      const mask  = grid.ocean_mask;
      const mLats = grid.ocean_mask_lats || grid.lats;
      const mLons = grid.ocean_mask_lons || grid.lons;
      if (mask) {
        const mNy = mLats.length;
        const mNx = mLons.length;
        const mi = Math.round(((lat - mLats[0]) / (mLats[mNy - 1] - mLats[0])) * (mNy - 1));
        const mj = Math.round(((lon - mLons[0]) / (mLons[mNx - 1] - mLons[0])) * (mNx - 1));
        if (mi < 0 || mi >= mNy || mj < 0 || mj >= mNx || !mask[mi]?.[mj]) {
          tip.style.display = 'none';
          return;
        }
      }

      // --- grid interpolation helper ---
      const ny = grid.lats.length;
      const nx = grid.lons.length;
      const gi = getGridIndices(lat, lon, grid.lats, grid.lons);
      if (!gi) { tip.style.display = 'none'; return; }

      const { latIdx, lonIdx, latFrac, lonFrac } = gi;
      const interp = (g: number[][] | null | undefined) =>
        g ? bilinearInterpolate(g, latIdx, lonIdx, latFrac, lonFrac, ny, nx) : null;

      // --- extract values from whichever data source is active ---
      let hs: number;
      let dir: number | null;
      let swH: number | null, swD: number | null, swT: number | null;
      let wwH: number | null, wwD: number | null, wwT: number | null;

      if (wd) {
        // Waves layer
        hs  = interp(wd.data) ?? 0;
        dir = interp(wd.direction);
        swH = interp(wd.swell?.height);
        swD = interp(wd.swell?.direction);
        swT = interp(wd.swell?.period);
        wwH = interp(wd.windwave?.height);
        wwD = interp(wd.windwave?.direction);
        wwT = interp(wd.windwave?.period);
      } else {
        // Swell layer (SwellFieldData)
        const s = swd!;
        hs  = interp(s.total_hs) ?? interp(s.data) ?? 0;
        dir = null; // swell layer has no total direction field
        swH = interp(s.swell_hs);
        swD = interp(s.swell_dir);
        swT = interp(s.swell_tp);
        wwH = interp(s.windsea_hs);
        wwD = interp(s.windsea_dir);
        wwT = interp(s.windsea_tp);
      }

      // --- build HTML ---
      let h = '<div style="font-weight:700;font-size:12px;color:#fff;margin-bottom:2px">'
            + `Hs ${hs.toFixed(1)} m`;
      if (dir != null) h += ` &nbsp;${dirLabel(dir)} ${dir.toFixed(0)}°`;
      h += '</div>';

      const hasDecomp = (swH != null && swD != null) || (wwH != null && wwD != null);

      if (swH != null && swD != null) {
        h += '<div style="color:#f59e0b">'
           + `<b>Swell</b> ${swH.toFixed(1)} m`
           + (swT != null ? ` &middot; ${swT.toFixed(0)} s` : '')
           + ` &nbsp;${dirLabel(swD)} ${swD.toFixed(0)}°</div>`;
      }
      if (wwH != null && wwD != null) {
        h += '<div style="color:#4ade80">'
           + `<b>Wind sea</b> ${wwH.toFixed(1)} m`
           + (wwT != null ? ` &middot; ${wwT.toFixed(0)} s` : '')
           + ` &nbsp;${dirLabel(wwD)} ${wwD.toFixed(0)}°</div>`;
      }

      if (!hasDecomp && dir != null) {
        h += `<div style="color:#94a3b8">Direction ${dirLabel(dir)} ${dir.toFixed(0)}°</div>`;
      }

      // Wind (if available)
      const wnd = windRef.current;
      if (wnd) {
        const wny = wnd.lats.length;
        const wnx = wnd.lons.length;
        const wgi = getGridIndices(lat, lon, wnd.lats, wnd.lons);
        if (wgi) {
          const u = bilinearInterpolate(wnd.u, wgi.latIdx, wgi.lonIdx, wgi.latFrac, wgi.lonFrac, wny, wnx);
          const v = bilinearInterpolate(wnd.v, wgi.latIdx, wgi.lonIdx, wgi.latFrac, wgi.lonFrac, wny, wnx);
          const spd = Math.sqrt(u * u + v * v);
          if (spd > 0.5) {
            const wd2 = ((270 - Math.atan2(v, u) * 180 / Math.PI) % 360 + 360) % 360;
            h += `<div style="color:#60a5fa;border-top:1px solid rgba(255,255,255,0.08);margin-top:2px;padding-top:2px">`
               + `<b>Wind</b> ${spd.toFixed(1)} m/s &nbsp;${dirLabel(wd2)} ${wd2.toFixed(0)}°</div>`;
          }
        }
      }

      tip.innerHTML = h;
      tip.style.display = 'block';

      // Position: above-right of cursor, flip if near edge
      const cw = container.clientWidth;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      let tx = pt.x + 18;
      let ty = pt.y - th - 10;
      if (tx + tw > cw - 4) tx = pt.x - tw - 18;
      if (ty < 4) ty = pt.y + 18;
      tip.style.left = tx + 'px';
      tip.style.top  = ty + 'px';
    };

    const onOut = () => { tip.style.display = 'none'; };

    map.on('mousemove', onMove);
    map.on('mouseout', onOut);

    return () => {
      map.off('mousemove', onMove);
      map.off('mouseout', onOut);
      tip.remove();
    };
  }, [map, L]);

  return null;
}
