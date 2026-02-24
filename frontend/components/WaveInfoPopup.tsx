'use client';

import { useEffect, useRef } from 'react';
import { WindFieldData, WaveFieldData, SwellFieldData, VelocityData, GridFieldData } from '@/lib/api';
import { bilinearInterpolate, getGridIndices } from '@/lib/gridInterpolation';

type ActiveLayer = 'waves' | 'swell' | 'wind' | 'currents' | 'visibility';

interface WeatherHoverTooltipProps {
  layer: ActiveLayer;
  windData: WindFieldData | null;
  waveData: WaveFieldData | null;
  swellData: SwellFieldData | null;
  currentVelocityData: VelocityData[] | null;
  visibilityData: GridFieldData | null;
}

/** 16-point compass label */
function dirLabel(deg: number): string {
  const d = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'];
  return d[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16];
}

/** Beaufort scale from m/s */
function beaufort(ms: number): number {
  const thresholds = [0.5,1.6,3.4,5.5,8.0,10.8,13.9,17.2,20.8,24.5,28.5,32.7];
  for (let i = 0; i < thresholds.length; i++) { if (ms < thresholds[i]) return i; }
  return 12;
}

/** Bilinear interpolation on a GRIB-JSON VelocityData flat grid.
 *  Returns { u, v } in the grid's native units (m/s) or null if out of bounds. */
function interpVelocity(vd: VelocityData[], lat: number, lon: number): { u: number; v: number } | null {
  if (!vd || vd.length < 2) return null;
  const hdr = vd[0].header;
  const { la1, lo1, dx, dy, nx, ny } = hdr;

  // Normalise longitude into the grid's range
  let glon = lon;
  if (glon < lo1) glon += 360;
  if (glon > lo1 + (nx - 1) * dx) return null;

  // Grid indices (la1 = north, rows go south)
  const fi = (la1 - lat) / dy;
  const fj = (glon - lo1) / dx;
  if (fi < 0 || fi >= ny - 1 || fj < 0 || fj >= nx - 1) return null;

  const i0 = Math.floor(fi), j0 = Math.floor(fj);
  const ft = fi - i0, fs = fj - j0;

  const idx = (r: number, c: number) => r * nx + c;
  const bilinear = (d: number[]) => {
    const v00 = d[idx(i0, j0)],     v01 = d[idx(i0, j0 + 1)];
    const v10 = d[idx(i0 + 1, j0)], v11 = d[idx(i0 + 1, j0 + 1)];
    return (1 - ft) * ((1 - fs) * v00 + fs * v01) + ft * ((1 - fs) * v10 + fs * v11);
  };

  return { u: bilinear(vd[0].data), v: bilinear(vd[1].data) };
}

/** Compute met "from" direction and speed from u,v components */
function uvToSpeedDir(u: number, v: number): { speed: number; dir: number } {
  const speed = Math.sqrt(u * u + v * v);
  const dir = ((270 - Math.atan2(v, u) * 180 / Math.PI) % 360 + 360) % 360;
  return { speed, dir };
}

/**
 * Hover tooltip for wind, waves, swell, and currents layers.
 *
 * Uses direct DOM manipulation on Leaflet's mousemove — no React re-renders,
 * throttled to ~16 fps. Content adapts to the active layer.
 */
export default function WaveInfoPopup({
  layer, windData, waveData, swellData, currentVelocityData, visibilityData,
}: WeatherHoverTooltipProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();

  const layerRef = useRef(layer);
  const windRef  = useRef(windData);
  const waveRef  = useRef(waveData);
  const swellRef = useRef(swellData);
  const curRef   = useRef(currentVelocityData);
  const visRef   = useRef(visibilityData);
  layerRef.current = layer;
  windRef.current  = windData;
  waveRef.current  = waveData;
  swellRef.current = swellData;
  curRef.current   = currentVelocityData;
  visRef.current   = visibilityData;

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

      const mode = layerRef.current;
      const { lat, lng: lon } = e.latlng;
      const pt: { x: number; y: number } = e.containerPoint;

      // ------------------------------------------------------------------
      // Helper: interpolate wind from WindFieldData (u/v 2D grids)
      // ------------------------------------------------------------------
      const interpWind = (): { speed: number; dir: number } | null => {
        const w = windRef.current;
        if (!w) return null;
        const gi = getGridIndices(lat, lon, w.lats, w.lons);
        if (!gi) return null;
        const u = bilinearInterpolate(w.u, gi.latIdx, gi.lonIdx, gi.latFrac, gi.lonFrac, w.lats.length, w.lons.length);
        const v = bilinearInterpolate(w.v, gi.latIdx, gi.lonIdx, gi.latFrac, gi.lonFrac, w.lats.length, w.lons.length);
        return uvToSpeedDir(u, v);
      };

      // ------------------------------------------------------------------
      // WIND layer
      // ------------------------------------------------------------------
      if (mode === 'wind') {
        const w = windRef.current;
        if (!w) { tip.style.display = 'none'; return; }
        // Ocean mask
        const mask = w.ocean_mask;
        if (mask) {
          const mLats = w.ocean_mask_lats || w.lats;
          const mLons = w.ocean_mask_lons || w.lons;
          const mNy = mLats.length, mNx = mLons.length;
          const mi = Math.round(((lat - mLats[0]) / (mLats[mNy - 1] - mLats[0])) * (mNy - 1));
          const mj = Math.round(((lon - mLons[0]) / (mLons[mNx - 1] - mLons[0])) * (mNx - 1));
          if (mi < 0 || mi >= mNy || mj < 0 || mj >= mNx || !mask[mi]?.[mj]) {
            tip.style.display = 'none'; return;
          }
        }
        const r = interpWind();
        if (!r || r.speed < 0.3) { tip.style.display = 'none'; return; }
        const bf = beaufort(r.speed);
        const kts = (r.speed * 1.944).toFixed(1);
        let h = `<div style="font-weight:700;font-size:12px;color:#fff;margin-bottom:2px">`
              + `Wind ${r.speed.toFixed(1)} m/s &nbsp;(${kts} kts)</div>`;
        h += `<div style="color:#60a5fa">${dirLabel(r.dir)} ${r.dir.toFixed(0)}° &nbsp;&middot;&nbsp; Beaufort ${bf}</div>`;
        tip.innerHTML = h;
      }

      // ------------------------------------------------------------------
      // CURRENTS layer
      // ------------------------------------------------------------------
      else if (mode === 'currents') {
        const cv = curRef.current;
        if (!cv || cv.length < 2) { tip.style.display = 'none'; return; }
        const r = interpVelocity(cv, lat, lon);
        if (!r) { tip.style.display = 'none'; return; }
        const { speed, dir } = uvToSpeedDir(r.u, r.v);
        if (speed < 0.01) { tip.style.display = 'none'; return; }
        const kts = (speed * 1.944).toFixed(2);
        // Current direction is "toward" (oceanographic convention) — show as "set"
        const setDir = ((dir + 180) % 360);
        let h = `<div style="font-weight:700;font-size:12px;color:#fff;margin-bottom:2px">`
              + `Current ${speed.toFixed(2)} m/s &nbsp;(${kts} kts)</div>`;
        h += `<div style="color:#22d3ee">Set ${dirLabel(setDir)} ${setDir.toFixed(0)}° &nbsp;&middot;&nbsp; Drift ${(speed * 1.944).toFixed(2)} kts</div>`;
        tip.innerHTML = h;
      }

      // ------------------------------------------------------------------
      // VISIBILITY layer
      // ------------------------------------------------------------------
      else if (mode === 'visibility') {
        const vd = visRef.current;
        if (!vd || !vd.data) { tip.style.display = 'none'; return; }
        // Ocean mask
        const mask = vd.ocean_mask;
        if (mask) {
          const mLats = vd.ocean_mask_lats || vd.lats;
          const mLons = vd.ocean_mask_lons || vd.lons;
          const mNy = mLats.length, mNx = mLons.length;
          const mi = Math.round(((lat - mLats[0]) / (mLats[mNy - 1] - mLats[0])) * (mNy - 1));
          const mj = Math.round(((lon - mLons[0]) / (mLons[mNx - 1] - mLons[0])) * (mNx - 1));
          if (mi < 0 || mi >= mNy || mj < 0 || mj >= mNx || !mask[mi]?.[mj]) {
            tip.style.display = 'none'; return;
          }
        }
        const gi = getGridIndices(lat, lon, vd.lats, vd.lons);
        if (!gi) { tip.style.display = 'none'; return; }
        const val = bilinearInterpolate(vd.data, gi.latIdx, gi.lonIdx, gi.latFrac, gi.lonFrac, vd.lats.length, vd.lons.length);
        if (val == null) { tip.style.display = 'none'; return; }
        // val is in km; convert to m and nm
        const km = val;
        const nm = km * 0.53996;
        // Fog classification
        let fogLabel = '';
        let fogColor = '#a3e635'; // good vis
        if (km < 1) { fogLabel = 'Dense fog'; fogColor = '#ef4444'; }
        else if (km < 2) { fogLabel = 'Fog'; fogColor = '#f97316'; }
        else if (km < 5) { fogLabel = 'Mist'; fogColor = '#facc15'; }
        else if (km < 10) { fogLabel = 'Haze'; fogColor = '#94a3b8'; }
        else { fogLabel = 'Clear'; fogColor = '#a3e635'; }
        let h = `<div style="font-weight:700;font-size:12px;color:#fff;margin-bottom:2px">`
              + `Visibility ${km.toFixed(1)} km &nbsp;(${nm.toFixed(1)} nm)</div>`;
        h += `<div style="color:${fogColor}">${fogLabel}</div>`;
        tip.innerHTML = h;
      }

      // ------------------------------------------------------------------
      // WAVES / SWELL layers
      // ------------------------------------------------------------------
      else {
        const wd  = (mode === 'waves')  ? waveRef.current  : null;
        const swd = (mode === 'swell')  ? swellRef.current : null;
        const grid = wd || swd;
        if (!grid) { tip.style.display = 'none'; return; }

        // Ocean mask
        const mask  = grid.ocean_mask;
        const mLats = grid.ocean_mask_lats || grid.lats;
        const mLons = grid.ocean_mask_lons || grid.lons;
        if (mask) {
          const mNy = mLats.length, mNx = mLons.length;
          const mi = Math.round(((lat - mLats[0]) / (mLats[mNy - 1] - mLats[0])) * (mNy - 1));
          const mj = Math.round(((lon - mLons[0]) / (mLons[mNx - 1] - mLons[0])) * (mNx - 1));
          if (mi < 0 || mi >= mNy || mj < 0 || mj >= mNx || !mask[mi]?.[mj]) {
            tip.style.display = 'none'; return;
          }
        }

        const ny = grid.lats.length, nx = grid.lons.length;
        const gi = getGridIndices(lat, lon, grid.lats, grid.lons);
        if (!gi) { tip.style.display = 'none'; return; }
        const { latIdx, lonIdx, latFrac, lonFrac } = gi;
        const interp = (g: number[][] | null | undefined) =>
          g ? bilinearInterpolate(g, latIdx, lonIdx, latFrac, lonFrac, ny, nx) : null;

        let hs: number, dir: number | null;
        let swH: number | null, swD: number | null, swT: number | null;
        let wwH: number | null, wwD: number | null, wwT: number | null;

        if (wd) {
          hs  = interp(wd.data) ?? 0;
          dir = interp(wd.direction);
          swH = interp(wd.swell?.height);  swD = interp(wd.swell?.direction);  swT = interp(wd.swell?.period);
          wwH = interp(wd.windwave?.height); wwD = interp(wd.windwave?.direction); wwT = interp(wd.windwave?.period);
        } else {
          const s = swd!;
          hs  = interp(s.total_hs) ?? interp(s.data) ?? 0;
          dir = null;
          swH = interp(s.swell_hs);  swD = interp(s.swell_dir);  swT = interp(s.swell_tp);
          wwH = interp(s.windsea_hs); wwD = interp(s.windsea_dir); wwT = interp(s.windsea_tp);
        }

        let h = `<div style="font-weight:700;font-size:12px;color:#fff;margin-bottom:2px">Hs ${hs.toFixed(1)} m`;
        if (dir != null) h += ` &nbsp;${dirLabel(dir)} ${dir.toFixed(0)}°`;
        h += '</div>';

        const hasDecomp = (swH != null && swD != null) || (wwH != null && wwD != null);
        if (swH != null && swD != null) {
          h += `<div style="color:#f59e0b"><b>Swell</b> ${swH.toFixed(1)} m`
             + (swT != null ? ` &middot; ${swT.toFixed(0)} s` : '')
             + ` &nbsp;${dirLabel(swD)} ${swD.toFixed(0)}°</div>`;
        }
        if (wwH != null && wwD != null) {
          h += `<div style="color:#4ade80"><b>Wind sea</b> ${wwH.toFixed(1)} m`
             + (wwT != null ? ` &middot; ${wwT.toFixed(0)} s` : '')
             + ` &nbsp;${dirLabel(wwD)} ${wwD.toFixed(0)}°</div>`;
        }
        if (!hasDecomp && dir != null) {
          h += `<div style="color:#94a3b8">Direction ${dirLabel(dir)} ${dir.toFixed(0)}°</div>`;
        }

        // Wind secondary row
        const wr = interpWind();
        if (wr && wr.speed > 0.5) {
          h += `<div style="color:#60a5fa;border-top:1px solid rgba(255,255,255,0.08);margin-top:2px;padding-top:2px">`
             + `<b>Wind</b> ${wr.speed.toFixed(1)} m/s &nbsp;${dirLabel(wr.dir)} ${wr.dir.toFixed(0)}°</div>`;
        }
        tip.innerHTML = h;
      }

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
