'use client';

import { useEffect, useRef, useState } from 'react';
import { VelocityData } from '@/lib/api';

interface VelocityParticleLayerProps {
  data: VelocityData[] | null;
  type: 'wind' | 'currents';
}

const WIND_COLOR_SCALE = [
  'rgb(36,104,180)',
  'rgb(60,157,194)',
  'rgb(128,205,193)',
  'rgb(198,231,181)',
  'rgb(255,238,159)',
  'rgb(255,182,100)',
  'rgb(252,150,75)',
  'rgb(250,112,52)',
  'rgb(245,64,32)',
  'rgb(237,45,28)',
  'rgb(220,24,32)',
  'rgb(180,7,23)',
];

const CURRENT_COLOR_SCALE = [
  'rgb(36,104,180)',
  'rgb(24,176,200)',
  'rgb(60,200,180)',
  'rgb(100,210,160)',
  'rgb(180,230,120)',
  'rgb(240,210,80)',
  'rgb(250,180,60)',
  'rgb(250,130,30)',
];

/** Compute particle density multiplier from zoom level.
 *  Wind: sparse Windy-style. Currents: denser since particles move slowly. */
function getParticleMultiplier(zoom: number, isWind: boolean): number {
  if (isWind) {
    if (zoom <= 3) return 1 / 1200;
    if (zoom <= 4) return 1 / 800;
    if (zoom === 5) return 1 / 500;
    if (zoom === 6) return 1 / 300;
    if (zoom <= 8) return 1 / 150;
    return 1 / 80;
  }
  // Currents — ~4x denser than wind (slow-moving particles need more density)
  if (zoom <= 3) return 1 / 300;
  if (zoom <= 4) return 1 / 200;
  if (zoom === 5) return 1 / 120;
  if (zoom === 6) return 1 / 70;
  if (zoom <= 8) return 1 / 40;
  return 1 / 20;
}

export default function VelocityParticleLayer(props: VelocityParticleLayerProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) return null;

  return <VelocityParticleLayerInner {...props} />;
}

// ---------------------------------------------------------------------------
// Architecture: Two separate effects following the same pattern as
// WeatherGridLayer — one for layer lifecycle (create/destroy), another for
// smooth data updates.  This prevents the React cleanup→recreate cycle that
// was destroying and rebuilding the entire particle animation on every
// forecast frame change.
// ---------------------------------------------------------------------------
function VelocityParticleLayerInner({ data, type }: VelocityParticleLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);
  const dataRef = useRef(data);
  dataRef.current = data;
  const [zoom, setZoom] = useState<number>(map.getZoom());

  // Track zoom changes
  useEffect(() => {
    const onZoom = () => setZoom(map.getZoom());
    map.on('zoomend', onZoom);
    return () => { map.off('zoomend', onZoom); };
  }, [map]);

  // Boolean flag: does data exist? Only transitions (null↔valid) trigger
  // layer creation/destruction — NOT every content change.
  const hasData = !!(data && data.length >= 2);

  // Flag to skip the redundant setData call that fires when both effects
  // trigger on the same render (initial data load or zoom change).
  const skipNextSetDataRef = useRef(false);

  // -------------------------------------------------------------------
  // Effect 1: Create / destroy layer on zoom, type, or data availability
  // -------------------------------------------------------------------
  useEffect(() => {
    const currentData = dataRef.current;

    if (!currentData || currentData.length < 2) {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
      return;
    }

    require('leaflet-velocity/dist/leaflet-velocity.css');
    require('leaflet-velocity');
    if (!L.velocityLayer) return;

    // Destroy previous layer if it exists (zoom or type changed)
    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

    const isWind = type === 'wind';

    const layer = L.velocityLayer({
      displayValues: true,
      displayOptions: {
        velocityType: isWind ? 'Wind' : 'Ocean Current',
        position: 'bottomleft',
        emptyString: 'No data',
        angleConvention: 'bearingCW',
        speedUnit: 'm/s',
        directionString: 'Direction',
        speedString: 'Speed',
      },
      data: currentData,
      minVelocity: 0,
      maxVelocity: isWind ? 15 : 1.5,
      velocityScale: isWind ? 0.01 : 0.05,
      colorScale: isWind ? WIND_COLOR_SCALE : CURRENT_COLOR_SCALE,
      lineWidth: 1.5,
      particleAge: isWind ? 90 : 60,
      particleMultiplier: getParticleMultiplier(zoom, isWind),
      frameRate: 20,
      opacity: 0.97,
    });

    // ------------------------------------------------------------------
    // Patch setData: persistent snapshot bridge with debounced removal.
    //
    // Problem: leaflet-velocity's _clearAndRestart() clears the canvas
    // then restarts the animation.  New particles take ~150-200 ms to
    // become visible → blank flash on every forecast frame change.
    //
    // Solution: On the FIRST setData call, screenshot the current canvas
    // onto a sibling overlay.  Keep that overlay visible (cancel removal)
    // across rapid calls (scrubbing).  Only fade it out 350 ms after the
    // LAST call — by then the new animation has rendered visible particles.
    // ------------------------------------------------------------------
    let snap: HTMLCanvasElement | null = null;
    let fadeTimer: ReturnType<typeof setTimeout> | null = null;
    let removeTimer: ReturnType<typeof setTimeout> | null = null;

    layer.setData = function (newData: any) {
      const ctx = this._context;
      if (ctx) {
        const canvas = ctx.canvas as HTMLCanvasElement;
        const parent = canvas.parentNode as HTMLElement | null;
        if (parent) {
          // Cancel all pending timers (rapid scrubbing resets the clock)
          if (fadeTimer) { clearTimeout(fadeTimer); fadeTimer = null; }
          if (removeTimer) { clearTimeout(removeTimer); removeTimer = null; }

          if (!snap || !snap.parentNode) {
            // Create snapshot once from last visible frame
            snap = document.createElement('canvas');
            snap.width = canvas.width;
            snap.height = canvas.height;
            snap.style.cssText = canvas.style.cssText;
            snap.style.pointerEvents = 'none';
            snap.style.opacity = '1';
            snap.style.transition = '';
            const sc = snap.getContext('2d');
            if (sc) sc.drawImage(canvas, 0, 0);
            parent.insertBefore(snap, canvas.nextSibling);
          } else {
            // Snapshot already visible (mid-fade or opaque) — reset to opaque
            snap.style.transition = '';
            snap.style.opacity = '1';
          }

          // Schedule fade-out 500 ms after last setData — covers leaflet-velocity's
          // grid rebuild + interpolation (~300-400ms) so particles are visible before
          // the snapshot fades out.
          fadeTimer = setTimeout(() => {
            fadeTimer = null;
            if (snap && snap.parentNode) {
              snap.style.transition = 'opacity 0.35s ease-out';
              snap.style.opacity = '0';
              removeTimer = setTimeout(() => {
                removeTimer = null;
                snap?.remove();
                snap = null;
              }, 370);
            }
          }, 500);
        }
      }

      // Standard leaflet-velocity data update
      this.options.data = newData;
      if (this._windy) {
        this._windy.setData(newData);
        this._clearAndRestart();
      }
      this.fire('load');
    };

    layer.addTo(map);
    layerRef.current = layer;
    skipNextSetDataRef.current = true; // layer already has initial data

    return () => {
      if (fadeTimer) clearTimeout(fadeTimer);
      if (removeTimer) clearTimeout(removeTimer);
      if (snap) { snap.remove(); snap = null; }
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [type, map, L, zoom, hasData]); // NOT data content — only structural changes

  // -------------------------------------------------------------------
  // Effect 2: Smooth data update — runs ONLY when data content changes.
  // The layer persists; we just swap the vector field via setData().
  // -------------------------------------------------------------------
  useEffect(() => {
    if (skipNextSetDataRef.current) {
      skipNextSetDataRef.current = false;
      return;
    }
    if (layerRef.current && data && data.length >= 2) {
      layerRef.current.setData(data);
    }
  }, [data]);

  return null;
}
