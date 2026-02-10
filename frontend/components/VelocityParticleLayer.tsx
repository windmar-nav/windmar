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

function VelocityParticleLayerInner({ data, type }: VelocityParticleLayerProps) {
  const { useMap } = require('react-leaflet');
  const L = require('leaflet');
  const map = useMap();
  const layerRef = useRef<any>(null);
  const prevZoomRef = useRef<number>(map.getZoom());
  const prevTypeRef = useRef<string>(type);
  const [zoom, setZoom] = useState<number>(map.getZoom());

  // Track zoom changes
  useEffect(() => {
    const onZoom = () => setZoom(map.getZoom());
    map.on('zoomend', onZoom);
    return () => { map.off('zoomend', onZoom); };
  }, [map]);

  // Create layer once, update data smoothly via setData()
  useEffect(() => {
    if (!data || data.length < 2) {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
      return;
    }

    require('leaflet-velocity/dist/leaflet-velocity.css');
    require('leaflet-velocity');
    if (!L.velocityLayer) return;

    const isWind = type === 'wind';

    // Only recreate layer on zoom or type change; smooth-update for data-only changes
    const needsRecreate = !layerRef.current ||
      prevZoomRef.current !== zoom ||
      prevTypeRef.current !== type;

    prevZoomRef.current = zoom;
    prevTypeRef.current = type;

    if (!needsRecreate && layerRef.current) {
      // Smooth transition: just swap the vector field, particles adapt
      layerRef.current.setData(data);
      return;
    }

    // Full (re)creation — destroy old layer first
    if (layerRef.current) {
      map.removeLayer(layerRef.current);
      layerRef.current = null;
    }

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
      data: data,
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

    // Patch setData with canvas snapshot bridge — prevents blank flash
    // between forecast frames (#21). Before clearing, we screenshot the
    // current canvas onto a temporary overlay. The original _clearAndRestart
    // runs underneath (clearing + async grid rebuild). Once the new animation
    // renders its first frames (~150ms), the snapshot is removed.
    layer.setData = function(newData: any) {
        const ctx = this._context;
        if (ctx && this._windy) {
            const canvas = ctx.canvas as HTMLCanvasElement;
            const parent = canvas.parentNode as HTMLElement | null;
            if (parent) {
                const snap = document.createElement('canvas');
                snap.width = canvas.width;
                snap.height = canvas.height;
                snap.style.cssText = canvas.style.cssText;
                snap.style.pointerEvents = 'none';
                const sc = snap.getContext('2d');
                if (sc) {
                    sc.drawImage(canvas, 0, 0);
                    parent.insertBefore(snap, canvas.nextSibling);
                    // Remove snapshot after new animation has rendered
                    setTimeout(() => {
                        snap.style.transition = 'opacity 0.15s ease-out';
                        snap.style.opacity = '0';
                        setTimeout(() => snap.remove(), 160);
                    }, 120);
                }
            }
        }
        // Original setData logic (clear canvas + restart animation)
        this.options.data = newData;
        if (this._windy) {
            this._windy.setData(newData);
            this._clearAndRestart();
        }
        this.fire('load');
    };

    layer.addTo(map);
    layerRef.current = layer;

    return () => {
      if (layerRef.current) {
        map.removeLayer(layerRef.current);
        layerRef.current = null;
      }
    };
  }, [data, type, map, L, zoom]);

  return null;
}
