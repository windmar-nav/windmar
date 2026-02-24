'use client';

import { useEffect, useRef } from 'react';
import { useMap } from 'react-leaflet';
import type { LatLngBoundsExpression } from 'leaflet';

/** Fit map to bounds once on mount. */
export default function InitialFitBounds({ bounds }: { bounds: LatLngBoundsExpression }) {
  const map = useMap();
  const firedRef = useRef(false);

  useEffect(() => {
    if (!firedRef.current) {
      firedRef.current = true;
      map.fitBounds(bounds, { animate: false });
    }
  }, [map, bounds]);

  return null;
}
