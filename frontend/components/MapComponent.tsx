'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';
import { Position, WindFieldData, WaveFieldData, VelocityData, CreateZoneRequest, WaveForecastFrames } from '@/lib/api';

// Dynamic imports for map components (client-side only)
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);
const WaypointEditor = dynamic(() => import('@/components/WaypointEditor'), {
  ssr: false,
});
const WeatherGridLayer = dynamic(
  () => import('@/components/WeatherGridLayer'),
  { ssr: false }
);
const WeatherLegend = dynamic(
  () => import('@/components/WeatherLegend'),
  { ssr: false }
);
const VelocityParticleLayer = dynamic(
  () => import('@/components/VelocityParticleLayer'),
  { ssr: false }
);
const ZoneLayer = dynamic(
  () => import('@/components/ZoneLayer'),
  { ssr: false }
);
const ZoneEditor = dynamic(
  () => import('@/components/ZoneEditor'),
  { ssr: false }
);
const ForecastTimeline = dynamic(
  () => import('@/components/ForecastTimeline'),
  { ssr: false }
);
const WaveInfoPopup = dynamic(
  () => import('@/components/WaveInfoPopup'),
  { ssr: false }
);
const MapViewportProvider = dynamic(
  () => import('@/components/MapViewportProvider'),
  { ssr: false }
);
const Polyline = dynamic(
  () => import('react-leaflet').then((mod) => mod.Polyline),
  { ssr: false }
);

const DEFAULT_CENTER: [number, number] = [45, 10];
const DEFAULT_ZOOM = 5;

export type WeatherLayer = 'wind' | 'waves' | 'currents' | 'none';

export interface MapComponentProps {
  waypoints: Position[];
  onWaypointsChange: (wps: Position[]) => void;
  isEditing: boolean;
  weatherLayer: WeatherLayer;
  windData: WindFieldData | null;
  waveData: WaveFieldData | null;
  windVelocityData: VelocityData[] | null;
  currentVelocityData: VelocityData[] | null;
  showZones?: boolean;
  visibleZoneTypes?: string[];
  zoneKey?: number;
  isDrawingZone?: boolean;
  onSaveZone?: (request: CreateZoneRequest) => Promise<void>;
  onCancelZone?: () => void;
  forecastEnabled?: boolean;
  onForecastClose?: () => void;
  onForecastHourChange?: (hour: number, data: VelocityData[] | null) => void;
  onWaveForecastHourChange?: (hour: number, allFrames: WaveForecastFrames | null) => void;
  onCurrentForecastHourChange?: (hour: number, allFrames: any | null) => void;
  optimizedWaypoints?: Position[];
  onViewportChange?: (viewport: { bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }; zoom: number }) => void;
  viewportBounds?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null;
  children?: React.ReactNode;
}

export default function MapComponent({
  waypoints,
  onWaypointsChange,
  isEditing,
  weatherLayer,
  windData,
  waveData,
  windVelocityData,
  currentVelocityData,
  showZones = true,
  visibleZoneTypes,
  zoneKey = 0,
  isDrawingZone = false,
  onSaveZone,
  onCancelZone,
  forecastEnabled = false,
  onForecastClose,
  onForecastHourChange,
  onWaveForecastHourChange,
  onCurrentForecastHourChange,
  optimizedWaypoints,
  onViewportChange,
  viewportBounds = null,
  children,
}: MapComponentProps) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-maritime-dark rounded-lg">
        <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
      </div>
    );
  }

  return (
    <div className="relative w-full h-full">
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        minZoom={3}
        maxBounds={[[-85, -180], [85, 180]]}
        maxBoundsViscosity={1.0}
        worldCopyJump={true}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg"
        wheelPxPerZoomLevel={120}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Viewport tracker */}
        {onViewportChange && <MapViewportProvider onViewportChange={onViewportChange} />}

        {/* Zone Layer */}
        {showZones && <ZoneLayer key={zoneKey} visible={showZones} visibleTypes={visibleZoneTypes} />}

        {/* Zone Editor (drawing) */}
        {isDrawingZone && onSaveZone && onCancelZone && (
          <ZoneEditor
            isDrawing={isDrawingZone}
            onSaveZone={onSaveZone}
            onCancel={onCancelZone}
          />
        )}

        {/* Weather Layers */}
        {weatherLayer === 'wind' && windData && (
          <WeatherGridLayer
            mode="wind"
            windData={windData}
            opacity={0.6}
            showArrows={false}
          />
        )}
        {weatherLayer === 'wind' && windVelocityData && (
          <VelocityParticleLayer data={windVelocityData} type="wind" />
        )}
        {weatherLayer === 'waves' && waveData && (
          <WeatherGridLayer
            mode="waves"
            waveData={waveData}
          />
        )}
        {weatherLayer === 'currents' && currentVelocityData && (
          <VelocityParticleLayer data={currentVelocityData} type="currents" />
        )}

        {/* Wave Info Popup (click-to-inspect polar diagram) */}
        {weatherLayer === 'waves' && (
          <WaveInfoPopup
            active={weatherLayer === 'waves'}
            waveData={waveData}
            windData={windData}
          />
        )}

        {/* Weather Legend */}
        {weatherLayer !== 'none' && (
          <WeatherLegend mode={weatherLayer} timelineVisible={forecastEnabled} />
        )}

        {/* Waypoint Editor */}
        <WaypointEditor
          waypoints={waypoints}
          onWaypointsChange={onWaypointsChange}
          isEditing={isEditing}
        />

        {/* Optimized route overlay (green dashed) */}
        {optimizedWaypoints && optimizedWaypoints.length >= 2 && (
          <Polyline
            positions={optimizedWaypoints.map(wp => [wp.lat, wp.lon] as [number, number])}
            pathOptions={{
              color: '#22c55e',
              weight: 3,
              opacity: 0.85,
              dashArray: '8, 6',
            }}
          />
        )}
      </MapContainer>

      {/* Floating overlay controls */}
      {children}

      {/* Forecast Timeline overlay at bottom of map */}
      {forecastEnabled && onForecastClose && onForecastHourChange && (
        <ForecastTimeline
          visible={forecastEnabled}
          onClose={onForecastClose}
          onForecastHourChange={onForecastHourChange}
          onWaveForecastHourChange={onWaveForecastHourChange}
          onCurrentForecastHourChange={onCurrentForecastHourChange}
          layerType={weatherLayer === 'none' ? 'wind' : weatherLayer}
          viewportBounds={viewportBounds}
        />
      )}
    </div>
  );
}
