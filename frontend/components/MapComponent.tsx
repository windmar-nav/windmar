'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';
import { Position, WindFieldData, WaveFieldData, VelocityData, CreateZoneRequest, WaveForecastFrames, IceForecastFrames, SstForecastFrames, VisForecastFrames, AllOptimizationResults, RouteVisibility, OptimizedRouteKey, ROUTE_STYLES } from '@/lib/api';
import { DEMO_MODE, DEMO_BOUNDS } from '@/lib/demoMode';

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
const CountryLabelsLayer = dynamic(
  () => import('@/components/CountryLabels'),
  { ssr: false }
);
const ZoneLayer = dynamic(
  () => import('@/components/ZoneLayer'),
  { ssr: false }
);
const CoastlineOverlay = dynamic(
  () => import('@/components/CoastlineOverlay'),
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
const FitBoundsHandler = dynamic(
  () => import('@/components/FitBoundsHandler'),
  { ssr: false }
);
const Polyline = dynamic(
  () => import('react-leaflet').then((mod) => mod.Polyline),
  { ssr: false }
);
const Tooltip = dynamic(
  () => import('react-leaflet').then((mod) => mod.Tooltip),
  { ssr: false }
);

const DEFAULT_CENTER: [number, number] = [45, 10];
const DEFAULT_ZOOM = 5;

export type WeatherLayer = 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell' | 'none';

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
  onIceForecastHourChange?: (hour: number, allFrames: IceForecastFrames | null) => void;
  onSwellForecastHourChange?: (hour: number, allFrames: WaveForecastFrames | null) => void;
  onSstForecastHourChange?: (hour: number, allFrames: SstForecastFrames | null) => void;
  onVisForecastHourChange?: (hour: number, allFrames: VisForecastFrames | null) => void;
  allResults?: AllOptimizationResults;
  routeVisibility?: RouteVisibility;
  onViewportChange?: (viewport: { bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }; zoom: number }) => void;
  viewportBounds?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number } | null;
  weatherModelLabel?: string;
  extendedWeatherData?: any;
  fitBounds?: [[number, number], [number, number]] | null;
  fitKey?: number;
  dataTimestamp?: string | null;
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
  onIceForecastHourChange,
  onSwellForecastHourChange,
  onSstForecastHourChange,
  onVisForecastHourChange,
  allResults,
  routeVisibility,
  onViewportChange,
  viewportBounds = null,
  weatherModelLabel,
  extendedWeatherData = null,
  fitBounds: fitBoundsProp = null,
  fitKey = 0,
  dataTimestamp = null,
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
        maxBounds={DEMO_MODE ? DEMO_BOUNDS : [[-85, -180], [85, 180]]}
        maxBoundsViscosity={1.0}
        worldCopyJump={true}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg"
        wheelPxPerZoomLevel={120}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
        />
        <CountryLabelsLayer />

        {/* Viewport tracker */}
        {onViewportChange && <MapViewportProvider onViewportChange={onViewportChange} />}

        {/* Fit bounds handler */}
        <FitBoundsHandler bounds={fitBoundsProp} fitKey={fitKey} />

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

        {/* Extended weather layers (SPEC-P1) */}
        {(weatherLayer === 'ice' || weatherLayer === 'visibility' || weatherLayer === 'sst' || weatherLayer === 'swell') && extendedWeatherData && (
          <WeatherGridLayer
            mode={weatherLayer as 'ice' | 'visibility' | 'sst' | 'swell'}
            extendedData={extendedWeatherData}
            opacity={0.6}
          />
        )}

        {/* GSHHS coastline overlay — crisp vector land boundaries above weather grids */}
        {weatherLayer !== 'none' && <CoastlineOverlay />}

        {/* Hover tooltip for wind, waves, swell, currents, visibility layers */}
        {(weatherLayer === 'wind' || weatherLayer === 'waves' || weatherLayer === 'swell' || weatherLayer === 'currents' || weatherLayer === 'visibility') && (
          <WaveInfoPopup
            layer={weatherLayer as 'wind' | 'waves' | 'swell' | 'currents' | 'visibility'}
            waveData={weatherLayer === 'waves' ? waveData : null}
            windData={windData}
            swellData={weatherLayer === 'swell' ? extendedWeatherData as any : null}
            currentVelocityData={currentVelocityData}
            visibilityData={weatherLayer === 'visibility' ? extendedWeatherData as any : null}
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
          routeColor={routeVisibility?.original === false ? 'transparent' : undefined}
        />

        {/* Optimized route overlays — dynamic loop over all 6 route keys */}
        {allResults && routeVisibility && (Object.keys(ROUTE_STYLES) as OptimizedRouteKey[]).map(key => {
          const result = allResults[key];
          if (!routeVisibility[key] || !result?.waypoints?.length || result.waypoints.length < 2) return null;
          const style = ROUTE_STYLES[key];
          return (
            <Polyline
              key={key}
              positions={result.waypoints.map(wp => [wp.lat, wp.lon] as [number, number])}
              pathOptions={{
                color: style.color,
                weight: 3,
                opacity: 0.85,
                dashArray: style.dashArray,
              }}
            >
              <Tooltip sticky>{style.label} route</Tooltip>
            </Polyline>
          );
        })}
      </MapContainer>

      {/* Weather model watermark */}
      {weatherModelLabel && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[999] pointer-events-none">
          <span className="text-white/15 text-sm font-medium tracking-wide select-none">
            {weatherModelLabel}
          </span>
        </div>
      )}

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
          onIceForecastHourChange={onIceForecastHourChange}
          onSwellForecastHourChange={onSwellForecastHourChange}
          onSstForecastHourChange={onSstForecastHourChange}
          onVisForecastHourChange={onVisForecastHourChange}
          layerType={(['wind', 'waves', 'currents', 'ice', 'swell', 'sst', 'visibility'] as const).includes(weatherLayer as any) ? weatherLayer as any : 'wind'}
          displayLayerName={{ wind: 'Wind Speed', waves: 'Waves', currents: 'Currents', ice: 'Ice', visibility: 'Visibility', sst: 'Sea Surface Temp', swell: 'Swell', none: undefined }[weatherLayer]}
          viewportBounds={viewportBounds}
          dataTimestamp={dataTimestamp}
        />
      )}
    </div>
  );
}
