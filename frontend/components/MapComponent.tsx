'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Loader2 } from 'lucide-react';
import { Position, WindFieldData, WaveFieldData, VelocityData, CreateZoneRequest, WaveForecastFrames, IceForecastFrames, SstForecastFrames, VisForecastFrames, AllOptimizationResults, RouteVisibility, OptimizedRouteKey, ROUTE_STYLES, GridFieldData, SwellFieldData } from '@/lib/api';
import { sogToColor } from '@/lib/utils';
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
const WeatherCanvasOverlay = dynamic(
  () => import('@/components/WeatherCanvasOverlay'),
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
const InitialFitBounds = dynamic(
  () => import('@/components/InitialFitBounds'),
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

const DEFAULT_CENTER: [number, number] = [45, 5];
const DEFAULT_ZOOM = 5;
const INITIAL_BOUNDS: [[number, number], [number, number]] = [[30, -30], [60, 40]];

export type WeatherLayer = 'wind' | 'waves' | 'currents' | 'ice' | 'visibility' | 'sst' | 'swell' | 'none';

export interface MapComponentProps {
  waypoints: Position[];
  onWaypointsChange: (wps: Position[]) => void;
  isEditing: boolean;
  weatherLayer: WeatherLayer;
  windData: WindFieldData | null;
  windVelocityData: VelocityData[] | null;
  waveData: WaveFieldData | null;
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
  currentForecastHour?: number;
  restoredViewport?: { bounds: { lat_min: number; lat_max: number; lon_min: number; lon_max: number }; zoom: number } | null;
  children?: React.ReactNode;
}

export default function MapComponent({
  waypoints,
  onWaypointsChange,
  isEditing,
  weatherLayer,
  windData,
  windVelocityData,
  waveData,
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
  currentForecastHour = 0,
  restoredViewport = null,
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
        minZoom={DEMO_MODE ? 4 : 3}
        zoomSnap={0.25}
        zoomDelta={0.5}
        maxBounds={DEMO_MODE ? DEMO_BOUNDS : [[-85, -180], [85, 180]]}
        maxBoundsViscosity={1.0}
        worldCopyJump={!DEMO_MODE}
        dragging={true}
        zoomControl={true}
        scrollWheelZoom={true}
        doubleClickZoom={true}
        touchZoom={true}
        boxZoom={!DEMO_MODE}
        keyboard={!DEMO_MODE}
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

        {/* Fit initial viewport — restored viewport if available, else default */}
        <InitialFitBounds
          bounds={restoredViewport
            ? [[restoredViewport.bounds.lat_min, restoredViewport.bounds.lon_min],
               [restoredViewport.bounds.lat_max, restoredViewport.bounds.lon_max]]
            : INITIAL_BOUNDS}
        />

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

        {/* Weather heatmap — client-side canvas overlay (seamless, no tile seams) */}
        {weatherLayer !== 'none' && weatherLayer !== 'currents' && (
          <WeatherCanvasOverlay
            mode={weatherLayer as 'wind' | 'waves' | 'ice' | 'visibility' | 'sst' | 'swell'}
            windData={windData}
            waveData={waveData}
            extendedData={extendedWeatherData as GridFieldData | SwellFieldData | null}
            opacity={0.6}
          />
        )}

        {/* Wind barbs overlay (WMO standard) */}
        {weatherLayer === 'wind' && windData && (
          <WeatherGridLayer mode="wind" windData={windData} opacity={0.7} />
        )}

        {/* Currents: particle animation */}
        {weatherLayer === 'currents' && currentVelocityData && (
          <VelocityParticleLayer data={currentVelocityData} type="currents" />
        )}

        {/* Wave direction crests overlay */}
        {weatherLayer === 'waves' && waveData && (
          <WeatherGridLayer
            mode="waves"
            waveData={waveData}
            opacity={0.7}
          />
        )}

        {/* Swell direction arrows overlay */}
        {weatherLayer === 'swell' && extendedWeatherData && (
          <WeatherGridLayer
            mode="swell"
            extendedData={extendedWeatherData}
            opacity={0.7}
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
          <WeatherLegend
            mode={weatherLayer}
            timelineVisible={forecastEnabled}
            dataRange={weatherLayer === 'sst' && extendedWeatherData?.colorscale ? { min: extendedWeatherData.colorscale.data_min, max: extendedWeatherData.colorscale.data_max } : null}
          />
        )}

        {/* Waypoint Editor */}
        <WaypointEditor
          waypoints={waypoints}
          onWaypointsChange={onWaypointsChange}
          isEditing={isEditing}
          routeColor={routeVisibility?.original === false ? 'transparent' : undefined}
        />

        {/* Optimized route overlays — per-leg SOG gradient coloring */}
        {allResults && routeVisibility && (Object.keys(ROUTE_STYLES) as OptimizedRouteKey[]).map(key => {
          const result = allResults[key];
          if (!routeVisibility[key] || !result?.waypoints?.length || result.waypoints.length < 2) return null;
          const style = ROUTE_STYLES[key];
          const legs = result.legs;

          // If legs with SOG data available and meaningful speed variation,
          // render per-leg colored segments. Skip gradient when spread < 1 kt
          // (constant-speed optimized routes) to avoid invisible default colors.
          if (legs && legs.length > 0 && legs.some(l => l.sog_kts > 0)) {
            const sogValues = legs.map(l => l.sog_kts);
            const minSog = Math.min(...sogValues);
            const maxSog = Math.max(...sogValues);
            if (maxSog - minSog >= 1.0) {
              return legs.map((leg, i) => (
                <Polyline
                  key={`${key}-leg-${i}`}
                  positions={[
                    [leg.from_lat, leg.from_lon] as [number, number],
                    [leg.to_lat, leg.to_lon] as [number, number],
                  ]}
                  pathOptions={{
                    color: sogToColor(leg.sog_kts, minSog, maxSog) ?? style.color,
                    weight: 3,
                    opacity: 0.9,
                  }}
                >
                  <Tooltip sticky>{style.label}: {leg.sog_kts.toFixed(1)} kts</Tooltip>
                </Polyline>
              ));
            }
          }

          // Fallback: single polyline with route style
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
