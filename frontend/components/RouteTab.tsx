'use client';

import { Navigation, MousePointer, Upload, Shield, PenTool, Eye, EyeOff } from 'lucide-react';
import Card from '@/components/Card';
import { WaypointList } from '@/components/WaypointEditor';
import RouteImport, { SampleRTZButton } from '@/components/RouteImport';
import SavedRoutes from '@/components/SavedRoutes';
import VoyageParameters from '@/components/VoyageParameters';
import CalibrationPanel from '@/components/CalibrationPanel';
import { Position, OptimizationResponse } from '@/lib/api';

interface RouteTabProps {
  // Route state
  waypoints: Position[];
  onWaypointsChange: (wps: Position[]) => void;
  isEditing: boolean;
  onIsEditingChange: (editing: boolean) => void;
  routeName: string;
  onRouteImport: (waypoints: Position[], name: string) => void;
  onLoadRoute: (waypoints: Position[]) => void;
  onClearRoute: () => void;
  totalDistance: number;

  // Voyage parameters
  calmSpeed: number;
  onCalmSpeedChange: (speed: number) => void;
  isLaden: boolean;
  onIsLadenChange: (laden: boolean) => void;
  useWeather: boolean;
  onUseWeatherChange: (use: boolean) => void;
  isCalculating: boolean;
  onCalculate: () => void;
  isOptimizing: boolean;
  onOptimize: () => void;
  optimizationResult: OptimizationResponse | null;
  onApplyOptimizedRoute: () => void;

  // Zones
  showZones: boolean;
  onShowZonesChange: (show: boolean) => void;
  isDrawingZone: boolean;
  onIsDrawingZoneChange: (drawing: boolean) => void;
}

export default function RouteTab({
  waypoints,
  onWaypointsChange,
  isEditing,
  onIsEditingChange,
  routeName,
  onRouteImport,
  onLoadRoute,
  onClearRoute,
  totalDistance,
  calmSpeed,
  onCalmSpeedChange,
  isLaden,
  onIsLadenChange,
  useWeather,
  onUseWeatherChange,
  isCalculating,
  onCalculate,
  isOptimizing,
  onOptimize,
  optimizationResult,
  onApplyOptimizedRoute,
  showZones,
  onShowZonesChange,
  isDrawingZone,
  onIsDrawingZoneChange,
}: RouteTabProps) {
  return (
    <>
      {/* Route Input Card */}
      <Card title="Route" icon={<Navigation className="w-5 h-5" />}>
        {/* Mode Tabs */}
        <div className="flex space-x-2 mb-4">
          <button
            onClick={() => onIsEditingChange(true)}
            className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              isEditing
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-dark text-gray-400 hover:text-white'
            }`}
          >
            <MousePointer className="w-4 h-4" />
            <span>Draw</span>
          </button>
          <button
            onClick={() => onIsEditingChange(false)}
            className={`flex-1 flex items-center justify-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
              !isEditing
                ? 'bg-primary-500 text-white'
                : 'bg-maritime-dark text-gray-400 hover:text-white'
            }`}
          >
            <Upload className="w-4 h-4" />
            <span>Import</span>
          </button>
        </div>

        {/* Import Section */}
        {!isEditing && (
          <div className="mb-4">
            <RouteImport onImport={onRouteImport} />
            <div className="mt-2 text-center">
              <SampleRTZButton />
            </div>
          </div>
        )}

        {/* Waypoint List */}
        <WaypointList
          waypoints={waypoints}
          onWaypointsChange={onWaypointsChange}
          onClear={onClearRoute}
          totalDistance={totalDistance}
        />

        {isEditing && waypoints.length === 0 && (
          <p className="text-xs text-gray-500 mt-2">
            Click on the map to add waypoints
          </p>
        )}

        {/* Saved Routes */}
        <div className="mt-4 pt-4 border-t border-white/10">
          <SavedRoutes
            currentWaypoints={waypoints}
            onLoadRoute={onLoadRoute}
          />
        </div>
      </Card>

      {/* Voyage Parameters */}
      <VoyageParameters
        calmSpeed={calmSpeed}
        onCalmSpeedChange={onCalmSpeedChange}
        isLaden={isLaden}
        onIsLadenChange={onIsLadenChange}
        useWeather={useWeather}
        onUseWeatherChange={onUseWeatherChange}
        isCalculating={isCalculating}
        onCalculate={onCalculate}
        isOptimizing={isOptimizing}
        onOptimize={onOptimize}
        waypointCount={waypoints.length}
        optimizationResult={optimizationResult}
        onApplyOptimizedRoute={onApplyOptimizedRoute}
      />

      {/* Vessel Calibration */}
      <CalibrationPanel />

      {/* Regulatory Zones */}
      <Card title="Regulatory Zones" icon={<Shield className="w-5 h-5" />}>
        <div className="space-y-2">
          <ZoneToggleButton
            active={showZones}
            onClick={() => onShowZonesChange(!showZones)}
          />
          <button
            onClick={() => onIsDrawingZoneChange(!isDrawingZone)}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
              isDrawingZone
                ? 'bg-amber-500/20 border border-amber-500/50 text-amber-400'
                : 'bg-maritime-dark text-gray-400 hover:text-white'
            }`}
          >
            <div className="flex items-center space-x-2">
              <PenTool className="w-4 h-4" />
              <span className="text-sm">Draw Custom Zone</span>
            </div>
          </button>
          {isDrawingZone && (
            <p className="text-xs text-amber-400 px-2">
              Click on the map to draw a polygon, then fill in zone properties.
            </p>
          )}
        </div>
      </Card>
    </>
  );
}

function ZoneToggleButton({ active, onClick }: { active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
        active
          ? 'bg-primary-500/20 border border-primary-500/50 text-primary-400'
          : 'bg-maritime-dark text-gray-400 hover:text-white'
      }`}
    >
      <div className="flex items-center space-x-2">
        <Shield className="w-4 h-4" />
        <span className="text-sm">Show Zones</span>
      </div>
      {active ? (
        <Eye className="w-4 h-4" />
      ) : (
        <EyeOff className="w-4 h-4" />
      )}
    </button>
  );
}
