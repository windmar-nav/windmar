declare module 'leaflet-velocity' {
  import * as L from 'leaflet';

  interface VelocityLayerOptions extends L.LayerOptions {
    displayValues?: boolean;
    displayOptions?: {
      velocityType?: string;
      position?: string;
      emptyString?: string;
      angleConvention?: 'bearingCW' | 'bearingCCW' | 'meteoCW' | 'meteoCCW';
      speedUnit?: string;
      directionString?: string;
      speedString?: string;
    };
    data?: object[];
    minVelocity?: number;
    maxVelocity?: number;
    velocityScale?: number;
    colorScale?: string[];
    lineWidth?: number;
    particleAge?: number;
    particleMultiplier?: number;
    frameRate?: number;
    opacity?: number;
    onAdd?: () => void;
    onRemove?: () => void;
  }

  const velocityLayer: (options?: VelocityLayerOptions) => L.Layer & {
    setData(data: object[]): void;
  };
  export default velocityLayer;
}
