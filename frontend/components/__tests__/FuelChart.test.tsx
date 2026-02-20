/**
 * Tests for the FuelChart component.
 *
 * Note: Recharts renders SVG which is challenging to test in jsdom.
 * These tests focus on component rendering and data handling.
 */

import { render, screen } from '@testing-library/react';
import FuelChart from '../FuelChart';

// Mock Recharts to avoid SVG rendering issues in tests
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container" style={{ width: 500, height: 300 }}>
        {children}
      </div>
    ),
    BarChart: ({ data, children }: { data: unknown[]; children: React.ReactNode }) => (
      <div data-testid="bar-chart" data-length={data.length}>
        {children}
      </div>
    ),
    Bar: ({ dataKey, name, fill }: { dataKey: string; name: string; fill: string }) => (
      <div
        data-testid={`bar-${dataKey}`}
        data-name={name}
        data-fill={fill}
      />
    ),
    XAxis: () => <div data-testid="x-axis" />,
    YAxis: () => <div data-testid="y-axis" />,
    CartesianGrid: () => <div data-testid="cartesian-grid" />,
    Tooltip: () => <div data-testid="tooltip" />,
    Legend: () => <div data-testid="legend" />,
  };
});

describe('FuelChart', () => {
  const mockData = [
    { name: 'Scenario 1', calm_water: 10, wind: 2, waves: 3 },
    { name: 'Scenario 2', calm_water: 12, wind: 3, waves: 4 },
    { name: 'Scenario 3', calm_water: 8, wind: 1, waves: 2 },
  ];

  it('renders the chart container', () => {
    render(<FuelChart data={mockData} />);

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument();
  });

  it('renders bar chart with correct data length', () => {
    render(<FuelChart data={mockData} />);

    const barChart = screen.getByTestId('bar-chart');
    expect(barChart).toHaveAttribute('data-length', '3');
  });

  it('renders calm water bar with correct config', () => {
    render(<FuelChart data={mockData} />);

    const calmWaterBar = screen.getByTestId('bar-calm_water');
    expect(calmWaterBar).toHaveAttribute('data-name', 'Calm Water');
    expect(calmWaterBar).toHaveAttribute('data-fill', '#3a5eae');
  });

  it('renders wind bar with correct config', () => {
    render(<FuelChart data={mockData} />);

    const windBar = screen.getByTestId('bar-wind');
    expect(windBar).toHaveAttribute('data-name', 'Wind');
    expect(windBar).toHaveAttribute('data-fill', '#5c7aa1');
  });

  it('renders waves bar with correct config', () => {
    render(<FuelChart data={mockData} />);

    const wavesBar = screen.getByTestId('bar-waves');
    expect(wavesBar).toHaveAttribute('data-name', 'Waves');
    expect(wavesBar).toHaveAttribute('data-fill', '#7692d1');
  });

  it('renders chart elements', () => {
    render(<FuelChart data={mockData} />);

    expect(screen.getByTestId('x-axis')).toBeInTheDocument();
    expect(screen.getByTestId('y-axis')).toBeInTheDocument();
    expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument();
    expect(screen.getByTestId('tooltip')).toBeInTheDocument();
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('renders with empty data', () => {
    render(<FuelChart data={[]} />);

    const barChart = screen.getByTestId('bar-chart');
    expect(barChart).toHaveAttribute('data-length', '0');
  });

  it('renders with single data point', () => {
    const singleData = [{ name: 'Single', calm_water: 5, wind: 1, waves: 1 }];
    render(<FuelChart data={singleData} />);

    const barChart = screen.getByTestId('bar-chart');
    expect(barChart).toHaveAttribute('data-length', '1');
  });
});
