/**
 * Tests for the Header component.
 */

import { render, screen } from '@testing-library/react';
import Header from '../Header';

// Mock VoyageContext
jest.mock('@/components/VoyageContext', () => ({
  useVoyage: () => ({
    calmSpeed: 14.5,
    setCalmSpeed: jest.fn(),
    isLaden: true,
    setIsLaden: jest.fn(),
    useWeather: true,
    setUseWeather: jest.fn(),
    zoneVisibility: {},
    setZoneTypeVisible: jest.fn(),
    isDrawingZone: false,
    setIsDrawingZone: jest.fn(),
  }),
  ZONE_TYPES: ['eca', 'seca', 'hra', 'tss', 'vts', 'ice', 'canal', 'environmental', 'exclusion'],
}));

describe('Header', () => {
  it('renders the WINDMAR logo and title', () => {
    render(<Header />);

    expect(screen.getByText('WINDMAR')).toBeInTheDocument();
    expect(screen.getByText('Weather Routing & Performance Analytics')).toBeInTheDocument();
  });

  it('renders Vessel link to /vessel', () => {
    render(<Header />);

    const vesselLink = screen.getByRole('link', { name: /vessel/i });
    expect(vesselLink).toHaveAttribute('href', '/vessel');
  });

  it('displays system status indicator', () => {
    render(<Header />);

    expect(screen.getByText('Online')).toBeInTheDocument();
  });

  it('renders logo link pointing to home', () => {
    render(<Header />);

    const logoLink = screen.getByRole('link', { name: /windmar/i });
    expect(logoLink).toHaveAttribute('href', '/');
  });
});
