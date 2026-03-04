"""
Weather and route visualization module.

Creates weather maps with wind/wave overlays and route visualizations
using matplotlib and cartopy.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)


class WeatherPlotter:
    """
    Visualize weather data and maritime routes.

    Creates publication-quality weather maps with wind vectors,
    wave heights, and route overlays.
    """

    def __init__(self, use_cartopy: bool = True):
        """
        Initialize weather plotter.

        Args:
            use_cartopy: Use cartopy for map projections (requires cartopy)
        """
        self.use_cartopy = use_cartopy

        if use_cartopy:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature

                self.ccrs = ccrs
                self.cfeature = cfeature
            except ImportError:
                logger.warning(
                    "cartopy not installed. Using basic matplotlib plotting."
                )
                self.use_cartopy = False

    def plot_wind_field(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        title: str = "Wind Field",
        output_file: Optional[Path] = None,
        route: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Plot wind field with optional route overlay.

        Args:
            lats: 2D array of latitudes
            lons: 2D array of longitudes
            u_wind: 2D array of U-component wind (m/s)
            v_wind: 2D array of V-component wind (m/s)
            title: Plot title
            output_file: Save to file (if None, display)
            route: Optional list of (lat, lon) waypoints
        """
        # Calculate wind speed
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)

        # Create figure
        if self.use_cartopy:
            fig, ax = plt.subplots(
                figsize=(12, 8),
                subplot_kw={"projection": self.ccrs.PlateCarree()},
            )
            ax.add_feature(self.cfeature.LAND, facecolor="lightgray")
            ax.add_feature(self.cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(self.cfeature.BORDERS, linewidth=0.3)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)

        # Plot wind speed as contour
        contour = ax.contourf(
            lons,
            lats,
            wind_speed,
            levels=15,
            cmap="YlOrRd",
            alpha=0.7,
            transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
        )
        plt.colorbar(contour, ax=ax, label="Wind Speed (m/s)")

        # Plot wind vectors (subsample for clarity)
        skip = max(1, len(lats) // 20)
        ax.quiver(
            lons[::skip, ::skip],
            lats[::skip, ::skip],
            u_wind[::skip, ::skip],
            v_wind[::skip, ::skip],
            transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
            alpha=0.6,
            scale=200,
        )

        # Plot route if provided
        if route:
            route_lats = [wp[0] for wp in route]
            route_lons = [wp[1] for wp in route]
            ax.plot(
                route_lons,
                route_lats,
                "b-",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Route",
                transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
            )
            ax.legend()

        ax.set_title(title, fontsize=14, fontweight="bold")

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved wind field plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_wave_field(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        wave_height: np.ndarray,
        wave_period: Optional[np.ndarray] = None,
        title: str = "Significant Wave Height",
        output_file: Optional[Path] = None,
        route: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Plot wave height field with optional route overlay.

        Args:
            lats: 2D array of latitudes
            lons: 2D array of longitudes
            wave_height: 2D array of significant wave height (m)
            wave_period: Optional 2D array of wave period (s)
            title: Plot title
            output_file: Save to file (if None, display)
            route: Optional list of (lat, lon) waypoints
        """
        # Create figure
        if self.use_cartopy:
            fig, ax = plt.subplots(
                figsize=(12, 8),
                subplot_kw={"projection": self.ccrs.PlateCarree()},
            )
            ax.add_feature(self.cfeature.LAND, facecolor="lightgray")
            ax.add_feature(self.cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(self.cfeature.BORDERS, linewidth=0.3)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)

        # Plot wave height
        contour = ax.contourf(
            lons,
            lats,
            wave_height,
            levels=15,
            cmap="Blues",
            transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
        )
        plt.colorbar(contour, ax=ax, label="Significant Wave Height (m)")

        # Add contour lines for better readability
        ax.contour(
            lons,
            lats,
            wave_height,
            levels=6,
            colors="black",
            alpha=0.3,
            linewidths=0.5,
            transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
        )

        # Plot route if provided
        if route:
            route_lats = [wp[0] for wp in route]
            route_lons = [wp[1] for wp in route]
            ax.plot(
                route_lons,
                route_lats,
                "r-",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Route",
                transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
            )
            ax.legend()

        ax.set_title(title, fontsize=14, fontweight="bold")

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved wave field plot to {output_file}")
        else:
            plt.show()

        plt.close()

    def plot_route_profile(
        self,
        distances: List[float],
        weather_data: List[dict],
        output_file: Optional[Path] = None,
    ) -> None:
        """
        Plot weather profile along a route.

        Args:
            distances: Distance along route (nm)
            weather_data: List of weather dictionaries at each point
            output_file: Save to file (if None, display)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Extract data
        wind_speeds = [
            w.get("wind_speed_ms", 0) * 1.944 for w in weather_data
        ]  # Convert to knots
        wave_heights = [w.get("sig_wave_height_m", 0) for w in weather_data]
        wind_dirs = [w.get("wind_dir_deg", 0) for w in weather_data]

        # Plot wind speed
        ax1.plot(distances, wind_speeds, "b-", linewidth=2)
        ax1.fill_between(distances, wind_speeds, alpha=0.3)
        ax1.set_ylabel("Wind Speed (kts)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Weather Profile Along Route", fontsize=14, fontweight="bold")

        # Plot wave height
        ax2.plot(distances, wave_heights, "c-", linewidth=2)
        ax2.fill_between(distances, wave_heights, alpha=0.3, color="cyan")
        ax2.set_ylabel("Sig. Wave Height (m)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Plot wind direction
        ax3.plot(distances, wind_dirs, "g-", linewidth=2, marker="o", markersize=3)
        ax3.set_ylabel("Wind Direction (°)", fontsize=12)
        ax3.set_xlabel("Distance Along Route (nm)", fontsize=12)
        ax3.set_ylim(0, 360)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved route profile to {output_file}")
        else:
            plt.show()

        plt.close()

    def create_forecast_animation(
        self,
        grib_parser,
        variable: str,
        output_file: Path,
        route: Optional[List[Tuple[float, float]]] = None,
        fps: int = 2,
    ) -> None:
        """
        Create animation of forecast evolution.

        Args:
            grib_parser: GRIBParser instance with loaded data
            variable: Variable to animate (e.g., "UGRD", "HTSGW")
            output_file: Output file path (.gif or .mp4)
            route: Optional route overlay
            fps: Frames per second
        """
        try:
            forecast_times = grib_parser.get_forecast_times()

            # Get first frame to set up plot
            lats, lons, values = grib_parser.get_grid_data(variable, forecast_times[0])

            # Create figure
            if self.use_cartopy:
                fig, ax = plt.subplots(
                    figsize=(12, 8),
                    subplot_kw={"projection": self.ccrs.PlateCarree()},
                )
                ax.add_feature(self.cfeature.LAND, facecolor="lightgray")
                ax.add_feature(self.cfeature.COASTLINE, linewidth=0.5)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))

            # Initial contour plot
            contour = ax.contourf(
                lons,
                lats,
                values,
                levels=15,
                cmap="YlOrRd",
                transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
            )
            plt.colorbar(contour, ax=ax, label=variable)

            # Plot route if provided
            if route:
                route_lats = [wp[0] for wp in route]
                route_lons = [wp[1] for wp in route]
                ax.plot(
                    route_lons,
                    route_lats,
                    "b-",
                    linewidth=2,
                    marker="o",
                    transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
                )

            title = ax.set_title(
                f"{variable} - {forecast_times[0]}",
                fontsize=14,
                fontweight="bold",
            )

            def update(frame):
                ax.clear()
                if self.use_cartopy:
                    ax.add_feature(self.cfeature.LAND, facecolor="lightgray")
                    ax.add_feature(self.cfeature.COASTLINE, linewidth=0.5)

                lats, lons, values = grib_parser.get_grid_data(
                    variable, forecast_times[frame]
                )

                ax.contourf(
                    lons,
                    lats,
                    values,
                    levels=15,
                    cmap="YlOrRd",
                    transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
                )

                if route:
                    ax.plot(
                        route_lons,
                        route_lats,
                        "b-",
                        linewidth=2,
                        marker="o",
                        transform=self.ccrs.PlateCarree() if self.use_cartopy else None,
                    )

                ax.set_title(
                    f"{variable} - {forecast_times[frame]}",
                    fontsize=14,
                    fontweight="bold",
                )

            anim = FuncAnimation(
                fig,
                update,
                frames=len(forecast_times),
                interval=1000 // fps,
                repeat=True,
            )

            # Save animation
            if output_file.suffix == ".gif":
                anim.save(output_file, writer="pillow", fps=fps)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps)

            logger.info(f"Saved animation to {output_file}")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create animation: {e}")
            raise

    def plot_fuel_comparison(
        self,
        routes: List[dict],
        labels: List[str],
        output_file: Optional[Path] = None,
    ) -> None:
        """
        Compare fuel consumption across different routes.

        Args:
            routes: List of route dictionaries with fuel data
            labels: Labels for each route
            output_file: Save to file (if None, display)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of total fuel
        fuels = [r["total_fuel_mt"] for r in routes]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(routes)))

        ax1.bar(labels, fuels, color=colors)
        ax1.set_ylabel("Total Fuel Consumption (MT)", fontsize=12)
        ax1.set_title("Fuel Consumption Comparison", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3, axis="y")

        # Add values on bars
        for i, fuel in enumerate(fuels):
            ax1.text(i, fuel, f"{fuel:.1f}", ha="center", va="bottom", fontsize=10)

        # Breakdown by component
        if all("fuel_breakdown" in r for r in routes):
            calm_water = [r["fuel_breakdown"]["calm_water"] for r in routes]
            wind = [r["fuel_breakdown"]["wind"] for r in routes]
            waves = [r["fuel_breakdown"]["waves"] for r in routes]

            x = np.arange(len(labels))
            width = 0.25

            ax2.bar(x - width, calm_water, width, label="Calm Water", color="skyblue")
            ax2.bar(x, wind, width, label="Wind", color="orange")
            ax2.bar(x + width, waves, width, label="Waves", color="green")

            ax2.set_ylabel("Fuel Component (MT)", fontsize=12)
            ax2.set_title("Fuel Breakdown by Component", fontsize=14, fontweight="bold")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved fuel comparison to {output_file}")
        else:
            plt.show()

        plt.close()
