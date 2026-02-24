"""
Time-series data storage and retrieval for sensor data.

Provides:
- In-memory ring buffer for real-time data
- SQLite persistence for historical data
- Efficient querying and aggregation
- Export to various formats
"""

import json
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .sbg_ellipse import SBGData


logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """Single point in a time series."""
    timestamp: datetime
    value: float
    quality: int = 0  # 0=good, 1=interpolated, 2=suspect


class TimeSeriesBuffer:
    """
    In-memory ring buffer for time-series data.

    Optimized for real-time streaming with O(1) append
    and efficient time-range queries.
    """

    def __init__(
        self,
        max_points: int = 10000,
        max_age_seconds: float = 3600.0,
    ):
        """
        Initialize buffer.

        Args:
            max_points: Maximum number of points to store
            max_age_seconds: Maximum age of points in seconds
        """
        self.max_points = max_points
        self.max_age_seconds = max_age_seconds

        self._timestamps: deque = deque(maxlen=max_points)
        self._values: deque = deque(maxlen=max_points)
        self._lock = threading.Lock()

    def append(self, timestamp: datetime, value: float) -> None:
        """Add a point to the buffer."""
        with self._lock:
            self._timestamps.append(timestamp)
            self._values.append(value)

    def get_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Tuple[List[datetime], List[float]]:
        """
        Get points within time range.

        Args:
            start: Start time (None = earliest)
            end: End time (None = now)

        Returns:
            Tuple of (timestamps, values) lists
        """
        with self._lock:
            timestamps = list(self._timestamps)
            values = list(self._values)

        if not timestamps:
            return [], []

        # Filter by time range
        if start is not None or end is not None:
            filtered_ts = []
            filtered_vals = []
            for ts, val in zip(timestamps, values):
                if start is not None and ts < start:
                    continue
                if end is not None and ts > end:
                    continue
                filtered_ts.append(ts)
                filtered_vals.append(val)
            return filtered_ts, filtered_vals

        return timestamps, values

    def get_latest(self, n: int = 1) -> Tuple[List[datetime], List[float]]:
        """Get the N most recent points."""
        with self._lock:
            n = min(n, len(self._timestamps))
            timestamps = list(self._timestamps)[-n:]
            values = list(self._values)[-n:]
        return timestamps, values

    def get_statistics(
        self,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute statistics over buffer or time window.

        Args:
            window_seconds: Time window (None = all data)

        Returns:
            Dictionary with mean, std, min, max, count
        """
        if window_seconds is not None:
            start = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
            _, values = self.get_range(start=start)
        else:
            _, values = self.get_range()

        if not values:
            return {"count": 0}

        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._timestamps.clear()
            self._values.clear()

    def __len__(self) -> int:
        return len(self._timestamps)


class SensorDataStore:
    """
    Complete data storage for SBG sensor data.

    Manages multiple time series channels with both
    real-time buffering and persistent storage.
    """

    # Channel names for SBG data
    CHANNELS = [
        "latitude", "longitude", "altitude",
        "sog", "cog", "heading",
        "roll", "pitch", "heave",
        "surge", "sway",
        "roll_rate", "pitch_rate", "yaw_rate",
        "accel_x", "accel_y", "accel_z",
    ]

    def __init__(
        self,
        db_path: Optional[str] = None,
        buffer_size: int = 10000,
        buffer_age_seconds: float = 3600.0,
        persist_interval_seconds: float = 60.0,
    ):
        """
        Initialize data store.

        Args:
            db_path: Path to SQLite database (None = memory only)
            buffer_size: Points per channel in memory
            buffer_age_seconds: Max age of in-memory data
            persist_interval_seconds: How often to persist to disk
        """
        self.db_path = db_path
        self.buffer_size = buffer_size
        self.persist_interval_seconds = persist_interval_seconds

        # Create buffers for each channel
        self._buffers: Dict[str, TimeSeriesBuffer] = {}
        for channel in self.CHANNELS:
            self._buffers[channel] = TimeSeriesBuffer(
                max_points=buffer_size,
                max_age_seconds=buffer_age_seconds,
            )

        # Database connection (lazy init)
        self._db: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()

        # Persistence thread
        self._persist_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending_writes: List[Dict] = []
        self._pending_lock = threading.Lock()

        # Initialize database if path provided
        if db_path:
            self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with self._db_lock:
            self._db = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
            )

            # Create tables
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    value REAL NOT NULL,
                    quality INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON sensor_data(timestamp);

                CREATE INDEX IF NOT EXISTS idx_channel_timestamp
                ON sensor_data(channel, timestamp);

                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    description TEXT,
                    metadata TEXT
                );
            """)

            logger.info(f"Initialized database: {self.db_path}")

    def store(self, data: SBGData) -> None:
        """
        Store a complete SBG data packet.

        Args:
            data: SBGData object to store
        """
        timestamp = data.timestamp

        # Update in-memory buffers
        channel_values = {
            "latitude": data.latitude,
            "longitude": data.longitude,
            "altitude": data.altitude,
            "sog": data.sog,
            "cog": data.cog,
            "heading": data.heading,
            "roll": data.roll,
            "pitch": data.pitch,
            "heave": data.heave,
            "surge": data.surge,
            "sway": data.sway,
            "roll_rate": data.roll_rate,
            "pitch_rate": data.pitch_rate,
            "yaw_rate": data.yaw_rate,
            "accel_x": data.accel_x,
            "accel_y": data.accel_y,
            "accel_z": data.accel_z,
        }

        for channel, value in channel_values.items():
            self._buffers[channel].append(timestamp, value)

        # Queue for persistence
        if self.db_path:
            with self._pending_lock:
                self._pending_writes.append({
                    "timestamp": timestamp.isoformat(),
                    "values": channel_values,
                })

    def get_channel(
        self,
        channel: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Tuple[List[datetime], List[float]]:
        """
        Get time series data for a channel.

        Args:
            channel: Channel name
            start: Start time
            end: End time

        Returns:
            Tuple of (timestamps, values)
        """
        if channel not in self._buffers:
            raise ValueError(f"Unknown channel: {channel}")

        return self._buffers[channel].get_range(start, end)

    def get_latest(
        self,
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[datetime, float]]:
        """
        Get latest value for each channel.

        Args:
            channels: List of channels (None = all)

        Returns:
            Dict of channel -> (timestamp, value)
        """
        if channels is None:
            channels = self.CHANNELS

        result = {}
        for channel in channels:
            if channel in self._buffers:
                ts, vals = self._buffers[channel].get_latest(1)
                if ts:
                    result[channel] = (ts[0], vals[0])

        return result

    def get_statistics(
        self,
        channel: str,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, float]:
        """Get statistics for a channel."""
        if channel not in self._buffers:
            raise ValueError(f"Unknown channel: {channel}")

        return self._buffers[channel].get_statistics(window_seconds)

    def get_all_statistics(
        self,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Get statistics for all channels."""
        return {
            channel: self._buffers[channel].get_statistics(window_seconds)
            for channel in self.CHANNELS
        }

    def start_persistence(self) -> None:
        """Start background persistence thread."""
        if not self.db_path:
            logger.warning("No database path configured")
            return

        self._stop_event.clear()
        self._persist_thread = threading.Thread(
            target=self._persistence_loop,
            daemon=True,
            name="SensorPersist"
        )
        self._persist_thread.start()
        logger.info("Started persistence thread")

    def stop_persistence(self) -> None:
        """Stop persistence thread and flush remaining data."""
        self._stop_event.set()
        if self._persist_thread:
            self._persist_thread.join(timeout=5.0)
            self._persist_thread = None

        # Flush remaining data
        self._flush_to_database()
        logger.info("Stopped persistence thread")

    def _persistence_loop(self) -> None:
        """Background persistence loop."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.persist_interval_seconds)
            self._flush_to_database()

    def _flush_to_database(self) -> None:
        """Write pending data to database."""
        with self._pending_lock:
            if not self._pending_writes:
                return
            to_write = self._pending_writes
            self._pending_writes = []

        if not self._db:
            return

        with self._db_lock:
            try:
                cursor = self._db.cursor()
                for record in to_write:
                    for channel, value in record["values"].items():
                        cursor.execute(
                            "INSERT INTO sensor_data (timestamp, channel, value) VALUES (?, ?, ?)",
                            (record["timestamp"], channel, value)
                        )
                self._db.commit()
                logger.debug(f"Persisted {len(to_write)} records")
            except Exception as e:
                logger.error(f"Failed to persist data: {e}")

    def query_historical(
        self,
        channel: str,
        start: datetime,
        end: datetime,
        resample_seconds: Optional[float] = None,
    ) -> Tuple[List[datetime], List[float]]:
        """
        Query historical data from database.

        Args:
            channel: Channel name
            start: Start time
            end: End time
            resample_seconds: Optional resampling interval

        Returns:
            Tuple of (timestamps, values)
        """
        if not self._db:
            raise RuntimeError("No database configured")

        with self._db_lock:
            cursor = self._db.execute(
                """
                SELECT timestamp, value FROM sensor_data
                WHERE channel = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                """,
                (channel, start.isoformat(), end.isoformat())
            )
            rows = cursor.fetchall()

        timestamps = [datetime.fromisoformat(row[0]) for row in rows]
        values = [row[1] for row in rows]

        # Resample if requested
        if resample_seconds and timestamps:
            timestamps, values = self._resample(
                timestamps, values, resample_seconds
            )

        return timestamps, values

    def _resample(
        self,
        timestamps: List[datetime],
        values: List[float],
        interval_seconds: float,
    ) -> Tuple[List[datetime], List[float]]:
        """Resample time series to fixed interval."""
        if not timestamps:
            return [], []

        start = timestamps[0]
        end = timestamps[-1]

        # Create time bins
        num_bins = int((end - start).total_seconds() / interval_seconds) + 1
        resampled_ts = []
        resampled_vals = []

        for i in range(num_bins):
            bin_start = start + timedelta(seconds=i * interval_seconds)
            bin_end = bin_start + timedelta(seconds=interval_seconds)

            # Find values in this bin
            bin_values = [
                v for t, v in zip(timestamps, values)
                if bin_start <= t < bin_end
            ]

            if bin_values:
                resampled_ts.append(bin_start)
                resampled_vals.append(np.mean(bin_values))

        return resampled_ts, resampled_vals

    def export_csv(
        self,
        filepath: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        channels: Optional[List[str]] = None,
    ) -> int:
        """
        Export data to CSV file.

        Args:
            filepath: Output file path
            start: Start time
            end: End time
            channels: Channels to export

        Returns:
            Number of rows exported
        """
        if channels is None:
            channels = self.CHANNELS

        # Get data from all channels
        all_data = {}
        timestamps = set()

        for channel in channels:
            ts, vals = self.get_channel(channel, start, end)
            for t, v in zip(ts, vals):
                timestamps.add(t)
                if t not in all_data:
                    all_data[t] = {}
                all_data[t][channel] = v

        # Sort by timestamp
        sorted_timestamps = sorted(timestamps)

        # Write CSV
        with open(filepath, 'w') as f:
            # Header
            f.write("timestamp," + ",".join(channels) + "\n")

            # Data rows
            for ts in sorted_timestamps:
                row = [ts.isoformat()]
                for channel in channels:
                    val = all_data.get(ts, {}).get(channel, "")
                    row.append(str(val) if val != "" else "")
                f.write(",".join(row) + "\n")

        logger.info(f"Exported {len(sorted_timestamps)} rows to {filepath}")
        return len(sorted_timestamps)

    def close(self) -> None:
        """Close data store and cleanup."""
        self.stop_persistence()
        if self._db:
            self._db.close()
            self._db = None
