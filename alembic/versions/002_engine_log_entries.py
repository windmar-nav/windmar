"""Engine log entries table.

Revision ID: 002_engine_log
Revises: 001_initial
Create Date: 2026-02-15

Adds engine_log_entries table for parsed engine log Excel workbook data.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002_engine_log"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "engine_log_entries",
        sa.Column("id", postgresql.UUID(), server_default=sa.text("uuid_generate_v4()"), primary_key=True),
        sa.Column("vessel_id", postgresql.UUID(), sa.ForeignKey("vessel_specs.id"), nullable=True),
        # Navigation
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("lapse_hours", sa.Float(), nullable=True),
        sa.Column("place", sa.String(255), nullable=True),
        sa.Column("event", sa.String(100), nullable=True),
        # ME Operational
        sa.Column("rpm", sa.Float(), nullable=True),
        sa.Column("engine_distance", sa.Float(), nullable=True),
        sa.Column("speed_stw", sa.Float(), nullable=True),
        sa.Column("me_power_kw", sa.Float(), nullable=True),
        sa.Column("me_load_pct", sa.Float(), nullable=True),
        sa.Column("me_fuel_index_pct", sa.Float(), nullable=True),
        sa.Column("shaft_power", sa.Float(), nullable=True),
        sa.Column("shaft_torque_knm", sa.Float(), nullable=True),
        sa.Column("slip_pct", sa.Float(), nullable=True),
        # HFO Consumption (MT)
        sa.Column("hfo_me_mt", sa.Float(), nullable=True),
        sa.Column("hfo_ae_mt", sa.Float(), nullable=True),
        sa.Column("hfo_boiler_mt", sa.Float(), nullable=True),
        sa.Column("hfo_total_mt", sa.Float(), nullable=True),
        # MGO Consumption (MT)
        sa.Column("mgo_me_mt", sa.Float(), nullable=True),
        sa.Column("mgo_ae_mt", sa.Float(), nullable=True),
        sa.Column("mgo_total_mt", sa.Float(), nullable=True),
        # Methanol
        sa.Column("methanol_me_mt", sa.Float(), nullable=True),
        # ROB
        sa.Column("rob_vlsfo_mt", sa.Float(), nullable=True),
        sa.Column("rob_mgo_mt", sa.Float(), nullable=True),
        sa.Column("rob_methanol_mt", sa.Float(), nullable=True),
        # Running Hours (period)
        sa.Column("rh_me", sa.Float(), nullable=True),
        sa.Column("rh_ae_total", sa.Float(), nullable=True),
        # Technical
        sa.Column("tc_rpm", sa.Float(), nullable=True),
        sa.Column("scav_air_press_bar", sa.Float(), nullable=True),
        sa.Column("fuel_temp_c", sa.Float(), nullable=True),
        sa.Column("sw_temp_c", sa.Float(), nullable=True),
        # Tracking
        sa.Column("upload_batch_id", postgresql.UUID(), nullable=False),
        sa.Column("source_sheet", sa.String(100), nullable=True),
        sa.Column("source_file", sa.String(500), nullable=True),
        # Metadata
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("extended_data", postgresql.JSONB(), nullable=True),
    )

    op.create_index("ix_engine_log_entries_timestamp", "engine_log_entries", ["timestamp"])
    op.create_index("ix_engine_log_entries_vessel_id", "engine_log_entries", ["vessel_id"])
    op.create_index("ix_engine_log_entries_event", "engine_log_entries", ["event"])
    op.create_index("ix_engine_log_entries_batch_id", "engine_log_entries", ["upload_batch_id"])
    op.create_index("ix_engine_log_vessel_timestamp", "engine_log_entries", ["vessel_id", "timestamp"])


def downgrade() -> None:
    op.drop_index("ix_engine_log_vessel_timestamp", table_name="engine_log_entries")
    op.drop_index("ix_engine_log_entries_batch_id", table_name="engine_log_entries")
    op.drop_index("ix_engine_log_entries_event", table_name="engine_log_entries")
    op.drop_index("ix_engine_log_entries_vessel_id", table_name="engine_log_entries")
    op.drop_index("ix_engine_log_entries_timestamp", table_name="engine_log_entries")
    op.drop_table("engine_log_entries")
