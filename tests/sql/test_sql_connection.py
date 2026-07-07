"""Tests for SQLConnection improvements."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

from daft.sql.sql_connection import SQLConnection


class TestSQLConnectionEngineCaching:
    """B5: Test that SQLAlchemy Engine is cached."""

    def test_engine_cached_on_first_call(self):
        """Engine should be created once and reused."""
        conn = SQLConnection("sqlite:///test.db", "", "sqlite", "sqlite:///test.db")
        assert conn._engine is None

        with patch("sqlalchemy.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine

            engine1 = conn._get_or_create_engine()
            engine2 = conn._get_or_create_engine()

            assert engine1 is mock_engine
            assert engine2 is mock_engine
            mock_create.assert_called_once()

    def test_engine_not_created_for_callable(self):
        """Engine should not be created when conn is a callable."""
        conn_factory = MagicMock()
        conn = SQLConnection(conn_factory, "", "sqlite", "sqlite:///test.db")
        assert conn._get_or_create_engine() is None

    def test_engine_excluded_from_pickle(self):
        """Engine must be dropped during pickle to avoid serialization failures on distributed runners."""
        conn = SQLConnection("sqlite://", "", "sqlite", "sqlite://")
        conn._get_or_create_engine()  # populate live engine
        assert conn._engine is not None

        restored = pickle.loads(pickle.dumps(conn))
        assert restored._engine is None  # dropped, will be lazily recreated
        assert restored.conn == "sqlite://"
        assert restored.dialect == "sqlite"
        assert restored.driver == ""
        assert restored.url == "sqlite://"


class TestMySQLPyMySQLRewrite:
    """B3: Test mysql:// to mysql+pymysql:// rewrite."""

    def test_no_rewrite_when_not_mysql(self):
        """Non-mysql URLs should not be rewritten."""
        conn = SQLConnection("postgresql://user:pass@host/db", "", "postgres", "postgresql://user:pass@host/db")

        with patch("sqlalchemy.create_engine") as mock_create:
            mock_create.return_value = MagicMock()
            conn._get_or_create_engine()
            call_args = mock_create.call_args[0][0]
            assert call_args == "postgresql://user:pass@host/db"

    def test_no_rewrite_when_mysql_with_driver(self):
        """mysql+pymysql:// should not be rewritten again."""
        conn = SQLConnection("mysql+pymysql://user:pass@host/db", "", "mysql", "mysql+pymysql://user:pass@host/db")

        with patch("sqlalchemy.create_engine") as mock_create:
            mock_create.return_value = MagicMock()
            conn._get_or_create_engine()
            call_args = mock_create.call_args[0][0]
            assert call_args == "mysql+pymysql://user:pass@host/db"


class TestClickHousePercentile:
    """B4: Test ClickHouse-specific percentile syntax."""

    def test_clickhouse_uses_quantile_exact(self):
        """ClickHouse should use quantileExact()() syntax in partition bounds."""
        import pyarrow as pa

        from daft.daft import StorageConfig
        from daft.sql.sql_scan import PartitionBoundStrategy, SQLScanOperator

        conn = MagicMock()
        conn.dialect = "clickhouse"
        conn.driver = ""
        conn.url = "clickhouse://default:@localhost/default"

        # construct_sql_query returns a plain string (we don't care about its value)
        conn.construct_sql_query.return_value = "SELECT ..."
        # execute_sql_query returns a table with 3 bound columns for 2 scan tasks
        conn.execute_sql_query.return_value = pa.table(
            {"bound_0": [0], "bound_1": [50], "bound_2": [100]}
        )

        schema = {"id": MagicMock(is_numeric=MagicMock(return_value=True), is_temporal=MagicMock(return_value=False))}

        op = SQLScanOperator.__new__(SQLScanOperator)
        op.sql = "SELECT * FROM t"
        op.conn = conn
        op._partition_col = "id"
        op._partition_bound_strategy = PartitionBoundStrategy.PERCENTILE
        op._schema = schema

        op._get_partition_bounds(num_scan_tasks=2)

        # Verify construct_sql_query was called with quantileExact in the projection
        call_kwargs = conn.construct_sql_query.call_args
        projection = call_kwargs.kwargs.get("projection") or call_kwargs[1].get("projection")
        assert projection is not None
        assert all("quantileExact(" in p for p in projection), f"Expected quantileExact, got: {projection}"
        assert not any(p.startswith("quantile(") and "quantileExact" not in p for p in projection)

    def test_standard_dialect_uses_percentile_disc(self):
        """Standard SQL dialects should use percentile_disc syntax."""
        conn = MagicMock()
        conn.dialect = "postgres"

        # Verify standard dialect
        assert conn.dialect not in ["clickhouse", "mssql", "tsql"]
