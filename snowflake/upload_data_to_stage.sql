-- ============================================================================
-- snow9 — stage setup for OFFLINE Snowflake training
--
-- SECTION A (below): run anywhere — Snowsight worksheet, the notebook, or a
--   client. Safe to re-run (all IF NOT EXISTS). The Snowflake notebook does
--   this itself, so running this file is OPTIONAL if you use the notebook.
--
-- SECTION B (the PUT commands in comments): MUST run from a client that can
--   see the files (SnowSQL / Snowflake CLI on the machine with the dataset),
--   because Snowflake itself has no internet. Alternative for a quick test:
--   Snowsight -> Data -> Databases -> SNOW9 -> PUBLIC -> Stages ->
--   BTC_SNOW9_STAGE -> "+ Files" and drop the zip + a few .parquet shards.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS SNOW9;
USE DATABASE SNOW9;
USE SCHEMA PUBLIC;

CREATE STAGE IF NOT EXISTS BTC_SNOW9_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'BTCUSDT 1-second klines 2020..today (offline upload only)';

-- Run these from SnowSQL on the machine that has the data (NOT in the notebook):
--
--   PUT file:///path/to/pridictior_snow9/data/raw/*.parquet
--       @SNOW9.PUBLIC.BTC_SNOW9_STAGE/data/
--       AUTO_COMPRESS = FALSE PARALLEL = 8 OVERWRITE = TRUE;
--
--   PUT file:///path/to/pridictior_snow9.zip
--       @SNOW9.PUBLIC.BTC_SNOW9_STAGE/code/
--       AUTO_COMPRESS = FALSE OVERWRITE = TRUE;
--
-- Verify afterwards:
--   LIST @SNOW9.PUBLIC.BTC_SNOW9_STAGE/data/;
