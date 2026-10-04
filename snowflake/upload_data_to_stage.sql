-- ============================================================================
-- snow9 — stage setup for OFFLINE Snowflake training
--
-- Snowflake has NO internet in this setup, so the dataset must be pushed from
-- a machine that does. Steps:
--   1) With internet:   scripts/01_download_data.sh        (creates data/raw/*.parquet)
--   2) From SnowSQL / Snowflake CLI on that machine:  run the PUTs below
--   3) In the Snowflake Notebook: use the @SNOW9.PUBLIC.BTC_SNOW9_STAGE stage
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
