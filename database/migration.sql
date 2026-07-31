-- =============================================================================
-- Migration: Create processing_jobs table for checkpoint / resume support.
--
-- Run this once in your Supabase SQL Editor (Dashboard > SQL Editor).
-- After running, the app will automatically switch to the PostgreSQL backend.
-- If the table doesn't exist, the app falls back to a JSON file at
--   database/processing_jobs.json
-- =============================================================================

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id             UUID PRIMARY KEY,
    pdf_name           TEXT NOT NULL,
    storage_path       TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'UPLOADED',
    current_stage      TEXT NOT NULL DEFAULT 'UPLOADED',
    total_chunks       INTEGER DEFAULT 0,
    last_processed_chunk INTEGER DEFAULT 0,
    error_message      TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_status
    ON processing_jobs(status);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_pdf_name
    ON processing_jobs(pdf_name);

-- Auto-update updated_at on every row change.
CREATE OR REPLACE FUNCTION update_processing_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_processing_jobs_updated_at ON processing_jobs;
CREATE TRIGGER trg_processing_jobs_updated_at
    BEFORE UPDATE ON processing_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_processing_jobs_updated_at();
