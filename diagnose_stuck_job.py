"""
Script to diagnose stuck job in FINALIZING status
"""
import sys
import uuid
from src.database import database_session
from src.models import Job, Segment, Paragraph, DiarizationSegment
from sqlalchemy import text
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for better readability
)
logger = logging.getLogger(__name__)

JOB_ID = "88e3a6cc-37c3-4ff0-b023-4051c347cb72"

def diagnose_job():
    """Diagnose why a job is stuck in FINALIZING"""

    with database_session() as db:
        try:
            # 1. Check job status
            print("=" * 80)
            print(f"DIAGNOSING JOB: {JOB_ID}")
            print("=" * 80)

            job = db.query(Job).filter(Job.id == uuid.UUID(JOB_ID)).first()

            if not job:
                logger.error(f"Job {JOB_ID} not found!")
                return

            print(f"\n1. JOB STATUS:")
            print(f"   - Status: {job.status}")
            print(f"   - Original filename: {job.original_filename}")
            print(f"   - Is post processed: {job.is_post_processed}")
            print(f"   - Error message: {job.error_message}")
            print(f"   - Created at: {job.created_at}")
            print(f"   - Updated at: {job.updated_at}")

            # 2. Check segments count
            segments_count = db.query(Segment).filter(Segment.job_id == job.id).count()
            print(f"\n2. SEGMENTS:")
            print(f"   - Total segments: {segments_count}")

            if segments_count > 0:
                # Show first 3 segments
                segments = db.query(Segment).filter(Segment.job_id == job.id).limit(3).all()
                print(f"   - First 3 segments:")
                for seg in segments:
                    print(f"     * Speaker: {seg.speaker}, Text: {seg.text[:50]}...")

            # 3. Check paragraphs count
            paragraphs_count = db.query(Paragraph).filter(Paragraph.job_id == job.id).count()
            print(f"\n3. PARAGRAPHS:")
            print(f"   - Total paragraphs: {paragraphs_count}")

            if paragraphs_count > 0:
                # Show first 3 paragraphs
                paragraphs = db.query(Paragraph).filter(Paragraph.job_id == job.id).limit(3).all()
                print(f"   - First 3 paragraphs:")
                for para in paragraphs:
                    print(f"     * Speaker: {para.speaker}, Original: {para.original_text[:50] if para.original_text else 'None'}...")
                    print(f"       Formatted: {para.formatted_text[:50] if para.formatted_text else 'None'}...")

            # 4. Check diarization segments
            diarization_count = db.query(DiarizationSegment).filter(
                DiarizationSegment.job_id == job.id
            ).count()
            print(f"\n4. DIARIZATION SEGMENTS:")
            print(f"   - Total diarization segments: {diarization_count}")

            # Check how many are processed
            processed_count = db.query(DiarizationSegment).filter(
                DiarizationSegment.job_id == job.id,
                DiarizationSegment.status == "PROCESSED"
            ).count()
            print(f"   - Processed diarization segments: {processed_count}")

            # 5. Try to execute the stored procedure
            print(f"\n5. TESTING STORED PROCEDURE:")
            try:
                query = f"SELECT * FROM dbo.GetGroupedTurnsByJob('{JOB_ID}') ORDER BY turn_id"
                result = db.execute(text(query))

                turns = list(result)
                print(f"   - Stored procedure returned {len(turns)} grouped turns")

                if len(turns) > 0:
                    print(f"   - First turn:")
                    first_turn = turns[0]
                    print(f"     * Turn ID: {first_turn.turn_id}")
                    print(f"     * Speaker: {first_turn.speaker}")
                    print(f"     * Segment count: {first_turn.segment_count}")
                    print(f"     * Text preview: {first_turn.combined_text[:100] if first_turn.combined_text else 'None'}...")
                else:
                    print("   - ⚠️  Stored procedure returned no turns!")

            except Exception as sp_error:
                print(f"   - ❌ ERROR executing stored procedure: {sp_error}")

            # 6. Summary and recommendations
            print(f"\n6. DIAGNOSIS SUMMARY:")
            print("=" * 80)

            if job.status == "FINALIZING":
                print("⚠️  Job is stuck in FINALIZING status")

                if segments_count == 0:
                    print("⚠️  No segments found - job may have failed during transcription")
                elif paragraphs_count == 0:
                    print("⚠️  No paragraphs created - likely failed during paragraph grouping")
                    print("💡 Recommendation: Check if stored procedure GetGroupedTurnsByJob is working")
                else:
                    print("✓ Job appears to have completed but status wasn't updated")
                    print("💡 Recommendation: Manually update job status to COMPLETED")
            else:
                print(f"✓ Job status is {job.status}")

            print("=" * 80)

        except Exception as e:
            logger.error(f"Error during diagnosis: {e}", exc_info=True)

if __name__ == "__main__":
    diagnose_job()