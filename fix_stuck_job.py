"""
Script to fix stuck job by manually creating paragraphs and updating status
"""
import sys
import uuid
from src.database import database_session
from src.models import Job, Segment, Paragraph, DiarizationSegment, JobStatus
from src.tasks import save_grouped_turns_to_paragraphs, mark_diarization_segments_as_processed
from sqlalchemy import text

JOB_ID = "88e3a6cc-37c3-4ff0-b023-4051c347cb72"

def fix_stuck_job():
    """Manually fix stuck job by creating paragraphs and updating status"""

    print("="*80)
    print(f"FIXING STUCK JOB: {JOB_ID}")
    print("="*80)

    with database_session() as db:
        try:
            # 1. Fetch job
            job = db.query(Job).filter(Job.id == uuid.UUID(JOB_ID)).first()

            if not job:
                print(f"ERROR: Job {JOB_ID} not found!")
                return

            print(f"\n1. Job Status: {job.status}")
            print(f"   Original filename: {job.original_filename}")

            # 2. Check if paragraphs already exist
            existing_paragraphs = db.query(Paragraph).filter(Paragraph.job_id == job.id).count()
            print(f"\n2. Existing paragraphs: {existing_paragraphs}")

            if existing_paragraphs > 0:
                print("   Job already has paragraphs. Updating status only...")
                job.is_post_processed = True
                job.status = JobStatus.COMPLETED
                db.commit()
                print("   Status updated to COMPLETED")
                return

            # 3. Create paragraphs
            print("\n3. Creating paragraphs from grouped turns...")
            try:
                paragraphs_created = save_grouped_turns_to_paragraphs(db, JOB_ID)
                print(f"   Successfully created {paragraphs_created} paragraphs")
            except Exception as para_error:
                print(f"   ERROR creating paragraphs: {para_error}")
                import traceback
                traceback.print_exc()
                return

            # 4. Mark diarization segments as processed
            print("\n4. Marking diarization segments as processed...")
            try:
                processed_count = mark_diarization_segments_as_processed(db, job.id)
                print(f"   Marked {processed_count} diarization segments as processed")
            except Exception as diaz_error:
                print(f"   ERROR marking diarization segments: {diaz_error}")

            # 5. Update job status
            print("\n5. Updating job status to COMPLETED...")
            job.is_post_processed = True
            job.status = JobStatus.COMPLETED
            db.commit()

            print("\n" + "="*80)
            print("SUCCESS! Job fixed successfully")
            print("="*80)
            print(f"  - Total paragraphs: {paragraphs_created}")
            print(f"  - Status: {job.status}")
            print(f"  - Is post processed: {job.is_post_processed}")
            print("="*80)

        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()

if __name__ == "__main__":
    fix_stuck_job()